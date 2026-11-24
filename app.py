import io
import json
import re
import threading
from pathlib import Path

import numpy as np
import torch
from executorch.runtime import Runtime
from flask import Flask, jsonify, render_template, request
from optimum.executorch import ExecuTorchModelForCausalLM
from PIL import Image
from transformers import AutoTokenizer

from knowledge_base import DEFAULT_REGION_ID, PaddyKnowledgeBase
from translator import OfflineArgosTranslator

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max image upload

# Use all 4 Cortex-A76 cores on Raspberry Pi 5
torch.set_num_threads(4)

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
VISION_PTE_PATH = BASE_DIR / "efficientnet_b4_paddy.pte"
LABELS_PATH = BASE_DIR / "class_names.json"
ADVISORY_PATH = BASE_DIR / "paddy_regional_advisories_india.json"
LLAMA_ET_DIR = BASE_DIR / "llama32_1b_et"

IMG_SIZE = 380
RESIZE_SHORT_EDGE = int(IMG_SIZE * 1.05)  # 399, matches eval_transforms
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MAX_CONTEXT_LEN = 2048  # Must match --max_seq_len used during Llama export
MAX_NEW_TOKENS_EN_INITIAL = 520  # Full room for all 4 diagnosis sections in English
MAX_NEW_TOKENS_EN_CHAT = 260     # Follow-up Q&A English budget

DEFAULT_LANGUAGE = "en"

SUPPORTED_LANGUAGES = [
    {"code": "en", "english_name": "English", "native_name": "English"},
    {"code": "hi", "english_name": "Hindi", "native_name": "हिंदी"},
    {"code": "bn", "english_name": "Bengali", "native_name": "বাংলা"},
]

LANGUAGE_SECTION_MARKERS = {
    "en": "",
    "hi": "\n\nहिंदी (Hindi):\n",
    "bn": "\n\nবাংলা (Bengali):\n",
}

ANY_TRANSLATED_SECTION_RE = re.compile(r"\n\n(?:हिंदी \(Hindi\)|বাংলা \(Bengali\)):\n")

# Pre-saved Bengali disease names (English & Hindi come from paddy_regional_advisories_india.json)
DISEASE_NAMES_BN = {
    "bacterial_leaf_blight": "ব্যাকটেরিয়াজনিত পাতা পোড়া রোগ (বিএলবি)",
    "bacterial_leaf_streak": "ব্যাকটেরিয়াজনিত পাতার দাগ রোগ (বিএলএস)",
    "bacterial_panicle_blight": "ব্যাকটেরিয়াজনিত শীষ পোড়া রোগ (বিপিবি)",
    "blast": "ধানের ঝলসা / ব্লাস্ট রোগ",
    "brown_spot": "ধানের বাদামী দাগ রোগ",
    "dead_heart": "মরা ডিগ / ডেড হার্ট (মাজরা পোকা)",
    "downy_mildew": "ডাউনি মিলডিউ রোগ",
    "hispa": "ধানের পামরি / হিসপা পোকা",
    "tungro": "ধানের টুংরো ভাইরাস রোগ",
    "normal": "সুস্থ পাতা (স্বাভাবিক)",
}

# Pre-saved Bengali agro-climatic region names
REGION_NAMES_BN = {
    "national_default": "সর্বভারতীয় সাধারণ পরামর্শ (জাতীয় মানদণ্ড)",
    "zone_01_western_himalayan": "অঞ্চল ১: পশ্চিম হিমালয় অঞ্চল",
    "zone_02_eastern_himalayan": "অঞ্চল ২: পূর্ব হিমালয় অঞ্চল",
    "zone_03_lower_gangetic_plains": "অঞ্চল ৩: নিম্ন গাঙ্গেয় সমভূমি অঞ্চল",
    "zone_04_middle_gangetic_plains": "অঞ্চল ৪: মধ্য গাঙ্গেয় সমভূমি অঞ্চল",
    "zone_05_upper_gangetic_plains": "অঞ্চল ৫: উচ্চ গাঙ্গেয় সমভূমি অঞ্চল",
    "zone_06_trans_gangetic_plains": "অঞ্চল ৬: ট্রান্স-গাঙ্গেয় সমভূমি অঞ্চল",
    "zone_07_eastern_plateau_and_hills": "অঞ্চল ৭: পূর্ব মালভূমি ও পাহাড়ি অঞ্চল",
    "zone_08_central_plateau_and_hills": "অঞ্চল ৮: মধ্য মালভূমি ও পাহাড়ি অঞ্চল",
    "zone_09_western_plateau_and_hills": "অঞ্চল ৯: পশ্চিম মালভূমি ও পাহাড়ি অঞ্চল",
    "zone_10_southern_plateau_and_hills": "অঞ্চল ১০: দক্ষিণ মালভূমি ও পাহাড়ি অঞ্চল",
    "zone_11_east_coast_plains_and_hills": "অঞ্চল ১১: পূর্ব উপকূলীয় সমভূমি ও পাহাড়ি অঞ্চল",
    "zone_12_west_coast_plains_and_ghats": "অঞ্চল ১২: পশ্চিম উপকূলীয় সমভূমি ও ঘাট অঞ্চল",
    "zone_13_gujarat_plains_and_hills": "অঞ্চল ১৩: গুজরাট সমভূমি ও পাহাড়ি অঞ্চল",
    "zone_14_western_dry": "অঞ্চল ১৪: পশ্চিম শুষ্ক অঞ্চল",
    "zone_15_island_region": "অঞ্চল ১৫: দ্বীপাঞ্চল",
}

RISK_LABELS_BN = {
    "high": "এই অঞ্চলে উচ্চ ঝুঁকি",
    "moderate": "মাঝারি / মৌসুমি ঝুঁকি",
    "preventive": "প্রতিরোধমূলক / নিয়মিত যত্ন",
}

# Pre-saved webpage contents for English, Hindi, and Bengali (zero real-time translation)
UI_TEXT = {
    "en": {
        "page_title": "Paddy Leaf Disease Detector",
        "header_title": "Paddy Leaf Disease Detector",
        "header_subtitle": "Select your agro-climatic region, upload a rice leaf photo for instant ICAR/CIBRC remedies, and ask follow-up questions offline.",
        "panel1_heading": "1. Region & Leaf Inspection",
        "language_label": "Language",
        "region_label": "Agro-Climatic Region",
        "file_label": "Choose a leaf image",
        "predict_btn": "Check Leaf",
        "predict_btn_loading": "Checking...",
        "panel2_heading": "2. Agriculture Assistant (Llama 3.2 1B)",
        "welcome_message": (
            "Select your region and upload a paddy leaf image on the left for instant verified remedies, "
            "or ask any rice farming question below in English."
        ),
        "chat_placeholder": "Ask in English...",
        "send_btn": "Send",
        "states_label": "States",
        "seasons_label": "Seasons",
        "disease_detected": "Disease Detected",
        "confidence": "Confidence",
        "region": "Region",
        "region_updated": "Region updated to",
        "region_updated_hint": "(Re-check a leaf photo or ask a question for Llama 3.2 advice grounded in this region's knowledge base.)",
        "thinking_initial": "Llama 3.2 is generating regional remedies from the Knowledge Base...",
        "thinking_chat": "Generating regional reply...",
        "detected_prefix": "Detected",
        "checked_prefix": "Checked leaf",
        "network_error": "Network error while contacting the assistant.",
        "server_error": "Could not reach the local server.",
        "section_marker": "",
    },
    "hi": {
        "page_title": "धान पत्ती रोग पहचानकर्ता",
        "header_title": "धान पत्ती रोग पहचानकर्ता",
        "header_subtitle": "अपना कृषि-जलवायु क्षेत्र चुनें, तुरंत प्रमाणित ICAR/CIBRC उपचार के लिए धान की पत्ती की फोटो अपलोड करें, और ऑफ़लाइन सवाल पूछें।",
        "panel1_heading": "1. क्षेत्र और पत्ती की जांच",
        "language_label": "भाषा (Language)",
        "region_label": "कृषि-जलवायु क्षेत्र",
        "file_label": "पत्ती की फोटो चुनें",
        "predict_btn": "पत्ती जांचें",
        "predict_btn_loading": "जांच हो रही है...",
        "panel2_heading": "2. कृषि सहायक (Llama 3.2 1B)",
        "welcome_message": (
            "तुरंत प्रमाणित उपचार के लिए बाईं ओर अपना क्षेत्र चुनें और धान की पत्ती की फोटो अपलोड करें, "
            "या नीचे हिंदी में धान की खेती से जुड़ा कोई भी सवाल पूछें।"
        ),
        "chat_placeholder": "हिंदी में पूछें...",
        "send_btn": "भेजें",
        "states_label": "राज्य",
        "seasons_label": "मौसम",
        "disease_detected": "पहचाना गया रोग",
        "confidence": "विश्वसनीयता",
        "region": "क्षेत्र",
        "region_updated": "क्षेत्र चुना गया",
        "region_updated_hint": "(इस क्षेत्र के ज्ञानकोश के अनुसार सलाह के लिए पत्ती की फोटो फिर से जांचें या कोई सवाल पूछें।)",
        "thinking_initial": "लामा 3.2 क्षेत्रीय ज्ञानकोश से अंग्रेजी और हिंदी उपचार तैयार कर रहा है...",
        "thinking_chat": "क्षेत्रीय अंग्रेजी और हिंदी उत्तर तैयार हो रहा है...",
        "detected_prefix": "पहचाना गया रोग",
        "checked_prefix": "जांची गई पत्ती",
        "network_error": "सहायक से संपर्क करते समय नेटवर्क त्रुटि हुई।",
        "server_error": "स्थानीय सर्वर से संपर्क नहीं हो सका।",
        "section_marker": "\n\nहिंदी (Hindi):\n",
    },
    "bn": {
        "page_title": "ধান পাতার রোগ শনাক্তকারী",
        "header_title": "ধান পাতার রোগ শনাক্তকারী",
        "header_subtitle": "আপনার কৃষি-জলবায়ু অঞ্চল নির্বাচন করুন, তাৎক্ষণিক ICAR/CIBRC প্রতিকারের জন্য ধান পাতার ছবি আপলোড করুন এবং অফলাইনে প্রশ্ন করুন।",
        "panel1_heading": "১. অঞ্চল ও পাতা পরীক্ষা",
        "language_label": "ভাষা (Language)",
        "region_label": "কৃষি-জলবায়ু অঞ্চল",
        "file_label": "পাতার ছবি নির্বাচন করুন",
        "predict_btn": "পাতা পরীক্ষা করুন",
        "predict_btn_loading": "পরীক্ষা চলছে...",
        "panel2_heading": "২. কৃষি সহায়ক (Llama 3.2 1B)",
        "welcome_message": (
            "তাৎক্ষণিক যাচাইকৃত প্রতিকারের জন্য বাম দিকে আপনার অঞ্চল নির্বাচন করে ধান পাতার ছবি আপলোড করুন, "
            "অথবা নিচে বাংলায় ধান চাষ সংক্রান্ত যেকোনো প্রশ্ন জিজ্ঞাসা করুন।"
        ),
        "chat_placeholder": "বাংলায় জিজ্ঞাসা করুন...",
        "send_btn": "পাঠান",
        "states_label": "রাজ্য",
        "seasons_label": "মৌসুম",
        "disease_detected": "শনাক্তকৃত রোগ",
        "confidence": "নির্ভরযোগ্যতা",
        "region": "অঞ্চল",
        "region_updated": "অঞ্চল নির্বাচিত হয়েছে",
        "region_updated_hint": "(এই অঞ্চলের জ্ঞানভাণ্ডার থেকে পরামর্শের জন্য পাতার ছবি পুনরায় পরীক্ষা করুন বা প্রশ্ন করুন।)",
        "thinking_initial": "Llama 3.2 আঞ্চলিক জ্ঞানভাণ্ডার থেকে ইংরেজি ও বাংলা প্রতিকার তৈরি করছে...",
        "thinking_chat": "আঞ্চলিক ইংরেজি ও বাংলা উত্তর তৈরি হচ্ছে...",
        "detected_prefix": "শনাক্তকৃত রোগ",
        "checked_prefix": "পরীক্ষিত পাতা",
        "network_error": "সহায়কের সাথে যোগাযোগে নেটওয়ার্ক ত্রুটি হয়েছে।",
        "server_error": "স্থানীয় সার্ভারের সাথে সংযোগ করা যায়নি।",
        "section_marker": "\n\nবাংলা (Bengali):\n",
    },
}

ENGLISH_SYSTEM_PROMPT = (
    "You are a practical agriculture assistant for Indian rice (paddy) farmers. "
    "ALWAYS answer ONLY in plain, simple English. "
    "Keep each point short, clear, and complete so no section is cut off. "
    "You MUST answer strictly using the RETRIEVED KNOWLEDGE BASE below as your source of truth. "
    "Keep exact chemical names, dosages, and recommended regional rice varieties from the Knowledge Base. "
    "NEVER invent chemical dosages, NEVER recommend banned pesticides (like monocrotophos) or synthetic pyrethroids, "
    "and NEVER advise routine antibiotic sprays (streptomycin/tetracycline) for bacterial diseases. "
    "If a detail is not in the Knowledge Base, tell the farmer to confirm with their nearest Krishi Vigyan Kendra (KVK)."
)


def resolve_language(lang_code: str | None) -> str:
    if lang_code in ("en", "hi", "bn"):
        return lang_code
    return DEFAULT_LANGUAGE


def get_bn_risk(risk_en: str) -> str:
    lowered = (risk_en or "").lower()
    if "high" in lowered:
        return RISK_LABELS_BN["high"]
    if "moderate" in lowered or "seasonal" in lowered:
        return RISK_LABELS_BN["moderate"]
    if "preventive" in lowered or "routine" in lowered:
        return RISK_LABELS_BN["preventive"]
    return risk_en


# --- 1. Load Labels & Offline Regional Knowledge Base (RAG) ---
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

print(f"Building offline regional Knowledge Base from: {ADVISORY_PATH}")
KB = PaddyKnowledgeBase(ADVISORY_PATH)
print(f"Knowledge Base indexed {len(KB.chunks)} chunks across {len(KB.regions)} regions.")

# Attach pre-saved Bengali region names for template rendering (without modifying knowledge_base.py)
REGIONS_FOR_UI = [
    {**r, "name_bn": REGION_NAMES_BN.get(r["id"], r["name_en"])}
    for r in KB.sorted_regions
]

# --- 2. Load Offline English <-> Hindi / Bengali Translator (Argos Translate / CTranslate2) ---
print("Loading offline English <-> Hindi / Bengali translator (Argos Translate)...")
TRANSLATOR = OfflineArgosTranslator()
print("Offline Argos translator (EN / HI / BN) loaded and ready.")

# --- 3. Load EfficientNet-B4 on ExecuTorch ---
print(f"Loading vision model via ExecuTorch: {VISION_PTE_PATH}")
et_runtime = Runtime.get()
vision_program = et_runtime.load_program(VISION_PTE_PATH)
vision_forward = vision_program.load_method("forward")
vision_lock = threading.Lock()

# --- 4. Load Llama 3.2 1B Instruct on ExecuTorch (English-Only Reasoning) ---
print(f"Loading Llama model via ExecuTorch: {LLAMA_ET_DIR}")
llama_tokenizer = AutoTokenizer.from_pretrained(str(LLAMA_ET_DIR), local_files_only=True)
llama_model = ExecuTorchModelForCausalLM.from_pretrained(str(LLAMA_ET_DIR), export=False)
llama_lock = threading.Lock()
print("Vision model, Llama 3.2 (English), RAG Knowledge Base, and offline translator are ready.")


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize shorter edge to 399 while preserving aspect ratio
    w, h = img.size
    if w < h:
        new_w = RESIZE_SHORT_EDGE
        new_h = int(round(h * (RESIZE_SHORT_EDGE / w)))
    else:
        new_h = RESIZE_SHORT_EDGE
        new_w = int(round(w * (RESIZE_SHORT_EDGE / h)))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Center crop to 380x380
    left = (new_w - IMG_SIZE) // 2
    top = (new_h - IMG_SIZE) // 2
    img = img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))

    # Normalize and convert HWC -> NCHW contiguous float32 tensor
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.ascontiguousarray(np.expand_dims(arr, axis=0), dtype=np.float32)
    return torch.from_numpy(arr)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _norm_for_dedup(text: str) -> str:
    return re.sub(r"[\s.!?,;:\-–—*#]+", " ", text).strip().lower()


def clean_repetition_loops(text: str) -> str:
    """Trim any hallucinated chat turn markers or repeated lines from Llama English output."""
    if not text:
        return ""

    for turn_marker in (
        "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>",
        "<|start_header_id|>", "\nuser\n", "\nUser:", "\nAssistant:",
    ):
        if turn_marker in text:
            text = text.split(turn_marker, 1)[0]

    kept_lines: list[str] = []
    seen_substantial: set[str] = set()
    prev_norm = ""

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue

        norm = _norm_for_dedup(stripped)
        if norm and (norm == prev_norm or (len(norm) >= 16 and norm in seen_substantial)):
            break

        kept_lines.append(stripped)
        if norm:
            seen_substantial.add(norm)
            prev_norm = norm

    return "\n".join(kept_lines).strip()


def strip_prompt_from_output(
    raw_output: str, prompt: str, prompt_ids: list, latest_user_text: str
) -> str:
    """Remove echoed system/chat prompt from ExecuTorch text_generation output."""
    reply = None

    if raw_output.startswith(prompt):
        reply = raw_output[len(prompt):]

    if reply is None:
        decoded_candidates = [
            llama_tokenizer.decode(prompt_ids, skip_special_tokens=True),
            llama_tokenizer.decode(
                llama_tokenizer.encode(prompt, add_special_tokens=True),
                skip_special_tokens=True,
            ),
        ]
        for prefix in decoded_candidates:
            if prefix and raw_output.startswith(prefix):
                reply = raw_output[len(prefix):]
                break
            stripped_prefix = prefix.strip()
            if stripped_prefix and raw_output.startswith(stripped_prefix):
                reply = raw_output[len(stripped_prefix):]
                break

    if reply is None:
        special_marker = "<|start_header_id|>assistant<|end_header_id|>"
        if special_marker in raw_output:
            reply = raw_output.rsplit(special_marker, 1)[-1]
        elif latest_user_text and f"{latest_user_text}assistant" in raw_output:
            reply = raw_output.rsplit(f"{latest_user_text}assistant", 1)[-1]
        elif "assistant\n\n" in raw_output:
            reply = raw_output.rsplit("assistant\n\n", 1)[-1]
        else:
            reply = raw_output

    for stop_tok in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"):
        reply = reply.replace(stop_tok, "")

    return reply.strip()


def run_llama_turn(system_prompt: str, messages: list, max_new_tokens: int) -> str:
    """Format English chat within context budget, run ExecuTorch Llama once, return clean English."""
    max_prompt_tokens = MAX_CONTEXT_LEN - max_new_tokens - 16
    trimmed = [dict(m) for m in messages]

    while True:
        convo = [{"role": "system", "content": system_prompt}] + trimmed
        prompt = llama_tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = llama_tokenizer.encode(prompt, add_special_tokens=False)

        if len(prompt_ids) <= max_prompt_tokens or len(trimmed) <= 1:
            break
        trimmed.pop(0)

    target_seq_len = min(MAX_CONTEXT_LEN, len(prompt_ids) + max_new_tokens)

    with llama_lock:
        raw_output = llama_model.text_generation(
            tokenizer=llama_tokenizer,
            prompt=prompt,
            max_seq_len=target_seq_len,
        )

    latest_user_text = trimmed[-1].get("content", "")
    extracted = strip_prompt_from_output(raw_output, prompt, prompt_ids, latest_user_text)
    return clean_repetition_loops(extracted)


def prepare_english_messages(
    messages: list,
    language: str = DEFAULT_LANGUAGE,
    is_initial_diagnosis: bool = False,
) -> tuple[list, str, str]:
    """
    Assume user input is in the selected language (no auto-detection):
    - If language == 'en': feed user query directly to Llama without translation.
    - If language in ('hi', 'bn'): translate user query from selected language -> English first.
    Internal English seed messages / initial auto-prompts marked already_english=True stay in English,
    and prior assistant messages retain only their English section.
    """
    lang = resolve_language(language)
    cleaned = []
    raw_latest_user = ""
    en_latest_user = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        already_english = bool(msg.get("already_english", False))

        if role == "assistant":
            split_match = ANY_TRANSLATED_SECTION_RE.search(content)
            if split_match:
                content = content[: split_match.start()]
            content = re.sub(r"^\s*English\s*:\s*", "", content, flags=re.IGNORECASE).strip()
        elif role == "user":
            raw_latest_user = content
            if not is_initial_diagnosis and not already_english:
                msg_lang = resolve_language(msg.get("lang") or lang)
                if msg_lang != "en":
                    content = TRANSLATOR.to_english(content, msg_lang)
            en_latest_user = content

        cleaned.append({"role": role, "content": content})

    if cleaned and cleaned[-1]["role"] == "user":
        if is_initial_diagnosis:
            cleaned[-1]["content"] = (
                cleaned[-1]["content"]
                + "\n\n(Answer strictly in simple English only using the Retrieved Knowledge Base. Keep every requested section brief and complete.)"
            )
        else:
            cleaned[-1]["content"] = (
                cleaned[-1]["content"]
                + "\n\n(Answer clearly in 2 to 5 simple English sentences using the Retrieved Knowledge Base.)"
            )

    return cleaned, raw_latest_user, en_latest_user


def generate_rag_reply(
    messages: list,
    region_id: str = DEFAULT_REGION_ID,
    language: str = DEFAULT_LANGUAGE,
    detected_disease: str | None = None,
    confidence: float | None = None,
    is_initial_diagnosis: bool = False,
) -> dict:
    lang = resolve_language(language)

    # Step 1: If non-English is selected, translate user query -> English before RAG & Llama
    en_messages, raw_latest_user, en_latest_user = prepare_english_messages(
        messages, language=lang, is_initial_diagnosis=is_initial_diagnosis
    )

    rag_query = (
        f"{raw_latest_user} {en_latest_user}".strip()
        if raw_latest_user != en_latest_user
        else en_latest_user
    )

    # Step 2: Retrieve grounded regional passages from the offline JSON Knowledge Base
    if is_initial_diagnosis and detected_disease:
        rag_result = KB.build_initial_diagnosis_context(
            region_id=region_id,
            raw_label=detected_disease,
            confidence=confidence,
        )
        max_en_tokens = MAX_NEW_TOKENS_EN_INITIAL
    else:
        rag_result = KB.retrieve(
            query=rag_query,
            region_id=region_id,
            detected_disease=detected_disease,
            top_k=3,
            max_chars=2400,
        )
        max_en_tokens = MAX_NEW_TOKENS_EN_CHAT

    kb_context = rag_result.get("context_text", "")
    kb_sources = rag_result.get("sources", [])

    en_system_prompt = ENGLISH_SYSTEM_PROMPT
    if kb_context:
        en_system_prompt += f"\n\nRETRIEVED KNOWLEDGE BASE:\n{kb_context}"

    # Step 3: Run Llama 3.2 1B ONCE — strictly in English
    english_reply = run_llama_turn(en_system_prompt, en_messages, max_en_tokens)
    english_reply = re.sub(r"^\s*English\s*:\s*", "", english_reply, flags=re.IGNORECASE).strip()

    # Step 4: If English is selected, display only Llama's original English output.
    # Otherwise, translate Llama's output with Argos Translate and display both English + translated output.
    if lang == "en":
        full_reply = english_reply
    else:
        translated_reply = TRANSLATOR.from_english(english_reply, lang)
        marker = LANGUAGE_SECTION_MARKERS[lang]
        full_reply = f"English:\n{english_reply}{marker}{translated_reply}"

    return {
        "reply": full_reply,
        "english_reply": english_reply,
        "english_user_text": en_latest_user,
        "language": lang,
        "kb_sources": kb_sources,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        regions=REGIONS_FOR_UI,
        languages=SUPPORTED_LANGUAGES,
        default_language=DEFAULT_LANGUAGE,
        ui_text=UI_TEXT,
    )


@app.route("/regions", methods=["GET"])
def get_regions():
    return jsonify({"regions": REGIONS_FOR_UI})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    region_id = request.form.get("region_id", DEFAULT_REGION_ID)

    try:
        tensor = preprocess_image(file.read())
        with vision_lock:
            et_outputs = vision_forward.execute((tensor,))

        logits = et_outputs[0].detach().cpu().numpy()[0]
        probs = softmax(logits)

        top_idx = int(np.argmax(probs))
        raw_label = CLASS_NAMES[top_idx]
        confidence = round(float(probs[top_idx]) * 100, 2)
        disease_key = KB.normalize_disease_key(raw_label)
        is_diseased = disease_key != "normal"

        resolved_region_id, region_meta, card = KB.get_regional_card(region_id, raw_label)
        label_en = card.get("name_en", raw_label.replace("_", " ").title())
        label_hi = card.get("name_hi", label_en)
        label_bn = DISEASE_NAMES_BN.get(disease_key, label_en)

        risk_en = card.get("regional_risk_en", "")
        risk_hi = card.get("regional_risk_hi", risk_en)
        risk_bn = get_bn_risk(risk_en)

        return jsonify({
            "raw_label": raw_label,
            "disease_key": disease_key,
            "label": label_en,
            "label_hi": label_hi,
            "label_bn": label_bn,
            "confidence": confidence,
            "is_diseased": is_diseased,
            "region_id": resolved_region_id,
            "region_name_en": region_meta["name_en"],
            "region_name_hi": region_meta["name_hi"],
            "region_name_bn": REGION_NAMES_BN.get(resolved_region_id, region_meta["name_en"]),
            "regional_risk_en": risk_en,
            "regional_risk_hi": risk_hi,
            "regional_risk_bn": risk_bn,
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {e}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    region_id = data.get("region_id", DEFAULT_REGION_ID)
    language = resolve_language(data.get("language", DEFAULT_LANGUAGE))
    detected_disease = data.get("detected_disease")
    confidence = data.get("confidence")
    is_initial_diagnosis = bool(data.get("is_initial_diagnosis", False))

    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    try:
        result = generate_rag_reply(
            messages,
            region_id=region_id,
            language=language,
            detected_disease=detected_disease,
            confidence=confidence,
            is_initial_diagnosis=is_initial_diagnosis,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Assistant error: {e}"}), 500


if __name__ == "__main__":
    # Accessible at http://localhost:5000 or http://<pi-ip>:5000 on your local network
    app.run(host="0.0.0.0", port=5000, debug=False)
