import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REGION_ID = "national_default"

# Direct, unambiguous water/irrigation rules derived strictly from each disease's
# ICAR/NRRI field_care_en guidance in paddy_regional_advisories_india.json.
# Written as a single clear stance so Llama 3.2 1B never outputs contradictory Yes/No lines.
DISEASE_WATER_RULES = {
    "bacterial_leaf_blight": (
        "Water & Irrigation Rule: NO — do NOT flood the field or keep standing water. "
        "Drain standing water immediately because deep water spreads Bacterial Leaf Blight rapidly, "
        "and keep the soil only lightly moist (alternate wetting and drying) without letting water flow into healthy plots."
    ),
    "bacterial_leaf_streak": (
        "Water & Irrigation Rule: NO — do NOT keep excess standing water. "
        "Drain excess standing water immediately after heavy rain or storms, and keep only light soil moisture."
    ),
    "bacterial_panicle_blight": (
        "Water & Irrigation Rule: YES — keep shallow, uniform moisture during flowering to reduce canopy heat, "
        "but avoid excessive deep flooding."
    ),
    "blast": (
        "Water & Irrigation Rule: YES — keep the soil evenly moist to avoid drought stress (especially in upland fields), "
        "without excessive deep flooding."
    ),
    "brown_spot": (
        "Water & Irrigation Rule: YES — maintain steady light soil moisture to prevent drought stress, "
        "especially during tillering and grain filling."
    ),
    "dead_heart": (
        "Water & Irrigation Rule: Avoid deep standing water; maintain only a thin 2–3 cm water film "
        "when applying granular insecticides."
    ),
    "downy_mildew": (
        "Water & Irrigation Rule: NO — do NOT flood nurseries or newly transplanted fields. "
        "Drain stagnant floodwater immediately because downy mildew spores swim and spread through standing water."
    ),
    "hispa": (
        "Water & Irrigation Rule: Maintain normal shallow water (2–5 cm) and keep marshy bunds clear of grassy weeds."
    ),
    "tungro": (
        "Water & Irrigation Rule: Maintain normal shallow water (2–5 cm) without waterlogging, "
        "and focus immediately on uprooting infected hills and controlling Green Leafhoppers."
    ),
    "normal": (
        "Water & Irrigation Rule: Follow Alternate Wetting and Drying (AWD) irrigation — maintain shallow 2–5 cm water "
        "during active flowering/panicle initiation and light soil moisture otherwise."
    ),
}

# English, Devanagari Hindi, and Romanized Hinglish disease keywords
DISEASE_KEYWORDS = {
    "bacterial_leaf_blight": [
        "bacterial leaf blight", "blb", "kresek", "xanthomonas",
        "पत्ती झुलसा", "जीवाणु झुलसा", "झुलसा", "patti jhulsa", "jhulsa",
    ],
    "bacterial_leaf_streak": [
        "bacterial leaf streak", "leaf streak", "bls", "oryzicola",
        "पत्ती धारी", "धारी रोग", "patti dhari", "dhari rog",
    ],
    "bacterial_panicle_blight": [
        "panicle blight", "burkholderia", "bpb", "chaffy", "empty grain",
        "बाली झुलसा", "bali jhulsa", "khokhla", "खोखला",
    ],
    "blast": [
        "blast", "neck blast", "leaf blast", "node blast", "magnaporthe", "pyricularia",
        "jhoka", "jhonka", "झोंका", "ब्लास्ट", "gardan jhoka", "गर्दन झोंका",
    ],
    "brown_spot": [
        "brown spot", "bipolaris", "helminthosporium", "sesame",
        "भूरा धब्बा", "ब्राउन स्पॉट", "bhura dhabba", "bhoora dhabba",
    ],
    "dead_heart": [
        "dead heart", "stem borer", "white ear", "scirpophaga",
        "तना छेदक", "डेड हार्ट", "गोभ सूखना", "सफेद बाली",
        "tana chedak", "tana chhedak", "gobh sukhna", "safed bali",
    ],
    "downy_mildew": [
        "downy mildew", "crazy top", "sclerophthora",
        "मृदुरोमिल", "डाउनी मिल्ड्यू", "आसिता",
    ],
    "hispa": [
        "hispa", "spiny beetle", "dicladispa",
        "हिस्पा", "कांटेदार भृंग", "kantedar",
    ],
    "tungro": [
        "tungro", "leafhopper", "glh", "nephotettix", "virus",
        "टुंगरो", "हरा फुदका", "hara phudka", "वायरस",
    ],
}

# Bilingual (Hindi/Hinglish/English -> focused KB terms) expansion for BM25 retrieval
BILINGUAL_QUERY_EXPANSIONS = {
    # Treatment / chemical / dosage
    "दवा": "chemical spray dosage fungicide insecticide cibrc",
    "दवाई": "chemical spray dosage fungicide insecticide cibrc",
    "छिड़काव": "spray chemical dosage fungicide insecticide",
    "स्प्रे": "spray chemical dosage",
    "फफूंदनाशक": "fungicide chemical spray tricyclazole mancozeb propiconazole",
    "कीटनाशक": "insecticide chemical spray chlorantraniliprole quinalphos",
    "मात्रा": "dosage litre acre gram ml chemical",
    "dawa": "chemical spray dosage fungicide insecticide cibrc",
    "dawai": "chemical spray dosage fungicide insecticide cibrc",
    "chhidkav": "spray chemical dosage fungicide insecticide",
    "spray": "spray chemical dosage fungicide insecticide",
    "medicine": "chemical spray dosage fungicide insecticide",
    "pesticide": "chemical insecticide fungicide cibrc regulatory safety",
    "antibiotic": "antibiotic streptomycin tetracycline regulatory safety copper",
    "एंटीबायोटिक": "antibiotic streptomycin tetracycline regulatory safety copper",
    # Organic / biocontrol
    "जैविक": "organic biocontrol pseudomonas trichoderma neem nske",
    "नीम": "organic neem azadirachtin nske biocontrol",
    "ऑर्गेनिक": "organic biocontrol pseudomonas trichoderma neem",
    "jaivik": "organic biocontrol pseudomonas trichoderma neem",
    "neem": "organic neem azadirachtin nske biocontrol",
    "organic": "organic biocontrol pseudomonas trichoderma neem compost",
    "natural": "organic biocontrol field care",
    # Field care / fertilizer / water / soil
    "खाद": "fertilizer urea nitrogen potash zinc soil field care",
    "यूरिया": "urea nitrogen top-dressing field care",
    "पोटाश": "potash mop soil field care",
    "जिंक": "zinc soil field care",
    "पानी": "irrigation drainage flooding standing moist field care",
    "सिंचाई": "irrigation drainage flooding standing moist field care",
    "जल": "irrigation drainage flooding standing moist field care",
    "मिट्टी": "soil lime acidic zinc potash region profile",
    "चूना": "lime acidic soil ph",
    "khad": "fertilizer urea nitrogen potash zinc soil field care",
    "urea": "urea nitrogen top-dressing field care",
    "pani": "irrigation drainage flooding standing moist field care",
    "water": "irrigation drainage flooding standing moist field care",
    "watering": "irrigation drainage flooding standing moist field care",
    "irrigate": "irrigation drainage flooding standing moist field care",
    "irrigation": "irrigation drainage flooding standing moist field care",
    "drain": "irrigation drainage flooding standing moist field care",
    "flood": "irrigation drainage flooding standing moist field care",
    "mitti": "soil lime acidic zinc potash",
    # Varieties & prevention
    "किस्म": "recommended varieties resistant prevention seed",
    "किस्में": "recommended varieties resistant prevention seed",
    "बीज": "seed treatment certified varieties prevention",
    "बचाव": "prevention resistant varieties seed treatment",
    "kism": "recommended varieties resistant prevention seed",
    "beej": "seed treatment certified varieties prevention",
    "variety": "recommended varieties resistant prevention seed",
    "varieties": "recommended varieties resistant prevention seed",
    "seed": "seed treatment certified varieties prevention",
    "prevent": "prevention resistant varieties seed treatment",
    # Symptoms & identification
    "लक्षण": "symptoms cause pathogen identification",
    "पहचान": "symptoms cause pathogen test identification",
    "कारण": "cause pathogen favorable conditions symptoms",
    "lakshan": "symptoms cause pathogen identification",
    "pehchan": "symptoms cause pathogen test identification",
    # Climate / season / region
    "मौसम": "climate season rainfall temperature region profile",
    "सीजन": "season kharif rabi boro aman region profile",
    "क्षेत्र": "region climate soil seasons institutes",
    "kvk": "escalation kvk institutes",
}

SAFETY_TRIGGER_TERMS = {
    "chemical", "spray", "fungicide", "insecticide", "pesticide", "dosage", "dose",
    "antibiotic", "streptomycin", "tetracycline", "monocrotophos", "carbofuran",
    "pyrethroid", "cypermethrin", "deltamethrin", "tricyclazole", "banned", "safe",
    "dawa", "dawai", "दवा", "दवाई", "छिड़काव", "कीटनाशक", "फफूंदनाशक", "एंटीबायोटिक",
}

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "in",
    "on", "for", "with", "and", "or", "my", "me", "i", "we", "you", "your", "it",
    "this", "that", "what", "how", "can", "do", "does", "should", "about", "from",
    "not", "crop", "crops", "crope", "plant", "plants", "paddy", "rice", "field",
    "क्या", "कैसे", "में", "के", "की", "का", "को", "है", "हैं", "करें", "करना", "मेरा", "मेरे",
}

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0900-\u097F]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


@dataclass
class KBChunk:
    chunk_id: str
    region_id: str
    disease_key: str | None
    section: str
    title: str
    text: str
    term_freqs: Counter
    length: int


class PaddyKnowledgeBase:
    """
    Zero-dependency offline BM25 + metadata-filtered Knowledge Base (RAG)
    built directly over paddy_regional_advisories_india.json.
    """

    def __init__(self, json_path: Path | str, k1: float = 1.5, b: float = 0.75):
        self.json_path = Path(json_path)
        self.k1 = k1
        self.b = b

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.db = json.load(f)

        self.regions = self.db.get("regions", {})
        self.advisories = self.db.get("regional_advisories", {})
        self.metadata = self.db.get("metadata", {})

        self.sorted_regions = sorted(
            [{"id": rid, **rmeta} for rid, rmeta in self.regions.items()],
            key=lambda r: r.get("zone_number", 999),
        )

        self.chunks: list[KBChunk] = []
        self.doc_freqs: Counter = Counter()
        self.avg_doc_len: float = 1.0
        self._build_index()

    def _add_chunk(
        self,
        chunk_id: str,
        region_id: str,
        disease_key: str | None,
        section: str,
        title: str,
        text: str,
        extra_search_terms: str = "",
    ) -> None:
        search_blob = f"{title} {text} {extra_search_terms}"
        tokens = tokenize(search_blob)
        tf = Counter(tokens)
        chunk = KBChunk(
            chunk_id=chunk_id,
            region_id=region_id,
            disease_key=disease_key,
            section=section,
            title=title,
            text=text.strip(),
            term_freqs=tf,
            length=max(len(tokens), 1),
        )
        self.chunks.append(chunk)
        for term in tf:
            self.doc_freqs[term] += 1

    def _build_index(self) -> None:
        # 1. Global regulatory safety chunk (CIBRC / ICAR-NIBSM / ICAR-NRRI guardrails)
        safety_en = self.metadata.get("regulatory_safety_notes", {}).get("en", "")
        if safety_en:
            self._add_chunk(
                chunk_id="global::regulatory_safety",
                region_id="global",
                disease_key=None,
                section="regulatory_safety",
                title="ICAR / CIBRC Regulatory Safety Rules",
                text=f"Regulatory Safety Rules: {safety_en}",
                extra_search_terms="antibiotic streptomycin tetracycline monocrotophos carbofuran pyrethroid bph banned pesticide safety",
            )

        # 2. Region profile chunks (16 regions)
        for region_id, rmeta in self.regions.items():
            states = ", ".join(rmeta.get("states_covered", []))
            seasons = ", ".join(rmeta.get("main_paddy_seasons", []))
            institutes = ", ".join(rmeta.get("key_institutes", []))
            risk_prof = rmeta.get("regional_risk_profile", {})
            high_risk = ", ".join(risk_prof.get("high_risk", []))
            soil_note = risk_prof.get("soil_note_en", "")
            water_regime = risk_prof.get("water_regime", "")

            profile_text = (
                f"Region: {rmeta['name_en']} ({states}). "
                f"Climate: {rmeta['climate_profile_en']} "
                f"Soil: {rmeta['soil_type_en']}. "
                f"Paddy Seasons: {seasons}. "
                f"Water Regime: {water_regime}. "
                f"High-Risk Diseases: {high_risk}. "
                f"Soil & Water Guidance: {soil_note} "
                f"Key Institutes: {institutes}."
            )
            self._add_chunk(
                chunk_id=f"{region_id}::region_profile",
                region_id=region_id,
                disease_key=None,
                section="region_profile",
                title=f"{rmeta['name_en']} — Regional Climate, Soil & Seasons",
                text=profile_text,
                extra_search_terms=f"{rmeta.get('name_hi', '')} {states} {seasons}",
            )

        # 3. Granular Disease Advisory Chunks (16 regions x 10 classes x 5 topical sections)
        for region_id, disease_map in self.advisories.items():
            rmeta = self.regions.get(region_id, {})
            r_name = rmeta.get("name_en", region_id)

            for disease_key, card in disease_map.items():
                d_en = card.get("name_en", disease_key)
                d_hi = card.get("name_hi", "")
                kw_hint = " ".join(DISEASE_KEYWORDS.get(disease_key, []))
                water_rule = DISEASE_WATER_RULES.get(disease_key, "")

                # Chunk A: Overview, Symptoms & Regional Risk
                overview_text = (
                    f"{d_en} ({card.get('category', '')}) in {r_name}. "
                    f"Regional Risk: {card.get('regional_risk_en', '')}. "
                    f"Cause/Pathogen: {card.get('pathogen', '')}. "
                    f"Symptoms: {card.get('symptoms_en', '')} "
                    f"Regional Context: {card.get('regional_context_en', '')}"
                )
                self._add_chunk(
                    chunk_id=f"{region_id}::{disease_key}::overview_symptoms",
                    region_id=region_id,
                    disease_key=disease_key,
                    section="overview_symptoms",
                    title=f"{r_name} | {d_en} — Symptoms & Regional Risk",
                    text=overview_text,
                    extra_search_terms=f"{d_hi} {kw_hint} cause symptoms identify diagnosis",
                )

                # Chunk B: Field Care & Soil/Water Management (with unambiguous Water Rule)
                field_items = "; ".join(card.get("field_care_en", []))
                field_text = (
                    f"{d_en} Field Care in {r_name}. "
                    f"{water_rule} "
                    f"Field Care Steps: {field_items}. "
                    f"Soil & Water Note: {card.get('soil_and_water_note_en', '')}"
                )
                self._add_chunk(
                    chunk_id=f"{region_id}::{disease_key}::field_care",
                    region_id=region_id,
                    disease_key=disease_key,
                    section="field_care",
                    title=f"{r_name} | {d_en} — Field Care, Water & Fertilizer",
                    text=field_text,
                    extra_search_terms=(
                        f"{d_hi} {kw_hint} field care water watering irrigation drainage flooding "
                        f"standing moist urea potash zinc lime fertilizer"
                    ),
                )

                # Chunk C: Organic & Biocontrol Options
                organic_items = "; ".join(card.get("organic_biocontrol_en", []))
                organic_text = (
                    f"{d_en} Organic & Biocontrol in {r_name}: {organic_items}"
                )
                self._add_chunk(
                    chunk_id=f"{region_id}::{disease_key}::organic_biocontrol",
                    region_id=region_id,
                    disease_key=disease_key,
                    section="organic_biocontrol",
                    title=f"{r_name} | {d_en} — Organic & Biocontrol",
                    text=organic_text,
                    extra_search_terms=f"{d_hi} {kw_hint} organic natural bio neem pseudomonas trichoderma",
                )

                # Chunk D: CIBRC Chemical Options & Dosages
                chem_items = "; ".join(card.get("chemical_cibrc_en", []))
                chem_text = (
                    f"{d_en} CIBRC Chemical Options in {r_name}: {chem_items} "
                    f"Regional Note: {card.get('regional_context_en', '')} "
                    f"{card.get('escalation_en', '')}"
                )
                self._add_chunk(
                    chunk_id=f"{region_id}::{disease_key}::chemical_cibrc",
                    region_id=region_id,
                    disease_key=disease_key,
                    section="chemical_cibrc",
                    title=f"{r_name} | {d_en} — CIBRC Chemicals & Dosages",
                    text=chem_text,
                    extra_search_terms=f"{d_hi} {kw_hint} chemical spray fungicide insecticide dosage medicine cibrc",
                )

                # Chunk E: Prevention & Recommended Regional Varieties
                prev_items = "; ".join(card.get("prevention_en", []))
                varieties = ", ".join(card.get("recommended_varieties", []))
                prev_text = (
                    f"{d_en} Prevention & Resistant Varieties for {r_name}: "
                    f"Recommended Varieties: {varieties}. "
                    f"Prevention Steps: {prev_items}. "
                    f"{card.get('escalation_en', '')}"
                )
                self._add_chunk(
                    chunk_id=f"{region_id}::{disease_key}::prevention_varieties",
                    region_id=region_id,
                    disease_key=disease_key,
                    section="prevention_varieties",
                    title=f"{r_name} | {d_en} — Prevention & Regional Varieties",
                    text=prev_text,
                    extra_search_terms=f"{d_hi} {kw_hint} prevention resistant varieties seed next crop kvk",
                )

        if self.chunks:
            self.avg_doc_len = sum(c.length for c in self.chunks) / len(self.chunks)

    def normalize_disease_key(self, raw_label: str) -> str:
        return raw_label.strip().lower().replace(" ", "_")

    def get_region_meta(self, region_id: str) -> tuple[str, dict]:
        if region_id in self.regions:
            return region_id, self.regions[region_id]
        return DEFAULT_REGION_ID, self.regions[DEFAULT_REGION_ID]

    def get_regional_card(self, region_id: str, raw_label: str) -> tuple[str, dict, dict]:
        resolved_region_id, region_meta = self.get_region_meta(region_id)
        disease_key = self.normalize_disease_key(raw_label)
        zone_map = self.advisories.get(resolved_region_id, self.advisories.get(DEFAULT_REGION_ID, {}))
        card = zone_map.get(disease_key) or self.advisories.get(DEFAULT_REGION_ID, {}).get(disease_key, {})
        return resolved_region_id, region_meta, card

    def get_water_rule(self, raw_label: str | None) -> str:
        if not raw_label:
            return ""
        return DISEASE_WATER_RULES.get(self.normalize_disease_key(raw_label), "")

    def infer_disease_from_text(self, text: str) -> str | None:
        lowered = text.lower()
        for disease_key, keywords in DISEASE_KEYWORDS.items():
            if any(kw in lowered for kw in keywords):
                return disease_key
        return None

    def _expand_query_tokens(self, query: str) -> list[str]:
        base_tokens = tokenize(query)
        expanded = list(base_tokens)
        for tok in base_tokens:
            if tok in BILINGUAL_QUERY_EXPANSIONS:
                expanded.extend(tokenize(BILINGUAL_QUERY_EXPANSIONS[tok]))
        return expanded

    def _bm25_score(self, query_tokens: list[str], chunk: KBChunk) -> float:
        n_docs = len(self.chunks)
        score = 0.0
        for term in set(query_tokens):
            tf = chunk.term_freqs.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + self.k1 * (1.0 - self.b + self.b * (chunk.length / self.avg_doc_len))
            score += idf * ((tf * (self.k1 + 1.0)) / denom)
        return score

    def build_initial_diagnosis_context(
        self, region_id: str, raw_label: str, confidence: float | None = None
    ) -> dict:
        """
        Build a complete, token-budgeted Knowledge Base bundle for the initial
        Llama 3.2 response immediately after EfficientNet-B4 detection.
        """
        resolved_region_id, region_meta, card = self.get_regional_card(region_id, raw_label)
        if not card:
            return {"context_text": "", "sources": []}

        disease_key = self.normalize_disease_key(raw_label)
        varieties = ", ".join(card.get("recommended_varieties", []))
        conf_note = f" ({confidence}% vision confidence)" if confidence is not None else ""
        water_rule = DISEASE_WATER_RULES.get(disease_key, "")

        if disease_key == "normal":
            lines = [
                "OUTPUT RULE: Start directly with '1. Routine Field Care:' with no introductory sentence.",
                f"REGION: {region_meta['name_en']} ({', '.join(region_meta.get('states_covered', [])[:4])})",
                f"STATUS: {card['name_en']}{conf_note} — {card['regional_risk_en']}",
                f"REGIONAL NOTE: {card['regional_context_en']}",
                f"ROUTINE FIELD CARE: {'; '.join(card.get('field_care_en', [])[:3])}",
                f"ORGANIC CARE: {card.get('organic_biocontrol_en', [''])[0]}",
                f"CHEMICAL GUIDANCE: {card.get('chemical_cibrc_en', [''])[0]}",
                f"RECOMMENDED REGIONAL VARIETIES: {varieties}",
            ]
        else:
            lines = [
                "OUTPUT RULE: Start directly with '1. Cause & Symptoms:' with no introductory sentence.",
                f"REGION: {region_meta['name_en']} ({', '.join(region_meta.get('states_covered', [])[:4])})",
                f"DIAGNOSIS: {card['name_en']}{conf_note} — {card['regional_risk_en']}",
                f"CAUSE: {card['pathogen']}",
                f"SYMPTOMS: {card['symptoms_en']}",
                f"REGIONAL ADAPTATION: {card['regional_context_en']}",
                water_rule,
                f"FIELD CARE: {'; '.join(card.get('field_care_en', [])[:3])}",
                f"ORGANIC / BIOCONTROL: {'; '.join(card.get('organic_biocontrol_en', [])[:2])}",
                f"CIBRC CHEMICAL OPTIONS: {'; '.join(card.get('chemical_cibrc_en', []))}",
                f"PREVENTION: {'; '.join(card.get('prevention_en', [])[:2])}",
                f"RECOMMENDED REGIONAL VARIETIES: {varieties}",
                f"ESCALATION: {card.get('escalation_en', '')}",
            ]

        context_text = "\n".join(line for line in lines if line)
        sources = [f"{region_meta['name_en']} | {card['name_en']} (Full Regional Advisory)"]
        return {"context_text": context_text, "sources": sources, "active_disease": disease_key}

    def retrieve(
        self,
        query: str,
        region_id: str = DEFAULT_REGION_ID,
        detected_disease: str | None = None,
        top_k: int = 2,
        max_chars: int = 1500,
    ) -> dict:
        """
        Retrieve the most relevant Knowledge Base chunks for a user chat query,
        strictly scoped to the user's selected agro-climatic zone and disease context.
        Caps effective context size and prunes weak chunks so Llama 3.2 1B gives
        one direct, consistent answer without contradicting itself.
        """
        resolved_region_id, region_meta = self.get_region_meta(region_id)
        queried_disease = self.infer_disease_from_text(query)
        active_disease = queried_disease or (
            self.normalize_disease_key(detected_disease) if detected_disease else None
        )

        # Keep follow-up context crisp even if caller passes larger defaults (e.g. top_k=3, max_chars=2400)
        effective_top_k = min(top_k, 2)
        effective_max_chars = min(max_chars, 2100 if any(t in query.lower() for t in SAFETY_TRIGGER_TERMS) else 1600)

        query_tokens = self._expand_query_tokens(query)
        query_lower = query.lower()

        candidates = [c for c in self.chunks if c.region_id == resolved_region_id]
        if not candidates:
            candidates = [c for c in self.chunks if c.region_id == DEFAULT_REGION_ID]

        if active_disease:
            disease_scoped = [
                c for c in candidates
                if c.disease_key == active_disease or c.section == "region_profile"
            ]
            if disease_scoped:
                candidates = disease_scoped

        scored: list[tuple[float, KBChunk]] = []
        for chunk in candidates:
            score = self._bm25_score(query_tokens, chunk)
            if active_disease and chunk.disease_key == active_disease and score > 0:
                score += 0.35
            scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        needs_safety = any(term in query_lower for term in SAFETY_TRIGGER_TERMS)
        disease_limit = max(1, effective_top_k - 1) if (needs_safety and effective_top_k > 1) else effective_top_k

        selected: list[KBChunk] = []
        top_score = scored[0][0] if scored else 0.0

        for score, chunk in scored:
            if len(selected) >= disease_limit:
                break
            if score > 0 and (not selected or score >= (0.60 * top_score)):
                selected.append(chunk)

        if not selected:
            if active_disease:
                fallback = self.build_initial_diagnosis_context(resolved_region_id, active_disease)
                fallback["active_disease"] = active_disease
                return fallback
            if scored:
                selected.append(scored[0][1])

        if needs_safety:
            safety_chunk = next(
                (c for c in self.chunks if c.chunk_id == "global::regulatory_safety"), None
            )
            if safety_chunk and safety_chunk not in selected:
                selected.append(safety_chunk)

        # Pin active region, disease, and single-stance answer guidance right inside retrieved context
        header_parts = [f"ACTIVE REGION: {region_meta['name_en']}"]
        if active_disease:
            _, _, active_card = self.get_regional_card(resolved_region_id, active_disease)
            disease_label = active_card.get("name_en", active_disease.replace("_", " ").title())
            header_parts.append(f"DETECTED DISEASE: {disease_label}")

        context_header = (
            " | ".join(header_parts)
            + "\nANSWER RULE: Give ONE clear, consistent recommendation based strictly on the facts below. "
            "Never write both 'Yes' and 'No', and start directly with the answer without filler."
        )

        packed_parts = [context_header]
        sources = []
        used_chars = len(context_header) + 2

        for idx, chunk in enumerate(selected, start=1):
            entry = f"[KB {idx}: {chunk.title}]\n{chunk.text}"
            if used_chars + len(entry) > effective_max_chars and len(packed_parts) > 1:
                remaining = effective_max_chars - used_chars
                if remaining > 220:
                    packed_parts.append(entry[: remaining - 3].rstrip() + "...")
                    sources.append(chunk.title)
                break
            packed_parts.append(entry)
            sources.append(chunk.title)
            used_chars += len(entry) + 2

        return {
            "context_text": "\n\n".join(packed_parts),
            "sources": sources,
            "active_disease": active_disease,
        }
