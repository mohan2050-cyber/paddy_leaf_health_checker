const headerTitleEl = document.getElementById("header-title");
const headerSubtitleEl = document.getElementById("header-subtitle");
const panel1HeadingEl = document.getElementById("panel1-heading");
const panel2HeadingEl = document.getElementById("panel2-heading");
const languageSelect = document.getElementById("language-select");
const languageLabelEl = document.getElementById("language-label");
const regionLabelEl = document.getElementById("region-label");
const fileLabelEl = document.getElementById("file-label");

const uploadForm = document.getElementById("upload-form");
const imageInput = document.getElementById("image-input");
const predictBtn = document.getElementById("predict-btn");
const previewBox = document.getElementById("preview-box");
const previewImg = document.getElementById("preview-img");
const resultBox = document.getElementById("result-box");
const resultStatus = document.getElementById("result-status");
const resultDetail = document.getElementById("result-detail");

const regionSelect = document.getElementById("region-select");
const regionMetaEl = document.getElementById("region-meta");

const chatMessages = document.getElementById("chat-messages");
const welcomeMessageEl = document.getElementById("welcome-message");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

const REGION_STORAGE_KEY = "paddy_selected_region";
const LANGUAGE_STORAGE_KEY = "paddy_selected_language";
const DEFAULT_LANGUAGE = window.DEFAULT_LANGUAGE || "en";
const UI_TEXT = window.APP_UI_TEXT || {};

const FILLER_PREAMBLE_EN_RE = /^(?:i (?:can|will|would be happy to) (?:provide|give|share|help|answer|explain)|here(?:'s| is| are) (?:the|your|a|some)|sure[,!.]?(?:\s+here|\s+i\s+can|$)|certainly[,!.]?(?:\s+here|\s+i\s+can|$)|of course[,!.]?|below (?:is|are)|as requested|following (?:the|your)|based on the (?:retrieved )?knowledge base[,:\s]*$)/i;
const FILLER_PREAMBLE_HI_RE = /^(?:मैं (?:सरल अंग्रेजी|अंग्रेजी|अनुरोध|उत्तर|जवाब|सलाह)|यहाँ (?:आपका|उत्तर|जवाब|सलाह)|ज़रूर|निश्चित रूप से|अनुरोध के अनुसार)/;
const FILLER_PREAMBLE_BN_RE = /^(?:আমি (?:সহজ ইংরেজি|ইংরেজি|অনুরোধ|উত্তর|পরামর্শ)|এখানে (?:আপনার|উত্তর|পরামর্শ)|অবশ্যই|নিশ্চয়ই|অনুরোধ অনুযায়ী)/;
const SECTION_ONE_RE = /(?:^|\n)\s*(?:\*\*|__)?\s*[1१১][\.\)]\s+/;

let chatHistory = [];
let currentDetectedDisease = null;
let currentDetectedLabel = null;
let currentRegionNameEn = null;
let currentConfidence = null;
let lastPredictionData = null;

function getSelectedRegion() {
    return regionSelect ? regionSelect.value : "national_default";
}

function getSelectedLanguage() {
    const code = languageSelect ? languageSelect.value : DEFAULT_LANGUAGE;
    return code === "hi" || code === "bn" ? code : "en";
}

function getUi(langCode = getSelectedLanguage()) {
    return UI_TEXT[langCode] || UI_TEXT.en || {};
}

function getRegionOptionName(opt, langCode) {
    if (!opt) return "";
    if (langCode === "hi") return opt.dataset.nameHi || opt.dataset.nameEn || opt.value;
    if (langCode === "bn") return opt.dataset.nameBn || opt.dataset.nameEn || opt.value;
    return opt.dataset.nameEn || opt.value;
}

function updateRegionMeta() {
    if (!regionSelect || !regionMetaEl) return;
    const opt = regionSelect.options[regionSelect.selectedIndex];
    if (!opt) return;

    const ui = getUi();
    const states = opt.dataset.states || "";
    const seasons = opt.dataset.seasons || "";
    const parts = [];
    if (states) parts.push(`${ui.states_label || "States"}: ${states}`);
    if (seasons) parts.push(`${ui.seasons_label || "Seasons"}: ${seasons}`);
    regionMetaEl.textContent = parts.join(" | ");
}

function getSelectedLanguageTag() {
    if (!languageSelect) return "";
    const opt = languageSelect.options[languageSelect.selectedIndex];
    const nativeName = opt?.dataset.nativeName || "";
    const englishName = opt?.dataset.englishName || "";
    if (nativeName && englishName && nativeName !== englishName) {
        return `${nativeName} (${englishName})`;
    }
    return nativeName || englishName;
}

function renderPredictionResult(data) {
    const lang = getSelectedLanguage();
    const uiEn = getUi("en");
    const ui = getUi(lang);

    resultBox.classList.remove("hidden", "healthy", "diseased");
    resultStatus.style.whiteSpace = "pre-line";
    resultDetail.style.whiteSpace = "pre-line";

    const labelLocal =
        lang === "hi"
            ? data.label_hi || data.label
            : lang === "bn"
                ? data.label_bn || data.label
                : data.label;

    const regionLocal =
        lang === "hi"
            ? data.region_name_hi || data.region_name_en
            : lang === "bn"
                ? data.region_name_bn || data.region_name_en
                : data.region_name_en;

    const riskLocal =
        lang === "hi"
            ? data.regional_risk_hi || data.regional_risk_en
            : lang === "bn"
                ? data.regional_risk_bn || data.regional_risk_en
                : data.regional_risk_en;

    const langTag = getSelectedLanguageTag();

    if (data.is_diseased) {
        resultBox.classList.add("diseased");
        if (lang === "en") {
            resultStatus.textContent = `${uiEn.disease_detected || "Disease Detected"} - ${data.label}`;
            resultDetail.textContent =
                `${uiEn.confidence || "Confidence"}: ${data.confidence}%\n` +
                `${uiEn.region || "Region"}: ${data.region_name_en} (${data.regional_risk_en})`;
        } else {
            resultStatus.textContent =
                `English: ${uiEn.disease_detected || "Disease Detected"} - ${data.label}\n` +
                `${langTag}: ${ui.disease_detected || "Disease Detected"} - ${labelLocal}`;
            resultDetail.textContent =
                `${uiEn.confidence || "Confidence"}: ${data.confidence}% | ${ui.confidence || "Confidence"}: ${data.confidence}%\n` +
                `${uiEn.region || "Region"}: ${data.region_name_en} (${data.regional_risk_en})\n` +
                `${ui.region || "Region"}: ${regionLocal} (${riskLocal})`;
        }
    } else {
        resultBox.classList.add("healthy");
        if (lang === "en") {
            resultStatus.textContent = data.label;
            resultDetail.textContent =
                `${uiEn.confidence || "Confidence"}: ${data.confidence}%\n` +
                `${uiEn.region || "Region"}: ${data.region_name_en}`;
        } else {
            resultStatus.textContent =
                `English: ${data.label}\n` +
                `${langTag}: ${labelLocal}`;
            resultDetail.textContent =
                `${uiEn.confidence || "Confidence"}: ${data.confidence}% | ${ui.confidence || "Confidence"}: ${data.confidence}%\n` +
                `${uiEn.region || "Region"}: ${data.region_name_en}\n` +
                `${ui.region || "Region"}: ${regionLocal}`;
        }
    }
}

function applyLanguageToPage() {
    const lang = getSelectedLanguage();
    const ui = getUi(lang);

    document.documentElement.lang = lang;
    if (ui.page_title) document.title = ui.page_title;
    if (headerTitleEl && ui.header_title) headerTitleEl.textContent = ui.header_title;
    if (headerSubtitleEl && ui.header_subtitle) headerSubtitleEl.textContent = ui.header_subtitle;
    if (panel1HeadingEl && ui.panel1_heading) panel1HeadingEl.textContent = ui.panel1_heading;
    if (panel2HeadingEl && ui.panel2_heading) panel2HeadingEl.textContent = ui.panel2_heading;
    if (languageLabelEl && ui.language_label) languageLabelEl.textContent = ui.language_label;
    if (regionLabelEl && ui.region_label) regionLabelEl.textContent = ui.region_label;
    if (fileLabelEl && ui.file_label) fileLabelEl.textContent = ui.file_label;
    if (predictBtn && !predictBtn.disabled && ui.predict_btn) predictBtn.textContent = ui.predict_btn;
    if (sendBtn && ui.send_btn) sendBtn.textContent = ui.send_btn;
    if (chatInput && ui.chat_placeholder) chatInput.placeholder = ui.chat_placeholder;
    if (welcomeMessageEl && ui.welcome_message) welcomeMessageEl.textContent = ui.welcome_message;

    if (regionSelect) {
        for (const opt of regionSelect.options) {
            const regionName = getRegionOptionName(opt, lang);
            const statesShort = opt.dataset.statesShort || "";
            opt.textContent = statesShort ? `${regionName} — ${statesShort}` : regionName;
        }
    }

    updateRegionMeta();

    if (lastPredictionData) {
        renderPredictionResult(lastPredictionData);
    }
}

function initLanguageSelector() {
    if (!languageSelect) return;

    const saved = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (saved && Array.from(languageSelect.options).some((o) => o.value === saved)) {
        languageSelect.value = saved;
    } else {
        languageSelect.value = DEFAULT_LANGUAGE;
    }
    applyLanguageToPage();

    languageSelect.addEventListener("change", () => {
        localStorage.setItem(LANGUAGE_STORAGE_KEY, getSelectedLanguage());
        applyLanguageToPage();
    });
}

function initRegionSelector() {
    if (!regionSelect) return;

    const saved = localStorage.getItem(REGION_STORAGE_KEY);
    if (saved && Array.from(regionSelect.options).some((o) => o.value === saved)) {
        regionSelect.value = saved;
    }
    updateRegionMeta();

    regionSelect.addEventListener("change", () => {
        const chosen = getSelectedRegion();
        localStorage.setItem(REGION_STORAGE_KEY, chosen);
        updateRegionMeta();

        const opt = regionSelect.options[regionSelect.selectedIndex];
        const lang = getSelectedLanguage();
        const ui = getUi(lang);
        const regionName = getRegionOptionName(opt, lang);

        appendMessage(
            `${ui.region_updated || "Region updated to"}: ${regionName}\n${ui.region_updated_hint || ""}`,
            "bot"
        );
    });
}

function appendMessage(text, type) {
    const div = document.createElement("div");
    div.className = `message ${type}`;
    div.style.whiteSpace = "pre-wrap";
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}

function setChatDisabled(disabled) {
    chatInput.disabled = disabled;
    sendBtn.disabled = disabled;
    if (predictBtn) {
        predictBtn.disabled = disabled;
    }
}

function stripBlockPreamble(blockText, isInitialDiagnosis, langCode) {
    if (!blockText) return "";
    let text = blockText.trim();

    if (isInitialDiagnosis) {
        const match = SECTION_ONE_RE.exec(text);
        if (match) {
            const startIdx = match.index + (match[0].startsWith("\n") ? 1 : 0);
            if (startIdx > 0) {
                const prefix = text.slice(0, startIdx).trim();
                if (prefix.length < 260 && prefix.split("\n").length <= 3) {
                    return text.slice(startIdx).trim();
                }
            }
        }
    }

    const fillerRe =
        langCode === "hi"
            ? FILLER_PREAMBLE_HI_RE
            : langCode === "bn"
                ? FILLER_PREAMBLE_BN_RE
                : FILLER_PREAMBLE_EN_RE;

    const lines = text.split("\n");
    for (let step = 0; step < 2; step++) {
        while (lines.length > 0 && !lines[0].trim()) {
            lines.shift();
        }
        if (lines.length >= 2 && fillerRe.test(lines[0].trim())) {
            const remainder = lines.slice(1).join("\n").trim();
            if (remainder) {
                lines.shift();
                continue;
            }
        }
        break;
    }

    return lines.join("\n").trim();
}

function cleanReply(rawReply, isInitialDiagnosis, langCode) {
    if (!rawReply) return "";
    const marker = (getUi(langCode).section_marker) || "";

    if (!marker || !rawReply.includes(marker)) {
        const enBody = rawReply.replace(/^\s*English\s*:\s*/i, "").trim();
        return stripBlockPreamble(enBody, isInitialDiagnosis, "en");
    }

    const [enPartRaw, localPartRaw] = rawReply.split(marker);
    const enBody = enPartRaw.replace(/^\s*English\s*:\s*/i, "").trim();
    const cleanedEn = stripBlockPreamble(enBody, isInitialDiagnosis, "en");
    const cleanedLocal = stripBlockPreamble(localPartRaw.trim(), isInitialDiagnosis, langCode);

    return `English:\n${cleanedEn}${marker}${cleanedLocal}`;
}

async function sendToChatbot(
    userPrompt,
    displayUserMessage = true,
    isInitialDiagnosis = false
) {
    if (displayUserMessage) {
        appendMessage(userPrompt, "user");
    }

    const lang = getSelectedLanguage();
    const ui = getUi(lang);

    const payloadMessages = isInitialDiagnosis
        ? [{ role: "user", content: userPrompt, already_english: true }]
        : [...chatHistory.slice(-4), { role: "user", content: userPrompt, lang: lang }];

    const thinkingText = isInitialDiagnosis
        ? (ui.thinking_initial || "Llama 3.2 is generating regional remedies from the Knowledge Base...")
        : (ui.thinking_chat || "Generating regional reply...");

    const thinkingBubble = appendMessage(thinkingText, "bot");
    setChatDisabled(true);

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                messages: payloadMessages,
                region_id: getSelectedRegion(),
                language: lang,
                detected_disease: currentDetectedDisease,
                confidence: currentConfidence,
                is_initial_diagnosis: isInitialDiagnosis,
            }),
        });
        const data = await response.json();

        if (!response.ok) {
            thinkingBubble.className = "message error";
            thinkingBubble.textContent = data.error || "Chatbot request failed.";
            return;
        }

        const cleanedReply = cleanReply(data.reply, isInitialDiagnosis, data.language || lang);
        let displayReply = cleanedReply;
        if (Array.isArray(data.kb_sources) && data.kb_sources.length > 0) {
            displayReply += `\n\n[Knowledge Base: ${data.kb_sources.join(" ; ")}]`;
        }

        thinkingBubble.textContent = displayReply;

        if (isInitialDiagnosis) {
            // Keep follow-up chatHistory compact so the 4-section wall of text does not
            // distract Llama 3.2 1B on subsequent follow-up questions (e.g., watering/dosage).
            const diseaseText = currentDetectedLabel || currentDetectedDisease || "paddy condition";
            const regionText = currentRegionNameEn || getSelectedRegion();
            chatHistory = [
                {
                    role: "user",
                    content: `Detected condition: ${diseaseText} in ${regionText}.`,
                    already_english: true,
                },
                {
                    role: "assistant",
                    content: `Noted: ${diseaseText} in ${regionText}. Ask any follow-up question.`,
                    already_english: true,
                },
            ];
        } else {
            // Store English versions of completed turns so prior turns are not re-translated
            chatHistory.push({
                role: "user",
                content: data.english_user_text || userPrompt,
                already_english: true,
            });
            chatHistory.push({
                role: "assistant",
                content: data.english_reply || cleanedReply,
                already_english: true,
            });
            if (chatHistory.length > 6) {
                chatHistory = chatHistory.slice(-6);
            }
        }
    } catch (err) {
        thinkingBubble.className = "message error";
        thinkingBubble.textContent = ui.network_error || "Network error while contacting the assistant.";
    } finally {
        setChatDisabled(false);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (!file) return;

    previewImg.src = URL.createObjectURL(file);
    previewBox.classList.remove("hidden");
    resultBox.classList.add("hidden");
});

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = imageInput.files[0];
    if (!file) return;

    const lang = getSelectedLanguage();
    const ui = getUi(lang);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("region_id", getSelectedRegion());

    predictBtn.disabled = true;
    predictBtn.textContent = ui.predict_btn_loading || "Checking...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            alert(data.error || "Prediction failed.");
            return;
        }

        lastPredictionData = data;
        currentDetectedDisease = data.disease_key || null;
        currentDetectedLabel = data.label || null;
        currentRegionNameEn = data.region_name_en || null;
        currentConfidence = data.confidence ?? null;
        chatHistory = [];

        renderPredictionResult(data);

        const displayLabel =
            lang === "hi"
                ? data.label_hi || data.label
                : lang === "bn"
                    ? data.label_bn || data.label
                    : data.label;
        const displayRegion =
            lang === "hi"
                ? data.region_name_hi || data.region_name_en
                : lang === "bn"
                    ? data.region_name_bn || data.region_name_en
                    : data.region_name_en;

        if (data.is_diseased) {
            const detectMsg =
                lang === "en"
                    ? `Detected "${data.label}" — ${data.region_name_en}`
                    : `Detected "${data.label}" — ${data.region_name_en}\n` +
                      `${ui.detected_prefix || "Detected"} "${displayLabel}" — ${displayRegion}`;
            appendMessage(detectMsg, "user");

            const autoPrompt =
                `Detected disease: ${data.label} (${data.confidence}% confidence). Region: ${data.region_name_en}. ` +
                `Output ONLY these 4 numbered items with no introduction:\n` +
                `1. Cause & Symptoms:\n` +
                `2. Field Care & Organic Options:\n` +
                `3. CIBRC Chemical Remedies:\n` +
                `4. Prevention & Regional Varieties:`;

            await sendToChatbot(autoPrompt, false, true);
        } else {
            const checkMsg =
                lang === "en"
                    ? `Checked leaf: "${data.label}" — ${data.region_name_en}`
                    : `Checked leaf: "${data.label}" — ${data.region_name_en}\n` +
                      `${ui.checked_prefix || "Checked leaf"}: "${displayLabel}" — ${displayRegion}`;
            appendMessage(checkMsg, "user");

            const healthyPrompt =
                `Leaf status: Healthy (${data.confidence}% confidence). Region: ${data.region_name_en}. ` +
                `Output ONLY these 2 numbered items with no introduction:\n` +
                `1. Routine Field Care:\n` +
                `2. Recommended Regional Varieties:`;

            await sendToChatbot(healthyPrompt, false, true);
        }
    } catch (err) {
        alert(ui.server_error || "Could not reach the local server.");
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = getUi().predict_btn || "Check Leaf";
    }
});

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = "";
    await sendToChatbot(text, true, false);
});

initLanguageSelector();
initRegionSelector();
