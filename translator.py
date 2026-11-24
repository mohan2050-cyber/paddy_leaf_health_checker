import re
import threading

SUPPORTED_LANG_CODES = ("en", "hi", "bn")
LIST_PREFIX_RE = re.compile(r"^(\s*(?:\d+[\.\)]\s+|[-*•]\s+))(.*)$")


class OfflineArgosTranslator:
    """
    100% offline neural translator backed by Argos Translate (OpenNMT + CTranslate2 int8 CPU)
    supporting English (en), Hindi (hi), and Bengali (bn). Zero internet calls at runtime.
    Expects input in the explicitly selected language (no auto language detection).
    """

    def __init__(self) -> None:
        try:
            import argostranslate.translate as argos_translate
        except ImportError as exc:
            raise RuntimeError(
                "argostranslate is not installed. Run once while online:\n"
                "  pip install argostranslate"
            ) from exc

        installed_languages = argos_translate.get_installed_languages()
        lang_by_code = {lang.code: lang for lang in installed_languages}

        en_lang = lang_by_code.get("en")
        self._from_en = {}
        self._to_en = {}

        for code in ("hi", "bn"):
            other = lang_by_code.get(code)
            if en_lang and other:
                en_to_other = en_lang.get_translation(other)
                other_to_en = other.get_translation(en_lang)
                if en_to_other is not None:
                    self._from_en[code] = en_to_other
                if other_to_en is not None:
                    self._to_en[code] = other_to_en

        missing = []
        for code in ("hi", "bn"):
            if code not in self._from_en:
                missing.append(f"translate-en_{code}")
            if code not in self._to_en:
                missing.append(f"translate-{code}_en")

        if missing:
            print(
                "[WARNING] Missing Argos Translate language packs: "
                + ", ".join(missing)
                + "\n  Install once while online:\n"
                + "  argospm update && argospm install "
                + " ".join(missing)
            )

        self._lock = threading.Lock()

    def _translate_structured(self, text: str, translator_obj) -> str:
        """
        Translate line-by-line while preserving blank lines, numbering (1., 2.),
        and bullet prefixes (-, *) so multi-section replies never lose headings or structure.
        """
        if not text or not text.strip():
            return ""

        out_lines: list[str] = []
        with self._lock:
            for raw_line in text.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    out_lines.append("")
                    continue

                # Clean markdown bold/italic markers so CTranslate2 doesn't mangle asterisks
                clean_line = stripped.replace("**", "").replace("__", "")
                match = LIST_PREFIX_RE.match(clean_line)
                if match:
                    prefix, body = match.group(1), match.group(2).strip()
                else:
                    prefix, body = "", clean_line

                if not body:
                    out_lines.append(prefix.rstrip())
                    continue

                translated = translator_obj.translate(body).strip()
                out_lines.append(f"{prefix}{translated}")

        return "\n".join(out_lines).strip()

    def to_english(self, text: str, src_lang: str) -> str:
        """
        Unconditionally translate user input from the selected language (hi/bn) into English.
        Returns text unchanged when English (en) is selected.
        """
        if src_lang == "en" or not text or not text.strip():
            return text
        translator_obj = self._to_en.get(src_lang)
        if translator_obj is None:
            raise RuntimeError(
                f"Offline Argos Translate pack '{src_lang}->en' is not installed.\n"
                f"Run once while online: argospm update && argospm install translate-{src_lang}_en"
            )
        return self._translate_structured(text, translator_obj)

    def from_english(self, text: str, tgt_lang: str) -> str:
        """
        Translate Llama's English output into the selected language (hi/bn).
        Returns text unchanged when English (en) is selected.
        """
        if tgt_lang == "en" or not text or not text.strip():
            return text
        translator_obj = self._from_en.get(tgt_lang)
        if translator_obj is None:
            raise RuntimeError(
                f"Offline Argos Translate pack 'en->{tgt_lang}' is not installed.\n"
                f"Run once while online: argospm update && argospm install translate-en_{tgt_lang}"
            )
        return self._translate_structured(text, translator_obj)
