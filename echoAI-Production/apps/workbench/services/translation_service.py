import os
import requests
from pydantic import BaseModel
from typing import Optional
from functools import lru_cache
import json

# ── Models ──────────────────────────────────────────────────────────────────

class TextInput(BaseModel):
    Text: str


class TranslationResponse(BaseModel):
    original_text: str
    detected_language: str
    detected_language_name: str
    confidence: float
    translated_text: str
    status: str


# ── Service ─────────────────────────────────────────────────────────────────

class TranslationService:
    """Azure Translator API wrapper for language detection and translation."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        charge_code: str,
        api_version: str = "3.0",
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.charge_code = charge_code
        self.api_version = api_version
        self._languages_cache: Optional[dict] = None

    # ── helpers ──────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Ocp-Apim-Subscription-Key": "7bd51d543bbb464da97d724e4cec241e",
            "Content-Type": "application/json",
            "x-kpmg-charge-code": "0000000",
        }

    def get_all_languages(self) -> dict:
        if self._languages_cache is not None:
            return self._languages_cache
        try:
            url = f"https://api.workbench.kpmg/translator/azure/text/languages?api-version=3.0"
            # resp = requests.get(url, headers=self._headers(), timeout=30)
            headers = {
            'api-version': '3.0',
            'x-kpmg-charge-code': '00000000',
            'Ocp-Apim-Subscription-Key': '7bd51d543bbb464da97d724e4cec241e'
            }

            resp = requests.request("GET", url, headers=headers)
            if resp.status_code == 200:
                self._languages_cache = resp.json()
                return self._languages_cache
        except Exception:
            pass
        return {}

    def get_language_name(self, code: str) -> str:
        data = self.get_all_languages()
        if "translation" in data:
            info = data["translation"].get(code, {})
            if "name" in info:
                return info["name"]
        return code.upper()

    # ── public API ───────────────────────────────────────────────────────

    def detect(self, text: str) -> dict:
        """Detect the language of *text*."""
        url = f"{self.base_url}/detect?api-version={self.api_version}"
        resp = requests.post(
            url, headers=self._headers(), data=json.dumps([{"Text": text}]), timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Detection failed: {resp.text}")

        detection = resp.json()[0]
        lang = detection.get("language", "unknown")
        return {
            "text": text,
            "detected_language": lang,
            "detected_language_name": self.get_language_name(lang),
            "confidence": detection.get("score", 0.0),
            "is_translation_supported": detection.get("isTranslationSupported", True),
            "status": "success",
        }

    def translate(self, text: str) -> TranslationResponse:
        """Detect language and translate *text* to English."""
        url = f"{self.base_url}/detect?api-version={self.api_version}"
        resp = requests.post(
            url, headers=self._headers(), data=json.dumps([{"Text": text}]), timeout=30
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Detection failed: {resp.text}")

        detection = resp.json()[0]
        print("detection done:",detection)
        lang = detection.get("language", "unknown")
        confidence = detection.get("score", 0.0)
        lang_name = self.get_language_name(lang)

        # Already English – skip translation
        if lang == "en":
            return TranslationResponse(
                original_text=text,
                detected_language=lang,
                detected_language_name=lang_name,
                confidence=confidence,
                translated_text=text,
                status="success",
            )

        # Translate to English
        t_url = (
            f"{self.base_url}/translate"
            f"?api-version={self.api_version}&from={lang}&to=en"
        )
        t_resp = requests.post(
            t_url, headers=self._headers(), data=json.dumps([{"Text": text}]), timeout=30
        )
        print("in to*****")
        if t_resp.status_code != 200:
            raise RuntimeError(f"Translation failed: {t_resp.text}")

        translated_text = t_resp.json()[0]["translations"][0]["text"]
        return TranslationResponse(
            original_text=text,
            detected_language=lang,
            detected_language_name=lang_name,
            confidence=confidence,
            translated_text=translated_text,
            status="success",
        )

# obj=TranslationService(base_url="https://api.workbench.kpmg/translator/azure/text", api_key="7bd51d543bbb464da97d724e4cec241e", charge_code="0000000")
# result=obj.translate("Bonjour tout le monde")
# print(result)