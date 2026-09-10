import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..features import extract_features, normalize_category
from ..features.html_signals import split_html_into_blocks, strip_tags
from .base import DEFAULT_THRESHOLDS, Scorer, ScoreResult, VerdictThresholds
from .engine import analyze_chunk_vector, analyze_email_vector
from .prompts import (
    HTML_SYSTEM_PROMPT,
    HTML_USER_PROMPT,
    OCR_USER_PROMPT,
    TEXT_SYSTEM_PROMPT,
    TEXT_USER_PROMPT,
)

PathLike = Union[str, Path]

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

FLAG_TO_CATEGORY = {
    "TEMPLATE_VARIABLE": "html_template",
    "SUSPICIOUS_COMMENT": "html_template",
    "DUPLICATE_STYLES": "html_template",
    "TOO_DEEP_NESTING": "html_template",
    "EMPTY_CELL": "html_template",
    "STRUCTURAL_REDUNDANCY": "html_template",
    "INLINE_STYLE_OVERUSE": "html_template",
    "REPETITIVE_PATTERN": "body",
    "MISSING_ALT": "html_template",
}

BLOCK_FLAG_FIELDS = {
    "template_variable": "TEMPLATE_VARIABLE",
    "repeating_pattern": "REPETITIVE_PATTERN",
    "suspicious_comments": "SUSPICIOUS_COMMENT",
    "deep_nesting": "TOO_DEEP_NESTING",
    "duplicated_styles": "DUPLICATE_STYLES",
    "empty_cell": "EMPTY_CELL",
}


class ResponseCache:
    def __init__(self, directory: PathLike):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / ("%s.json" % key)

    @staticmethod
    def key(model: str, prompt: str) -> str:
        digest = hashlib.sha256(("%s::%s" % (model, prompt)).encode("utf-8")).hexdigest()
        return digest[:32]

    def get(self, key: str) -> Optional[dict]:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, payload: dict) -> None:
        with open(self._path(key), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)


def parse_json_response(content: str) -> Optional[dict]:
    match = JSON_PATTERN.search(content or "")
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


class LLMScorer(Scorer):
    name = "llm"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        top_p: float = 0.8,
        timeout: int = 180,
        delay: float = 2.0,
        max_block_length: int = 10000,
        max_html_length: int = 25000,
        cache_dir: Optional[PathLike] = None,
        max_blocks: int = 6,
        thresholds: Optional[VerdictThresholds] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.delay = delay
        self.max_block_length = max_block_length
        self.max_html_length = max_html_length
        self.max_blocks = max_blocks
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.cache = ResponseCache(cache_dir) if cache_dir else None

    @classmethod
    def from_settings(cls, settings, model: Optional[str] = None) -> "LLMScorer":
        if not settings.api_key:
            raise ValueError("DEEPCODE_API_KEY is not configured")
        return cls(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model=model or settings.default_model,
            temperature=settings.temperature,
            top_p=settings.top_p,
            timeout=settings.request_timeout,
            delay=settings.request_delay,
            max_block_length=settings.max_block_length,
            max_html_length=settings.max_html_length,
            cache_dir=settings.cache_dir,
            thresholds=VerdictThresholds(
                mixed=settings.verdict_mixed_threshold, ai=settings.verdict_ai_threshold
            ),
        )

    def _request(self, system_prompt: str, user_prompt: str) -> dict:
        import requests

        cache_key = ResponseCache.key(self.model, system_prompt + user_prompt) if self.cache else None
        if cache_key:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        headers = {"Authorization": "Bearer %s" % self.api_key, "Content-Type": "application/json"}

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        except Exception as error:
            return {"error": str(error)[:200]}

        if response.status_code != 200:
            return {"error": "HTTP %s" % response.status_code}

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError):
            return {"error": "malformed response"}

        parsed = parse_json_response(content) or {"error": "unparsable json"}
        if cache_key:
            self.cache.put(cache_key, parsed)
        if self.delay:
            time.sleep(self.delay)
        return parsed

    def analyze_block(self, block_html: str) -> dict:
        clean = (block_html or "").encode("utf-8", errors="ignore").decode("utf-8")
        if len(clean) > self.max_block_length:
            clean = clean[: self.max_block_length]
        parsed = self._request(HTML_SYSTEM_PROMPT, HTML_USER_PROMPT.format(html=clean))
        return {
            "ai_probability": parsed.get("ai_probability", 0),
            "flags": parsed.get("flags", []) or [],
            "summary": parsed.get("summary", "") or parsed.get("error", ""),
        }

    def analyze_text(self, text: str) -> dict:
        clean = (text or "").strip()
        if not clean:
            return {"ai_probability": 0, "flags": [], "parts": [], "summary": ""}
        if len(clean) > self.max_block_length:
            clean = clean[: self.max_block_length]
        parsed = self._request(TEXT_SYSTEM_PROMPT, TEXT_USER_PROMPT.format(text=clean))
        return {
            "ai_probability": parsed.get("ai_probability", 0),
            "flags": parsed.get("flags", []) or [],
            "parts": parsed.get("parts", []) or [],
            "summary": parsed.get("summary", "") or parsed.get("error", ""),
        }

    def analyze_ocr(self, ocr_text: str) -> dict:
        clean = (ocr_text or "").strip()
        if not clean:
            return {"ai_probability": 0, "flags": [], "summary": ""}
        if len(clean) > self.max_block_length:
            clean = clean[: self.max_block_length]
        parsed = self._request(TEXT_SYSTEM_PROMPT, OCR_USER_PROMPT.format(text=clean))
        return {
            "ai_probability": parsed.get("ai_probability", 0),
            "flags": parsed.get("flags", []) or [],
            "summary": parsed.get("summary", "") or parsed.get("error", ""),
        }

    def _block_vector(self, block_result: dict) -> dict:
        flags = {str(flag).upper().strip() for flag in block_result.get("flags", [])}
        vector = {"llm_probability": block_result.get("ai_probability", 0)}
        for field, flag in BLOCK_FLAG_FIELDS.items():
            vector[field] = flag in flags
        return vector

    def score_email(
        self,
        text: str = "",
        subject: str = "",
        html: str = "",
        ocr_text: str = "",
    ) -> ScoreResult:
        blocks = split_html_into_blocks(html, self.max_html_length)[: self.max_blocks]
        block_results = []
        block_scores = []
        flags: List[str] = []

        for block in blocks:
            block_result = self.analyze_block(block["content"])
            vector_result = analyze_chunk_vector(self._block_vector(block_result))
            block_scores.append(vector_result["AI_Score"])
            for flag in block_result.get("flags", []):
                if flag not in flags:
                    flags.append(flag)
            block_results.append(
                {
                    "position": block["position"],
                    "block_type": block["type"],
                    "ai_score": vector_result["AI_Score"],
                    "flags": block_result.get("flags", []),
                    "summary": block_result.get("summary", ""),
                }
            )

        plain_text = text or strip_tags(html)
        text_result = self.analyze_text("%s\n\n%s" % (subject, plain_text) if subject else plain_text)
        ocr_result = self.analyze_ocr(ocr_text)

        html_component = sum(block_scores) / len(block_scores) if block_scores else 0.0
        text_component = max(0.0, min(1.0, float(text_result.get("ai_probability", 0) or 0) / 100.0))
        html_score = max(html_component, text_component) if text_component else html_component

        combined = analyze_email_vector(
            html_score=html_score,
            ocr_score=float(ocr_result.get("ai_probability", 0) or 0) / 100.0,
            has_ocr=bool((ocr_text or "").strip()),
        )

        categories: Dict[str, float] = {name: 0.0 for name in ("subject", "opener", "body", "cta", "closer", "html_template", "image")}
        for part in text_result.get("parts", []):
            key = normalize_category(str(part).strip().lower())
            if key in categories:
                categories[key] = max(categories[key], text_component or 1.0)
        for flag in flags:
            key = FLAG_TO_CATEGORY.get(str(flag).upper().strip())
            if key:
                categories[key] = max(categories[key], html_component or 1.0)
        if (ocr_text or "").strip():
            categories["image"] = max(
                categories["image"], float(ocr_result.get("ai_probability", 0) or 0) / 100.0
            )

        fallback_categories = extract_features(text=plain_text, subject=subject, html=html, ocr_text=ocr_text).categories
        for key, value in fallback_categories.items():
            if categories.get(key, 0.0) == 0.0 and combined["AI_Score"] >= 0.5:
                categories[key] = value

        explanation_parts = [combined["Explanation"]]
        if text_result.get("summary"):
            explanation_parts.append(str(text_result["summary"]))
        if flags:
            explanation_parts.append("Структурные флаги разметки: %s." % ", ".join(flags[:8]))

        return ScoreResult(
            score=combined["AI_Score"],
            confidence=0.5 + abs(combined["AI_Score"] - 0.5),
            categories=categories,
            signals={
                "html_score": html_component,
                "text_score": text_component,
                "ocr_score": float(ocr_result.get("ai_probability", 0) or 0) / 100.0,
            },
            explanation=" ".join(part for part in explanation_parts if part),
            scorer=self.name,
            thresholds=self.thresholds,
            meta={"model": self.model, "blocks": block_results, "flags": flags},
        )
