import base64
import io
import re
from typing import List, Optional, Sequence

DATA_URI_PATTERN = re.compile(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)", re.IGNORECASE)


class ImageOCR:
    def __init__(self, languages: str = "rus+eng"):
        self.languages = languages
        self._available: Optional[bool] = None

    def _load(self):
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            self._available = False
            return None, None
        self._available = True
        return pytesseract, Image

    @property
    def available(self) -> bool:
        if self._available is None:
            self._load()
        return bool(self._available)

    def image_to_text(self, image_bytes: bytes) -> str:
        pytesseract, Image = self._load()
        if not pytesseract:
            return ""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            return pytesseract.image_to_string(image, lang=self.languages).strip()
        except Exception:
            return ""

    def extract_data_uri_images(self, html: str) -> List[bytes]:
        images = []
        for match in DATA_URI_PATTERN.finditer(html or ""):
            try:
                images.append(base64.b64decode(re.sub(r"\s+", "", match.group(1))))
            except Exception:
                continue
        return images

    def analyze_images(self, images: Sequence[dict]) -> List[dict]:
        results = []
        for image in images:
            text = self.image_to_text(image.get("bytes", b""))
            results.append(
                {"source": image.get("source", "unknown"), "text": text, "has_text": bool(text)}
            )
        return results

    def combined_text(self, images: Sequence[dict]) -> str:
        return "\n\n".join(result["text"] for result in self.analyze_images(images) if result["has_text"])
