import base64
import io
import re

import pytesseract
from PIL import Image


class ImageOCR:
    """Распознавание текста на изображениях внутри email."""

    def __init__(self, languages="rus+eng"):
        self.languages = languages

    def image_to_text(self, image_bytes):
        """
        Распознать текст из байтов изображения.

        Возвращает:
            str: распознанный текст.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))

            # Переводим изображение в RGB Tesseract.
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            text = pytesseract.image_to_string(
                image,
                lang=self.languages
            )

            return text.strip()

        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    def extract_data_uri_images(self, html):
        """
        Извлекает изображения вида:

        data:image/png;base64,...
        """
        pattern = re.compile(
            r'data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)',
            re.IGNORECASE
        )

        images = []

        for match in pattern.finditer(html):
            try:
                encoded = re.sub(r"\s+", "", match.group(1))
                image_bytes = base64.b64decode(encoded)

                images.append(image_bytes)

            except Exception as e:
                print(f"Ошибка декодирования data URI: {e}")

        return images

    def analyze_images(self, images):
        """
        OCR для списка изображений.

        images:
            [{"source": "...", "bytes": b"..."}]

        Возвращает список результатов.
        """
        results = []

        for image in images:
            text = self.image_to_text(image["bytes"])

            results.append({
                "source": image.get("source", "unknown"),
                "text": text,
                "has_text": bool(text)
            })

        return results