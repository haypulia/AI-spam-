import io
import math
from typing import Dict, Optional, Tuple

IMAGE_FEATURE_NAMES: Tuple[str, ...] = (
    "image_decoded",
    "image_width_log",
    "image_height_log",
    "image_aspect_ratio",
    "image_bytes_per_pixel",
    "image_has_alpha",
    "image_is_palette",
    "image_is_grayscale",
    "image_unique_color_ratio",
    "image_dominant_color_share",
    "image_color_entropy",
    "image_flat_area_ratio",
    "image_edge_density",
    "image_noise_level",
    "image_has_metadata",
    "image_has_software_tag",
)

DIMENSION_FEATURES: Tuple[str, ...] = (
    "image_width_log",
    "image_height_log",
    "image_aspect_ratio",
)

SOFTWARE_TAGS = ("software", "creator", "creatortool", "producer", "generator", "comment", "parameters")
EXIF_SOFTWARE_TAG = 305
ANALYSIS_SIZE = 256
FLAT_THRESHOLD = 2.0
EDGE_THRESHOLD = 24.0


def _empty_features() -> Dict[str, float]:
    return {name: 0.0 for name in IMAGE_FEATURE_NAMES}


def load_image(payload: bytes):
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image
    except Exception:
        return None


def _metadata_flags(image) -> Tuple[float, float]:
    info = {str(key).lower(): value for key, value in (getattr(image, "info", {}) or {}).items()}
    has_metadata = 1.0 if info else 0.0
    has_software = 1.0 if any(tag in info for tag in SOFTWARE_TAGS) else 0.0

    exif = None
    try:
        exif = image.getexif()
    except Exception:
        exif = None
    if exif:
        has_metadata = 1.0
        if exif.get(EXIF_SOFTWARE_TAG):
            has_software = 1.0

    return has_metadata, has_software


def _color_features(image) -> Dict[str, float]:
    import numpy as np

    rgb = image.convert("RGB")
    rgb.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    pixels = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if not pixels.size:
        return {
            "image_unique_color_ratio": 0.0,
            "image_dominant_color_share": 0.0,
            "image_color_entropy": 0.0,
        }

    quantized = (pixels >> 3).astype(np.int32)
    codes = quantized[:, 0] * 1024 + quantized[:, 1] * 32 + quantized[:, 2]
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    shares = counts / counts.sum()

    return {
        "image_unique_color_ratio": float(len(counts) / len(codes)),
        "image_dominant_color_share": float(shares.max()),
        "image_color_entropy": float(-(shares * np.log2(shares)).sum()),
    }


def _structure_features(image) -> Dict[str, float]:
    import numpy as np

    grayscale = image.convert("L")
    grayscale.thumbnail((ANALYSIS_SIZE, ANALYSIS_SIZE))
    values = np.asarray(grayscale, dtype=np.float32)
    if values.size < 9:
        return {"image_flat_area_ratio": 0.0, "image_edge_density": 0.0, "image_noise_level": 0.0}

    horizontal = np.abs(np.diff(values, axis=1))
    vertical = np.abs(np.diff(values, axis=0))
    gradient = np.concatenate([horizontal.ravel(), vertical.ravel()])

    blurred = (
        values[:-2, 1:-1] + values[2:, 1:-1] + values[1:-1, :-2] + values[1:-1, 2:] + values[1:-1, 1:-1]
    ) / 5.0
    residual = values[1:-1, 1:-1] - blurred

    return {
        "image_flat_area_ratio": float((gradient < FLAT_THRESHOLD).mean()),
        "image_edge_density": float((gradient > EDGE_THRESHOLD).mean()),
        "image_noise_level": float(residual.std()),
    }


def extract_image_features(payload: bytes) -> Dict[str, float]:
    features = _empty_features()
    image = load_image(payload or b"")
    if image is None:
        return features

    width, height = image.size
    if not width or not height:
        return features

    has_metadata, has_software = _metadata_flags(image)
    features.update(
        {
            "image_decoded": 1.0,
            "image_width_log": math.log1p(width),
            "image_height_log": math.log1p(height),
            "image_aspect_ratio": float(width / height),
            "image_bytes_per_pixel": float(len(payload) / (width * height)),
            "image_has_alpha": 1.0 if image.mode in ("RGBA", "LA", "PA") or "transparency" in (image.info or {}) else 0.0,
            "image_is_palette": 1.0 if image.mode in ("P", "PA") else 0.0,
            "image_is_grayscale": 1.0 if image.mode in ("L", "LA", "1") else 0.0,
            "image_has_metadata": has_metadata,
            "image_has_software_tag": has_software,
        }
    )

    try:
        features.update(_color_features(image))
        features.update(_structure_features(image))
    except Exception:
        pass

    return {name: float(features.get(name, 0.0)) for name in IMAGE_FEATURE_NAMES}


def image_metadata(payload: bytes) -> Dict[str, object]:
    image = load_image(payload or b"")
    if image is None:
        return {"decoded": False, "bytes": len(payload or b"")}

    info = {str(key).lower(): value for key, value in (getattr(image, "info", {}) or {}).items()}
    software: Optional[str] = None
    for tag in SOFTWARE_TAGS:
        value = info.get(tag)
        if isinstance(value, (str, bytes)):
            software = value.decode("utf-8", "ignore") if isinstance(value, bytes) else value
            break

    return {
        "decoded": True,
        "format": image.format or "",
        "mode": image.mode,
        "width": image.size[0],
        "height": image.size[1],
        "bytes": len(payload or b""),
        "software": (software or "")[:120],
    }
