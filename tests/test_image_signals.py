import io
import random

import pytest

from email_ai_detector.features.image_signals import (
    DIMENSION_FEATURES,
    IMAGE_FEATURE_NAMES,
    extract_image_features,
    image_metadata,
)

Image = pytest.importorskip("PIL.Image")
PngImagePlugin = pytest.importorskip("PIL.PngImagePlugin")


def encode(image, **params) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **params)
    return buffer.getvalue()


def flat_banner() -> bytes:
    image = Image.new("RGB", (240, 120), (32, 96, 200))
    for x in range(40, 200):
        for y in range(50, 70):
            image.putpixel((x, y), (255, 255, 255))
    return encode(image)


def noisy_photo() -> bytes:
    generator = random.Random(7)
    image = Image.new("RGB", (240, 120))
    image.putdata([(generator.randrange(256), generator.randrange(256), generator.randrange(256)) for _ in range(240 * 120)])
    return encode(image)


def test_feature_set_is_complete_and_numeric():
    features = extract_image_features(flat_banner())
    assert set(features) == set(IMAGE_FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features.values())
    assert features["image_decoded"] == 1.0


def test_broken_payload_yields_zeros():
    features = extract_image_features(b"definitely not an image")
    assert features["image_decoded"] == 0.0
    assert set(value for value in features.values()) == {0.0}


def test_flat_image_differs_from_noise():
    flat = extract_image_features(flat_banner())
    noise = extract_image_features(noisy_photo())
    assert flat["image_flat_area_ratio"] > noise["image_flat_area_ratio"]
    assert noise["image_noise_level"] > flat["image_noise_level"]
    assert noise["image_unique_color_ratio"] > flat["image_unique_color_ratio"]
    assert noise["image_color_entropy"] > flat["image_color_entropy"]


def test_software_tag_is_detected():
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "SomeEditor 1.0")
    payload = encode(Image.new("RGB", (32, 32), (10, 10, 10)), pnginfo=info)
    features = extract_image_features(payload)
    assert features["image_has_software_tag"] == 1.0
    assert image_metadata(payload)["software"] == "SomeEditor 1.0"


def test_metadata_reports_geometry_and_format():
    metadata = image_metadata(flat_banner())
    assert metadata["decoded"] is True
    assert metadata["format"] == "PNG"
    assert (metadata["width"], metadata["height"]) == (240, 120)
    assert image_metadata(b"broken") == {"decoded": False, "bytes": 6}


def test_dimension_features_are_declared_for_exclusion():
    assert set(DIMENSION_FEATURES) <= set(IMAGE_FEATURE_NAMES)
    assert extract_image_features(flat_banner())["image_aspect_ratio"] == pytest.approx(2.0)
