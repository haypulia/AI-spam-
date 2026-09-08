from email_ai_detector.data.dataset import EmailRecord


def test_ai_char_fraction_uses_intervals():
    record = EmailRecord(id="em", text="a" * 100, ai_char_intervals=[(0, 25), (50, 75)])
    assert abs(record.ai_char_fraction - 0.5) < 1e-9


def test_ocr_text_from_image_payload():
    record = EmailRecord(id="em", image={"ocr_text": "CLAIM NOW"})
    assert record.ocr_text == "CLAIM NOW"


def test_from_dict_defaults_binary_label():
    record = EmailRecord.from_dict({"id": "em", "label": "ai"})
    assert record.ai_binary == 1
