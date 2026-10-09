import io

import pytest
from email.message import EmailMessage
import json
from unittest import TestCase
from unittest.mock import patch

from email_ai_detector.data.eml import load_email_from_bytes
from email_ai_detector.pipeline.analyze import EmailAnalyzer, summarize
from email_ai_detector.scoring.heuristic import HeuristicScorer


def build_message() -> bytes:
    message = EmailMessage()
    message["Subject"] = "Тестовое письмо"
    message["From"] = "sender@example.org"
    message.set_content("Обычный текст письма.")
    message.add_alternative("<html><body><p>Обычный текст письма.</p></body></html>", subtype="html")
    return message.as_bytes()


def test_email_parsing_extracts_html_and_headers():
    email = load_email_from_bytes(build_message())
    assert email.subject == "Тестовое письмо"
    assert "<p>" in email.html
    assert email.headers["From"] == "sender@example.org"


def test_analyzer_produces_report(tmp_path):
    path = tmp_path / "sample.eml"
    path.write_bytes(build_message())

    analyzer = EmailAnalyzer(scorer=HeuristicScorer(), results_dir=tmp_path / "out", summary_path=tmp_path / "all.json")
    reports = analyzer.analyze_directory(tmp_path)

    assert len(reports) == 1
    assert "ai_score" in reports[0]["analysis"]["text"]
    assert reports[0]["analysis"]["images"] == []
    assert reports[0]["analysis"]["attachments"] == []
    assert (tmp_path / "all.json").exists()
    assert summarize(reports)["count"] == 1


def test_empty_directory_saves_empty_json(tmp_path):
    output = tmp_path / "all.json"
    output.write_text('["old report"]', encoding="utf-8")
    analyzer = EmailAnalyzer(scorer=HeuristicScorer(), summary_path=output)
    assert analyzer.analyze_directory(tmp_path) == []
    assert json.loads(output.read_text(encoding="utf-8")) == []


def test_batch_preserves_reports_with_same_filename(tmp_path):
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "sample.eml").write_bytes(build_message())
    results = tmp_path / "results"
    summary = tmp_path / "all.json"
    analyzer = EmailAnalyzer(scorer=HeuristicScorer(), results_dir=results, summary_path=summary)
    reports = analyzer.analyze_directory(tmp_path)
    assert len(reports) == 2
    assert len(json.loads(summary.read_text(encoding="utf-8"))) == 2
    for name in ("first", "second"):
        report = json.loads((results / name / "sample.json").read_text(encoding="utf-8"))
        assert report["path"] == str(tmp_path / name / "sample.eml")


def test_batch_skips_invalid_emails_and_logs_reasons(tmp_path):
    invalid = {
        "01_empty.eml": b"",
        "02_garbage.eml": b"not an email",
        "03_mime.eml": b'From: a@example.org\r\nContent-Type: multipart/mixed; boundary="missing"\r\n\r\nbroken',
        "04_no_body.eml": b"From: a@example.org\r\n\r\n",
        "05_charset.eml": b'From: a@example.org\r\nContent-Type: text/plain; charset="unknown-charset"\r\n\r\nbody',
    }
    for name, content in invalid.items():
        (tmp_path / name).write_bytes(content)
    (tmp_path / "00_valid.eml").write_bytes(build_message())
    (tmp_path / "99_valid.eml").write_bytes(build_message())
    analyzer = EmailAnalyzer(HeuristicScorer(), results_dir=tmp_path / "out", summary_path=tmp_path / "all.json")
    with TestCase().assertLogs("email_ai_detector", level="WARNING") as captured:
        reports = analyzer.analyze_directory(tmp_path)
    assert [report["file"] for report in reports] == ["00_valid.eml", "99_valid.eml"]
    assert len(captured.output) == len(invalid)
    for name in invalid:
        assert any(str(tmp_path / name) in entry for entry in captured.output)
        assert not (tmp_path / "out" / name).with_suffix(".json").exists()
    assert len(json.loads((tmp_path / "all.json").read_text(encoding="utf-8"))) == 2
    assert (tmp_path / "out" / "99_valid.json").exists()


def test_batch_continues_after_analysis_error(tmp_path):
    for name in ("01.eml", "02.eml", "03.eml"):
        (tmp_path / name).write_bytes(build_message())
    scorer = HeuristicScorer()
    original = scorer.score_email
    calls = 0

    def score(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("private email content")
        return original(**kwargs)

    with patch.object(scorer, "score_email", side_effect=score):
        with TestCase().assertLogs("email_ai_detector", level="WARNING") as captured:
            reports = EmailAnalyzer(scorer).analyze_directory(tmp_path)
    assert [report["file"] for report in reports] == ["01.eml", "03.eml"]
    assert "02.eml" in captured.output[0]
    assert "ValueError" in captured.output[0]
    assert "private email content" not in captured.output[0]


def test_email_without_subject_or_sender_is_valid():
    email = load_email_from_bytes(b"To: recipient@example.org\r\n\r\nValid body")
    assert email.text.strip() == "Valid body"


def test_images_get_their_own_verdict(tmp_path):
    pytest.importorskip("PIL.Image")
    from PIL import Image

    from email_ai_detector.scoring.artifacts import ArtifactRegistry
    from email_ai_detector.scoring.image import ImageAnalyzer

    buffer = io.BytesIO()
    Image.new("RGB", (120, 60), (20, 80, 160)).save(buffer, format="PNG")

    message = EmailMessage()
    message["Subject"] = "Письмо с картинкой"
    message["From"] = "sender@example.org"
    message.set_content("Текст письма.")
    message.add_attachment(buffer.getvalue(), maintype="image", subtype="png", filename="banner.png")

    path = tmp_path / "with_image.eml"
    path.write_bytes(message.as_bytes())

    analyzer = EmailAnalyzer(scorer=HeuristicScorer(), registry=ArtifactRegistry([ImageAnalyzer()]))
    report = analyzer.analyze_file(path)

    images = report["analysis"]["images"]
    assert len(images) == 1
    assert images[0]["source"] == "banner.png"
    assert images[0]["content_type"] == "image/png"
    assert 0.0 <= images[0]["ai_score"] <= 1.0
    assert "image" in report["analysis"]["zones"]
