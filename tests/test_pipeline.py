from email.message import EmailMessage

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
    assert "ai_score" in reports[0]["analysis"]
    assert (tmp_path / "all.json").exists()
    assert summarize(reports)["count"] == 1
