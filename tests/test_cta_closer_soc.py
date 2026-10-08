from email_ai_detector.features import extract_features
from email_ai_detector.scoring.explainer import CategoryExplainer, explain_for_soc

def test_cta_and_closer_extraction():
    # Тестовое письмо с явным сгенерированным CTA и шаблоном закрытия
    text = (
        "Уважаемый пользователь!\n\n"
        "Ваш аккаунт требует верификации данных.\n"
        "Срочно перейдите по ссылке и подтвердите вашу учетную запись во избежание блокировки.\n\n"
        "С наилучшими пожеланиями,\n"
        "Служба технической поддержки"
    )
    subject = "Срочно: подтвердите доступ"

    features = extract_features(text=text, subject=subject)

    # Проверка, что экстрактор собрал признаки для cta и closer
    assert features.categories["cta"] > 0.0
    assert features.categories["closer"] > 0.0

    # Проверка наличия в evidence сниппетов
    cta_snippets = features.evidence.get("cta_snippets", [])
    assert len(cta_snippets) > 0
    assert any("подтвердите" in s.lower() for s in cta_snippets)

    closer_snippet = features.evidence.get("closer_snippet", "")
    assert "поддержки" in closer_snippet.lower() or "пожеланиями" in closer_snippet.lower()


def test_soc_explainer_report_generation():
    mock_scores = {
        "subject": 0.35,
        "opener": 0.40,
        "body": 0.30,
        "cta": 0.85,       # Искусственно завышенный скор CTA
        "closer": 0.72,    # Искусственно завышенный скор Closer
        "html_template": 0.10,
    }
    mock_vector = {
        "body_cta_phrase_rate": 1.0,
        "body_urgency_rate": 1.0,
        "closer_signoff": 1.0,
        "closer_capitalized": 1.0,
    }
    mock_evidence = {
        "cta_snippets": ["Срочно перейдите по ссылке и подтвердите аккаунт"],
        "closer_snippet": "С уважением, Администрация сервиса",
    }

    report = explain_for_soc(
        scores=mock_scores,
        vector=mock_vector,
        evidence=mock_evidence,
        threshold=0.60,
    )

    # Проверка структуры отчета для SOC
    assert report.severity == "HIGH"
    assert "cta" in report.flagged_categories
    assert "closer" in report.flagged_categories
    assert report.zones["cta"].is_ai_generated is True
    assert report.zones["closer"].is_ai_generated is True
    assert len(report.zones["cta"].reasons) > 0
    assert len(report.analyst_summary) >= 2

    # Проверка в JSON / Dict
    dict_report = report.to_dict()
    assert isinstance(dict_report, dict)
    assert dict_report["severity"] == "HIGH"