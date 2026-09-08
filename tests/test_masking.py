from email_ai_detector.data.dataset import EmailRecord
from email_ai_detector.evaluation.masking import MASKS, apply_mask

RECORD = EmailRecord(
    id="em_test",
    label="ai",
    ai_binary=1,
    subject="Exclusive opportunity - act now",
    text="We are pleased to inform you. Moreover, the offer is limited.\n\nBest regards, Team",
    html='<p style="color:red">We are pleased.</p><!-- Generated -->',
)


def test_every_mask_preserves_label():
    for name in MASKS:
        masked = apply_mask(name, [RECORD])[0]
        assert masked.ai_binary == RECORD.ai_binary
        assert masked.id == RECORD.id


def test_markup_cleanup_removes_service_markup():
    masked = apply_mask("markup_cleanup", [RECORD])[0]
    assert "<!--" not in masked.html
    assert "style=" not in masked.html


def test_boilerplate_removal_drops_signature():
    masked = apply_mask("boilerplate_removal", [RECORD])[0]
    assert "Best regards" not in masked.text


def test_combined_mask_changes_text():
    masked = apply_mask("combined_evasion", [RECORD])[0]
    assert masked.text != RECORD.text
