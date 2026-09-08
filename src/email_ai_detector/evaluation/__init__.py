from .explanations import evaluate_explanations
from .masking import MASKS, apply_mask
from .metrics import (
    average_precision_score,
    bootstrap_ci,
    roc_auc_score,
    roc_curve,
    threshold_metrics,
)
from .partial import evaluate_partial
from .report import render_markdown, save_markdown
from .runner import evaluate_detection, evaluate_masking, plot_roc_curve, run_evaluation, save_report

__all__ = [
    "evaluate_explanations",
    "MASKS",
    "apply_mask",
    "average_precision_score",
    "bootstrap_ci",
    "roc_auc_score",
    "roc_curve",
    "threshold_metrics",
    "evaluate_partial",
    "render_markdown",
    "save_markdown",
    "evaluate_detection",
    "evaluate_masking",
    "plot_roc_curve",
    "run_evaluation",
    "save_report",
]
