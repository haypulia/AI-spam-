import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import get_settings
from .data.dataset import extract_dataset_archive, load_split, load_splits
from .evaluation.calibration import calibrate, load_calibration, save_calibration
from .evaluation.report import save_markdown
from .evaluation.runner import plot_roc_curve, run_evaluation, save_report
from .pipeline.analyze import EmailAnalyzer, summarize
from .scoring import build_scorer
from .scoring.training import train_document_model, train_explainer, train_segment_model
from .utils import save_json

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def command_prepare_data(args, settings) -> int:
    root = extract_dataset_archive(
        args.archive or settings.dataset_archive, args.output or settings.interim_dir, force=args.force
    )
    logging.info("датасет распакован: %s", root)
    for name, split in sorted(load_splits(root).items()):
        positives = sum(split.labels())
        logging.info("%s: писем %d, AI %d, human %d", name, len(split), positives, len(split) - positives)
    return 0


def command_train(args, settings) -> int:
    root = args.dataset or settings.dataset_root
    split = load_split(root, args.split)
    logging.info("обучение на выборке %s, писем: %d", args.split, len(split))

    dataset_name = Path(root).name
    metadata = {"split": args.split, "dataset": dataset_name}

    document_model = train_document_model(split.records, metadata=dict(metadata))
    document_path = document_model.save(args.model_path or settings.heuristic_model_path)
    logging.info("документная модель сохранена: %s", document_path)

    segment_model = train_segment_model(split.records, metadata=dict(metadata))
    segment_path = segment_model.save(args.segment_model_path or settings.models_dir / "segment_model.json")
    logging.info("сегментная модель сохранена: %s", segment_path)

    explainer = train_explainer(split.records, metadata=dict(metadata))
    explainer_path = explainer.save(settings.models_dir / "category_models.json")
    logging.info("модели зон объяснения сохранены: %s", explainer_path)

    if not args.skip_calibration:
        calibration_split = load_split(root, args.calibration_split)
        scorer = build_scorer("heuristic", settings)
        calibration = calibrate(scorer, calibration_split.records)
        calibration["split"] = args.calibration_split
        calibration_path = save_calibration(calibration, settings.models_dir / "calibration.json")
        logging.info(
            "пороги откалиброваны на выборке %s: решение %.2f, зоны %.2f, сегменты %.2f",
            args.calibration_split,
            calibration["decision_threshold"],
            calibration["explanation_threshold"],
            calibration["segment_threshold"],
        )
        logging.info("файл калибровки: %s", calibration_path)

    return 0


def command_analyze(args, settings) -> int:
    scorer = build_scorer(args.scorer, settings, model=args.model)
    analyzer = EmailAnalyzer(
        scorer=scorer,
        results_dir=args.results_dir or settings.analysis_results_dir,
        summary_path=args.output or settings.full_analysis_path,
    )
    reports = analyzer.analyze_directory(
        args.input or settings.raw_dir,
        patterns=tuple(args.patterns),
        limit=args.limit,
    )
    stats = summarize(reports)
    logging.info("обработано писем: %s", stats.get("count", 0))
    for verdict, count in (stats.get("verdicts") or {}).items():
        logging.info("%s: %d", verdict, count)
    return 0


def command_evaluate(args, settings) -> int:
    root = args.dataset or settings.dataset_root
    split = load_split(root, args.split)
    records = split.records[: args.limit] if args.limit else split.records
    scorer = build_scorer(args.scorer, settings, model=args.model)
    calibration = load_calibration(settings.models_dir / "calibration.json")

    threshold = args.threshold if args.threshold is not None else calibration["decision_threshold"]
    explanation_threshold = (
        args.explanation_threshold
        if args.explanation_threshold is not None
        else calibration["explanation_threshold"]
    )
    segment_threshold = (
        args.segment_threshold if args.segment_threshold is not None else calibration["segment_threshold"]
    )

    logging.info("оценка скорера %s на выборке %s, писем: %d", scorer.name, args.split, len(records))

    report = run_evaluation(
        scorer=scorer,
        records=records,
        split_name=args.split,
        threshold=threshold,
        explanation_threshold=explanation_threshold,
        segment_threshold=segment_threshold,
        include_masking=not args.skip_masking,
    )

    prefix = args.prefix or ("metrics_%s" % scorer.name)
    directory = args.output or settings.metrics_dir
    json_path = save_report(report, directory, prefix=prefix)
    image = plot_roc_curve(report, Path(directory) / ("%s_roc_curve.png" % prefix))

    logging.info("AUC-ROC: %.4f", report["detection"]["roc_auc"])
    logging.info("значения метрик: %s", json_path)
    if image:
        logging.info("график ROC: %s", image)

    if args.markdown:
        markdown = save_markdown(report, Path(directory) / ("%s.md" % prefix), image.name if image else None)
        logging.info("отчёт в markdown: %s", markdown)

    return 0


def command_score(args, settings) -> int:
    scorer = build_scorer(args.scorer, settings, model=args.model)
    analyzer = EmailAnalyzer(scorer=scorer)
    report = analyzer.analyze_file(args.path)
    if report is None:
        logging.error("не удалось разобрать письмо: %s", args.path)
        return 1
    if args.output:
        save_json(report, args.output)
    print(report["analysis"]["ai_score_percent"], report["analysis"]["verdict"])
    print(report["analysis"]["explanation"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="email-ai-detector", description="Детектор AI-generated писем")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--env-file", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="распаковать датасет из архива")
    prepare.add_argument("--archive", type=Path, default=None)
    prepare.add_argument("--output", type=Path, default=None)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(handler=command_prepare_data)

    train = subparsers.add_parser("train", help="обучить линейные модели детектора")
    train.add_argument("--dataset", type=Path, default=None)
    train.add_argument("--split", default="train")
    train.add_argument("--model-path", type=Path, default=None)
    train.add_argument("--segment-model-path", type=Path, default=None)
    train.add_argument("--calibration-split", default="validation")
    train.add_argument("--skip-calibration", action="store_true")
    train.set_defaults(handler=command_train)

    analyze = subparsers.add_parser("analyze", help="проанализировать каталог с письмами")
    analyze.add_argument("--input", type=Path, default=None)
    analyze.add_argument("--output", type=Path, default=None)
    analyze.add_argument("--results-dir", type=Path, default=None)
    analyze.add_argument("--scorer", default="heuristic")
    analyze.add_argument("--model", default=None)
    analyze.add_argument("--patterns", nargs="+", default=["*.eml"])
    analyze.add_argument("--limit", type=int, default=None)
    analyze.set_defaults(handler=command_analyze)

    evaluate = subparsers.add_parser("evaluate", help="посчитать метрики на размеченной выборке")
    evaluate.add_argument("--dataset", type=Path, default=None)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--scorer", default="heuristic")
    evaluate.add_argument("--model", default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--threshold", type=float, default=None)
    evaluate.add_argument("--explanation-threshold", type=float, default=None)
    evaluate.add_argument("--segment-threshold", type=float, default=None)
    evaluate.add_argument("--skip-masking", action="store_true")
    evaluate.add_argument("--output", type=Path, default=None)
    evaluate.add_argument("--prefix", default=None)
    evaluate.add_argument("--markdown", action="store_true")
    evaluate.set_defaults(handler=command_evaluate)

    score = subparsers.add_parser("score", help="оценить одно письмо")
    score.add_argument("path", type=Path)
    score.add_argument("--scorer", default="heuristic")
    score.add_argument("--model", default=None)
    score.add_argument("--output", type=Path, default=None)
    score.set_defaults(handler=command_score)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    settings = get_settings(args.env_file)
    return args.handler(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
