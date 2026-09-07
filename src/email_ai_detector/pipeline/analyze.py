import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..data.eml import LoadedEmail, iter_eml_files, load_email_from_file
from ..features.ocr import ImageOCR
from ..scoring.base import Scorer
from ..utils import ensure_dir, get_timestamp, save_json, truncate_text

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


class EmailAnalyzer:
    def __init__(
        self,
        scorer: Scorer,
        ocr: Optional[ImageOCR] = None,
        results_dir: Optional[PathLike] = None,
        summary_path: Optional[PathLike] = None,
        category_threshold: float = 0.5,
    ):
        self.scorer = scorer
        self.ocr = ocr or ImageOCR()
        self.results_dir = Path(results_dir) if results_dir else None
        self.summary_path = Path(summary_path) if summary_path else None
        self.category_threshold = category_threshold

    def analyze_email(self, email: LoadedEmail) -> Dict[str, object]:
        ocr_results = self.ocr.analyze_images(email.images) if email.images else []
        ocr_text = "\n\n".join(item["text"] for item in ocr_results if item["has_text"])

        result = self.scorer.score_email(
            text=email.text,
            subject=email.subject,
            html=email.html,
            ocr_text=ocr_text,
        )

        return {
            "file": Path(email.path).name if email.path else "",
            "path": email.path or "",
            "timestamp": get_timestamp(),
            "email": {
                "subject": email.subject,
                "sender": email.sender,
                "headers": email.headers,
                "html_preview": truncate_text(email.html, 500),
                "images_count": len(email.images),
            },
            "ocr": {"results": [{k: v for k, v in item.items()} for item in ocr_results]},
            "analysis": result.to_dict(self.category_threshold),
        }

    def analyze_file(self, path: PathLike) -> Optional[Dict[str, object]]:
        email = load_email_from_file(path)
        if email is None:
            logger.warning("не удалось загрузить письмо: %s", path)
            return None
        return self.analyze_email(email)

    def analyze_directory(
        self,
        directory: PathLike,
        patterns: Sequence[str] = ("*.eml",),
        limit: Optional[int] = None,
        save_individual: bool = True,
    ) -> List[Dict[str, object]]:
        files = list(iter_eml_files(directory, patterns))
        if limit:
            files = files[:limit]

        if not files:
            logger.warning("в каталоге %s нет писем", directory)
            return []

        reports: List[Dict[str, object]] = []
        for index, path in enumerate(files, start=1):
            logger.info("[%d/%d] %s", index, len(files), path.name)
            report = self.analyze_file(path)
            if report is None:
                continue
            reports.append(report)
            if save_individual and self.results_dir:
                ensure_dir(self.results_dir)
                save_json(report, self.results_dir / ("%s.json" % path.stem))

        if self.summary_path:
            save_json(reports, self.summary_path)

        return reports


def summarize(reports: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not reports:
        return {"count": 0}
    scores = [report["analysis"]["ai_score"] for report in reports]
    verdicts: Dict[str, int] = {}
    for report in reports:
        verdict = report["analysis"]["verdict"]
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    return {
        "count": len(reports),
        "mean_ai_score": sum(scores) / len(scores),
        "max_ai_score": max(scores),
        "verdicts": verdicts,
    }
