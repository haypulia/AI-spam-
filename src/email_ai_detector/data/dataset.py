import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..utils import read_jsonl

PathLike = Union[str, Path]

SPLITS = ("train", "validation", "test")


@dataclass
class EmailRecord:
    id: str
    lang: str = ""
    label: str = ""
    ai_origin: str = ""
    ai_binary: int = 0
    is_spam: int = 0
    data_type: str = ""
    model: Optional[str] = None
    prompt_type: Optional[str] = None
    subject: str = ""
    text: str = ""
    html: str = ""
    ai_elements: List[str] = field(default_factory=list)
    ai_char_intervals: List[Sequence[int]] = field(default_factory=list)
    has_image: bool = False
    image: Optional[dict] = None
    mime_headers: Dict[str, str] = field(default_factory=dict)
    split: str = ""
    eml_path: Optional[str] = None

    @property
    def ocr_text(self) -> str:
        if not self.image:
            return ""
        return self.image.get("ocr_text", "") or ""

    @property
    def is_mixed(self) -> bool:
        return self.label == "mixed"

    @property
    def ai_char_fraction(self) -> float:
        if not self.text:
            return 0.0
        covered = sum(max(0, int(end) - int(start)) for start, end in self.ai_char_intervals)
        return min(1.0, covered / len(self.text))

    @classmethod
    def from_dict(cls, row: dict, split: str = "") -> "EmailRecord":
        return cls(
            id=row.get("id") or row.get("topic_id") or "",
            lang=row.get("lang", ""),
            label=row.get("label", ""),
            ai_origin=row.get("ai_origin", ""),
            ai_binary=int(row.get("ai_binary", 0 if row.get("label") == "human" else 1)),
            is_spam=int(row.get("is_spam", 0)),
            data_type=row.get("data_type", ""),
            model=row.get("model"),
            prompt_type=row.get("prompt_type"),
            subject=row.get("subject", "") or "",
            text=row.get("text", "") or "",
            html=row.get("html", "") or "",
            ai_elements=list(row.get("ai_elements") or []),
            ai_char_intervals=[tuple(item) for item in (row.get("ai_char_intervals") or [])],
            has_image=bool(row.get("has_image", False)),
            image=row.get("image"),
            mime_headers=dict(row.get("mime_headers") or {}),
            split=row.get("split", split),
            eml_path=row.get("eml_path"),
        )


@dataclass
class DatasetSplit:
    name: str
    records: List[EmailRecord]

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> List[int]:
        return [record.ai_binary for record in self.records]

    def filter(self, predicate) -> "DatasetSplit":
        return DatasetSplit(self.name, [record for record in self.records if predicate(record)])


def extract_dataset_archive(archive: PathLike, target_dir: PathLike, force: bool = False) -> Path:
    archive = Path(archive)
    target_dir = Path(target_dir)
    root = target_dir / "ai_assisted_spam"
    if root.exists() and not force:
        return root
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target_dir)
    return root


def load_dataset(dataset_root: PathLike, filename: str = "emails.jsonl") -> List[EmailRecord]:
    path = Path(dataset_root) / filename
    return [EmailRecord.from_dict(row) for row in read_jsonl(path)]


def load_split(dataset_root: PathLike, split: str) -> DatasetSplit:
    if split not in SPLITS and split != "all":
        raise ValueError("unknown split: %s" % split)
    records = load_dataset(dataset_root)
    if split == "all":
        return DatasetSplit("all", records)
    return DatasetSplit(split, [record for record in records if record.split == split])


def load_splits(dataset_root: PathLike) -> Dict[str, DatasetSplit]:
    records = load_dataset(dataset_root)
    grouped: Dict[str, List[EmailRecord]] = {name: [] for name in SPLITS}
    for record in records:
        grouped.setdefault(record.split, []).append(record)
    return {name: DatasetSplit(name, items) for name, items in grouped.items()}


def split_sizes(dataset_root: PathLike) -> List[Tuple[str, int]]:
    return [(name, len(split)) for name, split in load_splits(dataset_root).items()]
