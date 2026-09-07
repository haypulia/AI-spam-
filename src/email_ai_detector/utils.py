import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Union

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8", errors="ignore").decode("utf-8")


def clean_json_data(obj: Any) -> Any:
    if isinstance(obj, str):
        return clean_string(obj)
    if isinstance(obj, dict):
        return {clean_string(key): clean_json_data(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_json_data(item) for item in obj]
    return obj


def save_json(data: Any, path: PathLike, clean: bool = True) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    payload = clean_json_data(data) if clean else data
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def load_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: PathLike) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict], path: PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def truncate_text(text: Any, max_len: int = 500) -> str:
    text = clean_string(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
