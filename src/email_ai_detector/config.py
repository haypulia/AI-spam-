import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


def _detect_project_root() -> Path:
    override = os.getenv("PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return Path.cwd()


PROJECT_ROOT = _detect_project_root()


def _env_path(name: str, default: str, root: Optional[Path] = None) -> Path:
    raw = os.getenv(name, default)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (root or _detect_project_root()) / path
    return path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    project_root: Path = PROJECT_ROOT
    api_key: Optional[str] = None
    base_url: str = "http://deepcode.ci.nsu.ru/api/chat/completions"
    default_model: str = "deepseek-ai/DeepSeek-V4-Flash"
    temperature: float = 0.1
    top_p: float = 0.8
    request_timeout: int = 180
    request_delay: float = 2.0
    max_html_length: int = 25000
    max_block_length: int = 10000
    ocr_languages: str = "rus+eng"
    raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "drweb")
    case_samples_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw" / "case_samples")
    interim_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "interim")
    processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    datasets_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "datasets")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "interim" / "llm_cache")

    @property
    def analysis_results_dir(self) -> Path:
        return self.processed_dir / "analysis_results"

    @property
    def full_analysis_path(self) -> Path:
        return self.processed_dir / "full_analysis.json"

    @property
    def metrics_dir(self) -> Path:
        return self.reports_dir / "metrics"

    @property
    def dataset_root(self) -> Path:
        return self.interim_dir / "ai_assisted_spam"

    @property
    def dataset_archive(self) -> Path:
        return self.datasets_dir / "ai_assisted_spam_dataset.zip"

    @property
    def heuristic_model_path(self) -> Path:
        return self.models_dir / "heuristic_model.json"


def get_settings(env_file: Optional[str] = None) -> Settings:
    load_dotenv(env_file) if env_file else load_dotenv()
    root = _detect_project_root()
    return Settings(
        project_root=root,
        api_key=os.getenv("DEEPCODE_API_KEY"),
        base_url=os.getenv("DEEPCODE_BASE_URL", "http://deepcode.ci.nsu.ru/api/chat/completions"),
        default_model=os.getenv("DEFAULT_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        temperature=_env_float("LLM_TEMPERATURE", 0.1),
        top_p=_env_float("LLM_TOP_P", 0.8),
        request_timeout=_env_int("LLM_REQUEST_TIMEOUT", 180),
        request_delay=_env_float("LLM_REQUEST_DELAY", 2.0),
        max_html_length=_env_int("MAX_HTML_LENGTH", 25000),
        max_block_length=_env_int("MAX_BLOCK_LENGTH", 10000),
        ocr_languages=os.getenv("OCR_LANGUAGES", "rus+eng"),
        raw_dir=_env_path("DRWEB_DATA_DIR", "data/raw/drweb", root),
        case_samples_dir=_env_path("CASE_SAMPLES_DIR", "data/raw/case_samples", root),
        interim_dir=_env_path("INTERIM_DIR", "data/interim", root),
        processed_dir=_env_path("PROCESSED_DIR", "data/processed", root),
        datasets_dir=_env_path("DATASETS_DIR", "data/datasets", root),
        models_dir=_env_path("MODELS_DIR", "models", root),
        reports_dir=_env_path("REPORTS_DIR", "reports", root),
        cache_dir=_env_path("LLM_CACHE_DIR", "data/interim/llm_cache", root),
    )
