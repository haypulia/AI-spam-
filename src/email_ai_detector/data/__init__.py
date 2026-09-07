from .eml import LoadedEmail, load_email_from_bytes, load_email_from_file, iter_eml_files
from .dataset import (
    EmailRecord,
    DatasetSplit,
    extract_dataset_archive,
    load_dataset,
    load_split,
)

__all__ = [
    "LoadedEmail",
    "load_email_from_bytes",
    "load_email_from_file",
    "iter_eml_files",
    "EmailRecord",
    "DatasetSplit",
    "extract_dataset_archive",
    "load_dataset",
    "load_split",
]
