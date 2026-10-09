import base64
import logging
import re
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

PathLike = Union[str, Path]

DATA_URI_PATTERN = re.compile(r"data:(image/[^;]+);base64,([A-Za-z0-9+/=\s]+)", re.IGNORECASE)

HEADER_FIELDS = ("From", "To", "Subject", "Date", "Reply-To", "Return-Path", "X-Mailer", "Message-ID")
logger = logging.getLogger(__name__)


class InvalidEmailError(ValueError):
    pass


@dataclass
class LoadedEmail:
    path: Optional[str] = None
    subject: str = ""
    sender: str = ""
    html: str = ""
    text: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    images: List[dict] = field(default_factory=list)

    @property
    def identifier(self) -> str:
        return Path(self.path).name if self.path else self.subject

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "subject": self.subject,
            "sender": self.sender,
            "html": self.html,
            "text": self.text,
            "headers": self.headers,
            "images_count": len(self.images),
        }


def _decode_header(value: Optional[str], fallback: str = "") -> str:
    if not value:
        return fallback
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value.encode("utf-8", errors="ignore").decode("utf-8")


def _part_content(part) -> str:
    try:
        return part.get_content()
    except Exception:
        payload = part.get_payload(decode=True)
        if not payload:
            return ""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="ignore")


def extract_data_uri_images(html: str) -> List[Tuple[str, bytes]]:
    images = []
    for match in DATA_URI_PATTERN.finditer(html or ""):
        try:
            encoded = re.sub(r"\s+", "", match.group(2))
            images.append((match.group(1).lower(), base64.b64decode(encoded)))
        except Exception:
            continue
    return images


def extract_images(message, html: str = "") -> List[dict]:
    images = []
    for part in message.walk():
        content_type = part.get_content_type()
        if not content_type.startswith("image/"):
            continue
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            payload = None
        if not payload:
            continue
        content_id = part.get("Content-ID")
        images.append(
            {
                "source": part.get_filename() or content_id or content_type,
                "bytes": payload,
                "content_type": content_type,
                "content_id": content_id,
                "inline": bool(content_id),
            }
        )

    for index, (content_type, payload) in enumerate(extract_data_uri_images(html)):
        images.append(
            {
                "source": "data_uri_%d" % index,
                "bytes": payload,
                "content_type": content_type,
                "content_id": None,
                "inline": True,
            }
        )

    return images


def load_email_from_bytes(raw: bytes, path: Optional[str] = None) -> LoadedEmail:
    if not raw.strip():
        raise InvalidEmailError("пустой файл")
    message = BytesParser(policy=policy.default).parsebytes(raw)
    structural_defects = {
        "MissingHeaderBodySeparatorDefect", "NoBoundaryInMultipartDefect",
        "StartBoundaryNotFoundDefect", "CloseBoundaryNotFoundDefect",
        "MultipartInvariantViolationDefect",
    }
    for part in message.walk():
        for defect in part.defects:
            if type(defect).__name__ in structural_defects:
                raise InvalidEmailError("повреждённая структура MIME: " + type(defect).__name__)
    if not any(message.get(name) for name in HEADER_FIELDS):
        raise InvalidEmailError("не найдены заголовки письма")

    html = ""
    text = ""
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/html" and not html:
                html = _part_content(part)
            elif content_type == "text/plain" and not text:
                text = _part_content(part)
    else:
        content = _part_content(message)
        if message.get_content_type() == "text/html":
            html = content
        else:
            text = content

    if not text:
        try:
            body = message.get_body(preferencelist=("plain",))
            text = _part_content(body) if body is not None else ""
        except Exception:
            text = ""

    headers = {name: _decode_header(message.get(name)) for name in HEADER_FIELDS if message.get(name)}
    if not text.strip() and not html.strip():
        raise InvalidEmailError("нет текста или HTML для анализа")

    return LoadedEmail(
        path=path,
        subject=_decode_header(message.get("Subject"), "Без темы"),
        sender=_decode_header(message.get("From"), "Неизвестный отправитель"),
        html=html or "",
        text=text or "",
        headers=headers,
        images=extract_images(message, html or ""),
    )


def load_email_from_file(path: PathLike) -> Optional[LoadedEmail]:
    path = Path(path)
    try:
        with open(path, "rb") as handle:
            return load_email_from_bytes(handle.read(), path=str(path))
    except Exception as error:
        reason = str(error) if isinstance(error, (InvalidEmailError, OSError, LookupError)) else type(error).__name__
        logger.warning("письмо пропущено при чтении/разборе: %s | %s", path.resolve(), reason)
        return None


def iter_eml_files(directory: PathLike, patterns=("*.eml",), recursive: bool = True) -> Iterator[Path]:
    directory = Path(directory)
    if not directory.exists():
        return
    for pattern in patterns:
        glob = directory.rglob(pattern) if recursive else directory.glob(pattern)
        for path in sorted(glob):
            if path.is_file():
                yield path
