import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass
class Document:
    text: str
    metadata: dict = field(default_factory=dict)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_file(path: Path) -> list[Document]:
    """Load a file into Documents: one per PDF page, one per text file."""
    source = path.name
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        docs = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(text, {"source": source, "page": page_num}))
        return docs

    text = path.read_text(encoding="utf-8").strip()
    return [Document(text, {"source": source, "page": 1})] if text else []


def discover_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        logger.warning("Documents directory %s does not exist", directory)
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
