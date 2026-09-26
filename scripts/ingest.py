"""Build or refresh the Chroma knowledge base from the documents directory.

Usage:
    uv run python -m scripts.ingest           # only re-embed changed files
    uv run python -m scripts.ingest --force   # re-embed everything
"""

import argparse

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.rag.pipeline import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-embed all documents")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    report = RAGService.from_settings(settings).ingest(force=args.force)
    print(
        f"Ingested: {report.ingested or '-'}\n"
        f"Skipped (unchanged): {report.skipped or '-'}\n"
        f"Removed: {report.removed or '-'}\n"
        f"Total chunks: {report.total_chunks}"
    )


if __name__ == "__main__":
    main()
