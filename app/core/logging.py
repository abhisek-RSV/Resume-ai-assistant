import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    # Chroma and httpx are noisy at INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
