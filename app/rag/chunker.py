import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import Document


def _normalize(text: str) -> str:
    # Collapse runs of spaces/tabs but keep paragraph breaks so the splitter has
    # natural boundaries to cut on
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        for piece in splitter.split_text(_normalize(doc.text)):
            chunks.append(Document(piece, dict(doc.metadata)))
    return chunks
