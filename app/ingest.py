"""Knowledge-base ingestion: Markdown -> header-aware chunks -> embeddings -> ChromaDB.

Run:  python -m app.ingest
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import Settings, get_settings

log = logging.getLogger("ask_saurav.ingest")

HEADERS = [("#", "h1"), ("##", "h2")]


def load_documents(settings: Settings) -> list[Document]:
    """Split every Markdown file by its headings first (so chunks never straddle two topics),
    then into overlapping windows. The heading path is kept as metadata for source citations
    and prepended to the chunk text so the embedding knows what the chunk is about."""
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=True)
    window_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n- ", "\n", ". ", " ", ""],
    )

    files = sorted(Path(settings.data_dir).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown files found in {settings.data_dir}")

    chunks: list[Document] = []
    for path in files:
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        for section in sections:
            h1, h2 = section.metadata.get("h1", ""), section.metadata.get("h2", "")
            title = " › ".join(t for t in (h1, h2) if t)
            for i, piece in enumerate(window_splitter.split_text(section.page_content)):
                chunk_id = hashlib.md5(f"{path.name}|{title}|{i}".encode()).hexdigest()[:16]
                chunks.append(
                    Document(
                        page_content=f"[{title}]\n{piece}",
                        metadata={"source": path.name, "section": title or path.stem, "chunk": i, "id": chunk_id},
                    )
                )
    return chunks


class FastEmbedEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 exported to ONNX and run with onnxruntime via fastembed.

    Same weights as the sentence-transformers model, but no PyTorch: the whole service fits in a
    512 MB free-tier container and query embedding stays fast on a fraction of a CPU."""

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int | None = 1) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads)

    @staticmethod
    def _unit(vectors) -> list[list[float]]:
        import numpy as np

        arr = np.asarray(list(vectors), dtype="float32")
        arr /= np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)  # cosine == dot product
        return arr.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._unit(self._model.embed(texts, batch_size=32))

    def embed_query(self, text: str) -> list[float]:
        return self._unit(self._model.embed([text]))[0]


def get_embeddings(settings: Settings) -> Embeddings:
    return FastEmbedEmbeddings(settings.embed_model, cache_dir=str(settings.model_cache_dir))


def build_index(settings: Settings, embeddings: Embeddings):
    """(Re)build the Chroma collection from scratch so deleted facts never linger."""
    from langchain_chroma import Chroma

    docs = load_documents(settings)
    store = Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    existing = store.get(include=[])["ids"]
    if existing:
        store.delete(ids=existing)
    store.add_documents(docs, ids=[d.metadata["id"] for d in docs])
    log.info("Indexed %d chunks from %s into %s", len(docs), settings.data_dir, settings.chroma_dir)
    return store


def open_index(settings: Settings, embeddings: Embeddings):
    """Open the persisted index, building it first if it is missing or empty."""
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )
    if not store.get(limit=1, include=[])["ids"]:
        store = build_index(settings, embeddings)
    return store


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = get_settings()
    t0 = time.perf_counter()
    build_index(s, get_embeddings(s))
    log.info("Done in %.1fs", time.perf_counter() - t0)
