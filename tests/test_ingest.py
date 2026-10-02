from app.config import Settings
from app.ingest import load_documents


def test_every_markdown_file_is_chunked():
    s = Settings()
    docs = load_documents(s)
    sources = {d.metadata["source"] for d in docs}
    assert sources == {p.name for p in s.data_dir.glob("*.md")}


def test_chunks_respect_size_and_carry_section_titles():
    s = Settings()
    for d in load_documents(s):
        body = d.page_content.split("\n", 1)[1]
        assert len(body) <= s.chunk_size
        assert d.page_content.startswith("[") and d.metadata["section"]


def test_chunk_ids_are_unique_and_deterministic():
    s = Settings()
    a = [d.metadata["id"] for d in load_documents(s)]
    b = [d.metadata["id"] for d in load_documents(s)]
    assert a == b and len(a) == len(set(a))


def test_knowledge_base_is_written_in_first_person():
    s = Settings()
    text = " ".join(p.read_text() for p in s.data_dir.glob("*.md"))
    assert text.count(" I ") + text.count("I'm") > 30


def test_fastembed_wrapper_returns_unit_vectors(monkeypatch):
    import fastembed
    import numpy as np

    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts, batch_size=32):
            return (np.arange(1, 5, dtype="float32") * (i + 1) for i, _ in enumerate(texts))

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeModel)
    from app.ingest import FastEmbedEmbeddings

    emb = FastEmbedEmbeddings("any")
    docs = emb.embed_documents(["a", "b"])
    assert len(docs) == 2 and all(abs(np.linalg.norm(v) - 1) < 1e-5 for v in docs)
    assert abs(np.linalg.norm(emb.embed_query("q")) - 1) < 1e-5
