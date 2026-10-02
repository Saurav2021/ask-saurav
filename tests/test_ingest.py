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
