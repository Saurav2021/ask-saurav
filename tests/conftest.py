"""Offline test fixtures: deterministic fake embeddings + a fake chat model, so the whole API
can be tested in CI without a Groq key or downloading an embedding model."""

import itertools

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.config import Settings
from app.ingest import build_index
from app.main import create_app
from app.rag import RAGService

FAKE_ANSWER = "I built **AURA**, a four-stage explainable pipeline for women's safety surveillance."


def fake_llm(text: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=itertools.cycle([AIMessage(text)]))


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        groq_api_key="test",
        chroma_dir=tmp_path / "chroma",
        db_path=tmp_path / "logs.sqlite3",
        collection="test_kb",
        rate_limit_per_minute=5,
        admin_token="secret",
    )


@pytest.fixture
def service(settings) -> RAGService:
    store = build_index(settings, DeterministicFakeEmbedding(size=64))
    return RAGService(settings, store, fake_llm(FAKE_ANSWER), fake_llm("What is AURA?"))


@pytest.fixture
def client(settings, service):
    with TestClient(create_app(settings, service)) as c:
        yield c
