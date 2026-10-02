"""Central configuration. Every value can be overridden with an environment variable
(or a .env file), e.g. LLM_MODEL=openai/gpt-oss-20b."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (Groq free tier) ---
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"        # answers
    rewrite_model: str = "openai/gpt-oss-20b"     # fast follow-up question rewriting
    temperature: float = 0.3
    max_answer_tokens: int = 700

    # --- Embeddings + vector store ---
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"   # ONNX build, served by fastembed
    model_cache_dir: Path = ROOT / "storage" / "models"
    data_dir: Path = ROOT / "data"
    chroma_dir: Path = ROOT / "storage" / "chroma"
    collection: str = "saurav_kb"
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4
    fetch_k: int = 12
    mmr_lambda: float = 0.6

    # --- API ---
    db_path: Path = ROOT / "storage" / "chat_logs.sqlite3"
    allowed_origins: list[str] = Field(
        default_factory=lambda: [
            "https://saurav2021.github.io",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
        ]
    )
    rate_limit_per_minute: int = 12
    max_question_chars: int = 500
    max_history_turns: int = 6
    admin_token: str = ""          # protects /analytics; empty = analytics disabled
    ip_hash_salt: str = "ask-saurav"


@lru_cache
def get_settings() -> Settings:
    return Settings()
