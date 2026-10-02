"""FastAPI service: streaming chat (SSE), JSON chat, health, suggestions and SQL analytics.

Run locally:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app import __version__
from app.config import Settings, get_settings
from app.db import ChatLog
from app.rag import RAGService

log = logging.getLogger("ask_saurav")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

SUGGESTIONS = [
    "Tell me about yourself",
    "What is AURA?",
    "Explain your SCRBM research",
    "What did you do at L&T Smart City?",
    "What's your Gen AI / LLM experience?",
    "How did you build this chatbot?",
    "Are you open to relocation?",
    "How can I contact you?",
]

FALLBACK = (
    "Sorry, my AI brain hit a snag just now. Please try again in a moment, "
    "or email the real me at sauravsuz@gmail.com."
)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[Turn] = Field(default_factory=list, max_length=40)
    session_id: str | None = Field(default=None, max_length=64)

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question is empty")
        return v


class RateLimiter:
    """Sliding-window limiter per visitor. In-memory is fine for a single free-tier container."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        return True


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")  # Render (and most PaaS hosts) sit behind a proxy
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def default_service(settings: Settings) -> RAGService:
    from app.ingest import get_embeddings, open_index
    from app.rag import build_llms

    llm, rewriter = build_llms(settings)
    return RAGService(settings, open_index(settings, get_embeddings(settings)), llm, rewriter)


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or default_service(settings)
        app.state.chatlog = ChatLog(settings.db_path, settings.ip_hash_salt)
        app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
        if not settings.groq_api_key and service is None:
            log.warning("GROQ_API_KEY is not set: chat requests will fail until it is configured.")
        log.info("Ask Saurav %s ready (llm=%s)", __version__, settings.llm_model)
        yield

    app = FastAPI(
        title="Ask Saurav API",
        version=__version__,
        description="RAG chatbot that answers questions about Saurav Kumar in his own voice.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Admin-Token"],
    )

    def check_request(req: ChatRequest, request: Request) -> tuple[ChatRequest, str]:
        if len(req.question) > settings.max_question_chars:
            raise HTTPException(422, f"Please keep questions under {settings.max_question_chars} characters.")
        ip = client_ip(request)
        if not request.app.state.limiter.allow(ip):
            raise HTTPException(429, "You're asking faster than I can think! Please wait a minute.")
        return req, ip

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__, "llm": settings.llm_model, "embeddings": settings.embed_model}

    @app.get("/suggestions")
    async def suggestions():
        return {"suggestions": SUGGESTIONS}

    @app.post("/chat")
    async def chat(request: Request, checked=Depends(check_request)):
        req, ip = checked
        svc: RAGService = request.app.state.service
        history = [t.model_dump() for t in req.history]
        try:
            result = await svc.ainvoke(req.question, history)
            status = "ok"
        except Exception:
            log.exception("chat failed")
            result, status = {"answer": FALLBACK, "sources": [], "latency_ms": None}, "error"
        request.app.state.chatlog.log(
            session_id=req.session_id, ip=ip, question=req.question,
            standalone_question=result.get("standalone_question"), answer=result["answer"],
            sources=result.get("sources", []), latency_ms=result.get("latency_ms"), status=status,
        )
        return JSONResponse({"answer": result["answer"], "sources": result.get("sources", []),
                             "latency_ms": result.get("latency_ms")}, status_code=200 if status == "ok" else 503)

    @app.post("/chat/stream")
    async def chat_stream(request: Request, checked=Depends(check_request)):
        req, ip = checked
        svc: RAGService = request.app.state.service
        history = [t.model_dump() for t in req.history]

        async def events():
            answer, meta, status = [], {}, "ok"
            try:
                async for ev in svc.astream(req.question, history):
                    if ev["type"] == "token":
                        answer.append(ev["text"])
                    else:
                        meta.update(ev)
                    public = {k: v for k, v in ev.items() if k != "standalone_question"}
                    yield f"data: {json.dumps(public)}\n\n"
            except Exception:
                log.exception("stream failed")
                status = "error"
                yield f"data: {json.dumps({'type': 'error', 'text': FALLBACK})}\n\n"
            finally:
                request.app.state.chatlog.log(
                    session_id=req.session_id, ip=ip, question=req.question,
                    standalone_question=meta.get("standalone_question"), answer="".join(answer),
                    sources=meta.get("sources", []), latency_ms=meta.get("latency_ms"), status=status,
                )

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/analytics")
    async def analytics(request: Request, days: int = 30, x_admin_token: str | None = Header(default=None)):
        if not settings.admin_token or x_admin_token != settings.admin_token:
            raise HTTPException(401, "Admin token required.")
        return request.app.state.chatlog.analytics(days)

    # Standalone chat page + the embeddable widget, served from the same container.
    if FRONTEND.exists():
        app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(FRONTEND / "index.html")

    return app


app = create_app()
