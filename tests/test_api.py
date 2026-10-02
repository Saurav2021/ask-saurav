import json

from tests.conftest import FAKE_ANSWER


def parse_sse(text: str) -> list[dict]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_suggestions(client):
    assert len(client.get("/suggestions").json()["suggestions"]) >= 4


def test_chat_returns_grounded_answer_with_sources(client):
    r = client.post("/chat", json={"question": "What is AURA?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] == FAKE_ANSWER
    assert body["sources"] and {"source", "section"} <= set(body["sources"][0])


def test_stream_emits_sources_then_tokens_then_done(client):
    r = client.post("/chat/stream", json={"question": "What is AURA?", "history": []})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "sources" and types[-1] == "done" and "token" in types
    assert "".join(e["text"] for e in events if e["type"] == "token") == FAKE_ANSWER
    assert "standalone_question" not in events[0]  # internal detail is not leaked to the browser


def test_follow_up_uses_history(client):
    history = [{"role": "user", "content": "Tell me about AURA"}, {"role": "assistant", "content": "AURA is..."}]
    r = client.post("/chat", json={"question": "What FPS does it run at?", "history": history})
    assert r.status_code == 200


def test_rejects_empty_and_too_long_questions(client):
    assert client.post("/chat", json={"question": "   "}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 501}).status_code == 422


def test_rate_limit(client):
    codes = [client.post("/chat", json={"question": f"q{i}"}).status_code for i in range(7)]
    assert codes[:5] == [200] * 5 and 429 in codes[5:]


def test_analytics_requires_token_and_aggregates_with_sql(client):
    client.post("/chat", json={"question": "What is AURA?", "session_id": "s1"})
    client.post("/chat/stream", json={"question": "Skills?", "session_id": "s1"})
    assert client.get("/analytics").status_code == 401
    stats = client.get("/analytics", headers={"X-Admin-Token": "secret"}).json()
    assert stats["questions"] == 2 and stats["sessions"] == 1 and stats["unique_visitors"] == 1
    assert stats["top_sections"] and stats["questions_per_day"][0]["questions"] == 2


def test_llm_failure_degrades_gracefully(client, service):
    class Boom:
        async def astream(self, *a, **k):
            raise RuntimeError("groq down")
            yield  # pragma: no cover

    service._answer_chain = Boom()
    r = client.post("/chat", json={"question": "hi"})
    assert r.status_code == 503 and "sauravsuz@gmail.com" in r.json()["answer"]
