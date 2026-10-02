import asyncio

from langchain_core.documents import Document

from app.rag import PERSONA_PROMPT, to_messages, unique_sources


def test_persona_prompt_enforces_first_person_grounding_and_honesty():
    p = PERSONA_PROMPT.lower()
    assert "first person" in p
    assert "only the facts in <context>" in p
    assert "be honest" in p and "ai assistant" in p
    assert "{context}" in PERSONA_PROMPT


def test_history_is_truncated_to_last_turns():
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(30)]
    msgs = to_messages(hist, max_turns=3)
    assert len(msgs) == 6 and msgs[-1].content == "29"


def test_unique_sources_dedupes():
    d = Document("x", metadata={"source": "a.md", "section": "A"})
    assert unique_sources([d, d]) == [{"source": "a.md", "section": "A"}]


def test_condense_skips_llm_without_history(service):
    assert asyncio.run(service.condense("What is AURA?", [])) == "What is AURA?"
