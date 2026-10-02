"""End-to-end answer evaluation against the real LLM (needs GROQ_API_KEY).

Checks, for every question:
  * keyword recall  - the facts that must appear (e.g. "87.0%", "8.79") are in the answer
  * first person    - the answer speaks as Saurav ("I", "my") rather than about him
  * guardrails      - out-of-scope / injection / private-data questions are declined safely

    GROQ_API_KEY=... python -m eval.answer_eval
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from app.config import Settings
from app.ingest import build_index, get_embeddings
from app.rag import RAGService, build_llms

HERE = Path(__file__).parent
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|my|me)\b")


def jsonl(name: str) -> list[dict]:
    return [json.loads(x) for x in (HERE / name).read_text().splitlines() if x.strip()]


async def run() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(chroma_dir=Path(tmp) / "chroma", collection="eval")
        if not s.groq_api_key:
            print("GROQ_API_KEY not set - skipping answer eval.")
            return 0
        llm, rewriter = build_llms(s)
        svc = RAGService(s, build_index(s, get_embeddings(s)), llm, rewriter)

        rows, kw_hits, kw_total, fp = [], 0, 0, 0
        for item in jsonl("questions.jsonl"):
            ans = (await svc.ainvoke(item["q"], []))["answer"]
            await asyncio.sleep(1.5)  # stay inside Groq free-tier rate limits
            found = [k for k in item["keywords"] if k.lower() in ans.lower()]
            kw_hits += len(found)
            kw_total += len(item["keywords"])
            fp += bool(FIRST_PERSON.search(ans))
            rows.append((item["q"], f"{len(found)}/{len(item['keywords'])}", ans.replace("\n", " ")[:160]))

        g_pass, g_rows = 0, []
        for item in jsonl("guardrails.jsonl"):
            ans = (await svc.ainvoke(item["q"], []))["answer"]
            await asyncio.sleep(1.5)
            low = ans.lower()
            ok = all(bad.lower() not in low for bad in item.get("must_not_include", []))
            if item.get("must_include_any"):
                ok &= any(good.lower() in low for good in item["must_include_any"])
            g_pass += ok
            g_rows.append((item["q"], "✓" if ok else "✗", ans.replace("\n", " ")[:160]))

    n, g = len(rows), len(g_rows)
    md = [
        "# Answer evaluation",
        "",
        f"LLM: `{s.llm_model}` · embeddings: `{s.embed_model}`",
        "",
        "| Keyword recall | First-person answers | Guardrails passed |",
        "|---|---|---|",
        f"| {kw_hits / kw_total:.1%} | {fp}/{n} | {g_pass}/{g} |",
        "",
        "| Question | Keywords | Answer (truncated) |",
        "|---|---|---|",
        *[f"| {q} | {k} | {a} |" for q, k, a in rows],
        "",
        "| Guardrail probe | Pass | Answer (truncated) |",
        "|---|---|---|",
        *[f"| {q} | {k} | {a} |" for q, k, a in g_rows],
    ]
    (HERE / "results_answers.md").write_text("\n".join(md) + "\n")
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice title=Answer eval::keyword recall {kw_hits / kw_total:.1%}, first-person {fp}/{n}, "
              f"guardrails {g_pass}/{g}")
        for q, ok, a in g_rows:
            if ok != "✓":
                print(f"::warning title=Guardrail failed::{q} | {a}")
    print("\n".join(md))
    return 0


def main() -> int:
    try:
        return asyncio.run(run())
    except Exception as e:  # report the real cause (bad key, rate limit, ...) instead of a bare traceback
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error title=Answer eval failed ({type(e).__name__})::{str(e)[:600]}")
        raise


if __name__ == "__main__":
    sys.exit(main())
