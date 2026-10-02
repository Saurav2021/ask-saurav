"""Retrieval evaluation: does the right knowledge-base file come back for each question?

Metrics: Hit@1, Hit@k (k = settings.top_k) and MRR over eval/questions.jsonl.
Writes eval/results_retrieval.md and exits non-zero if Hit@k falls below --min-hit (used in CI).

    python -m eval.retrieval_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from app.config import Settings
from app.ingest import build_index, get_embeddings

HERE = Path(__file__).parent


def load_questions() -> list[dict]:
    return [json.loads(line) for line in (HERE / "questions.jsonl").read_text().splitlines() if line.strip()]


def evaluate(settings: Settings, embeddings, k: int) -> dict:
    store = build_index(settings, embeddings)
    rows, hit1, hitk, rr = [], 0, 0, 0.0
    t0 = time.perf_counter()
    for item in load_questions():
        # plain similarity ranking (not MMR) so ranks are comparable across questions
        docs = store.similarity_search(item["q"], k=k)
        ranked = [d.metadata["source"] for d in docs]
        rank = ranked.index(item["source"]) + 1 if item["source"] in ranked else None
        hit1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0
        rows.append((item["q"], item["source"], rank, docs[0].metadata["section"] if docs else ""))
    n = len(rows)
    return {
        "n": n, "k": k,
        "hit@1": hit1 / n, f"hit@{k}": hitk / n, "mrr": rr / n,
        "ms_per_query": (time.perf_counter() - t0) * 1000 / n,
        "rows": rows,
    }


def to_markdown(r: dict, model: str) -> str:
    k = r["k"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"Embedding model: `{model}` · questions: {r['n']} · k = {k}",
        "",
        f"| Hit@1 | Hit@{k} | MRR | ms / query |",
        "|---|---|---|---|",
        f"| {r['hit@1']:.3f} | {r[f'hit@{k}']:.3f} | {r['mrr']:.3f} | {r['ms_per_query']:.1f} |",
        "",
        "| Question | Expected file | Rank | Top retrieved section |",
        "|---|---|---|---|",
    ]
    lines += [f"| {q} | `{src}` | {rank or '✗'} | {top} |" for q, src, rank, top in r["rows"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hit", type=float, default=0.85, help="fail if Hit@k is below this")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(chroma_dir=Path(tmp) / "chroma", collection="eval")
        result = evaluate(s, get_embeddings(s), k=s.top_k)
        md = to_markdown(result, s.embed_model)

    (HERE / "results_retrieval.md").write_text(md)
    print(md)
    ok = result[f"hit@{s.top_k}"] >= args.min_hit
    print("PASS" if ok else f"FAIL: Hit@{s.top_k} below {args.min_hit}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
