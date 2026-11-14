"""Benchmark retrieval methods on the same chunks/questions (original query, no rewrite).

Reports Hit@1, Evidence Hit@3 (a top-3 chunk is on a gold page AND contains the whole evidence), MRR@20 and latency,
only over answerable questions. Run:  python scripts/eval_retrieval.py --split dev
Test is only run after the configuration is frozen:  --split test --final
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS  # noqa: E402
from src.data import is_hit, load_chunks, load_questions  # noqa: E402
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, tok_bigram  # noqa: E402


def build_methods():
    chunks = load_chunks()
    dense = DenseRetriever()
    bm25 = BM25Retriever(chunks, name="bm25")
    bm25_bi = BM25Retriever(chunks, tokenizer=tok_bigram, name="bm25_bigram")
    return {
        "dense": dense,
        "bm25": bm25,
        "hybrid_rrf": HybridRetriever([dense, bm25], name="hybrid_rrf"),
        "bm25_bigram": bm25_bi,
        "hybrid_rrf_bigram": HybridRetriever([dense, bm25_bi], name="hybrid_rrf_bigram"),
    }


def evaluate(method, questions, use_filter: bool):
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        hits = method.search(q["question"], top_k=20, filters=q["filter"] if use_filter else None)
        ms = (time.perf_counter() - t0) * 1000
        flags = [is_hit(q, h) for h in hits]
        first = flags.index(True) + 1 if any(flags) else None
        rows.append({"id": q["id"], "type": q["type"], "first_hit_rank": first, "ms": round(ms, 1),
                     "top3": [{"id": h["id"], "page": h["page"], "score": h["score"], "hit": f} for h, f in zip(hits[:3], flags)]})
    n = len(rows)
    return rows, {"n": n, "hit_at_1": f"{sum(r['first_hit_rank'] == 1 for r in rows)}/{n}",
                  "hit_at_3": f"{sum(bool(r['first_hit_rank']) and r['first_hit_rank'] <= 3 for r in rows)}/{n}",
                  "mrr_at_20": round(sum(1 / r["first_hit_rank"] for r in rows if r["first_hit_rank"]) / n, 3),
                  "avg_ms": round(sum(r["ms"] for r in rows) / n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test split")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--no-filter", action="store_true", help="ignore the user-provided chapter filter")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("Refusing to run test without --final (configuration must be frozen on dev first).")
    questions = [q for q in load_questions(args.split) if q["answerable"]]
    methods = build_methods()
    out = {"split": args.split, "filter": not args.no_filter, "summary": {}, "per_question": {}}
    for name, m in methods.items():
        if args.methods and name not in args.methods:
            continue
        rows, summ = evaluate(m, questions, not args.no_filter)
        out["summary"][name], out["per_question"][name] = summ, rows
        print(f"{name:20s} {summ}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"retrieval_{args.split}{'_nofilter' if args.no_filter else ''}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
