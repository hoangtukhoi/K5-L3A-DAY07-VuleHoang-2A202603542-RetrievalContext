"""Run the real LLM agent for one or more configurations on a split and save traces.

  python scripts/run_agent.py --split dev --configs base multi_search
  python scripts/run_agent.py --split test --final --configs base chosen      (only after freezing the config)

Gold is used here only AFTER the run to compute Context Evidence (is a valid evidence chunk in the final context?).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import Agent, AgentConfig  # noqa: E402
from src.config import LLM_MODEL, RESULTS  # noqa: E402
from src.data import is_hit, load_chunks, load_questions  # noqa: E402
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, tok_bigram  # noqa: E402

CONFIGS = {
    # Baseline: a real agent with ONE dense search (top-3, 2400 chars ~ 3 chunks).
    "base": dict(tools=["dense"], top_k=3, max_searches=1, budget=2400,
                 notes={"hypothesis": "reference: single dense retrieval"}),
    # Improvement 1: allow query reformulation (up to 3 searches).
    "multi_search": dict(tools=["dense"], top_k=3, max_searches=3, budget=2400,
                         notes={"hypothesis": "re-searching with a reformulated query recovers missed evidence"}),
    # Improvement 2: hybrid dense+BM25(bigram) retrieval with RRF.
    "hybrid": dict(tools=["hybrid"], top_k=3, max_searches=1, budget=2400,
                   notes={"hypothesis": "lexical signal fixes dense misses on identifiers/terms"}),
    # Improvement 3: keep neighbouring chunks, bigger context budget.
    "neighbors": dict(tools=["dense"], top_k=3, max_searches=1, budget=4000, neighbors=1,
                      notes={"hypothesis": "evidence cut by chunking lies in the adjacent chunk"}),
    # Improvement 4: two tools (semantic + keyword), model picks, up to 3 searches.
    "two_tools": dict(tools=["dense", "bm25"], top_k=3, max_searches=3, budget=3600,
                      notes={"hypothesis": "model can pick keyword search for exact terms"}),
}


def build_retrievers(chunks):
    dense = DenseRetriever()
    bm25_bi = BM25Retriever(chunks, tokenizer=tok_bigram, name="bm25_bigram")
    return {"dense": dense, "bm25": bm25_bi, "hybrid": HybridRetriever([dense, bm25_bi], name="hybrid_rrf_bigram")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--only", nargs="*", help="restrict to these question ids")
    ap.add_argument("--retry-errors", action="store_true", help="re-run only runs that ended in error or produced an empty answer and merge into the saved file")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("Refusing to run test without --final.")
    chunks, questions = load_chunks(), load_questions(args.split)
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    retrievers = build_retrievers(chunks)
    out_dir = RESULTS / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.configs:
        cfg = AgentConfig(name=name, model=LLM_MODEL, **CONFIGS[name])
        agent, traces = Agent(cfg, retrievers, chunks), []
        path = out_dir / f"{name}_{args.split}.json"
        todo = questions
        if args.retry_errors and path.exists():
            traces = json.loads(path.read_text(encoding="utf-8"))["traces"]
            bad = {t["query_id"] for t in traces if t["status"] == "error" or not t["answer"].strip()}
            traces = [t for t in traces if t["query_id"] not in bad]
            todo = [q for q in questions if q["id"] in bad]
        for q in todo:
            r = agent.run(q["question"], filters=q["filter"])     # question text + user filter only
            ctx_hit = None
            if q["answerable"]:
                ctx_hit = any(is_hit(q, {"source_id": "root_data", **c}) for c in r["context"])
            traces.append({"query_id": q["id"], "split": q["split"], "answerable": q["answerable"], "config": name,
                           "question": q["question"], "context_has_evidence": ctx_hit, **r})
            print(f"{name:13s} {q['id']} {r['status']:8s} searches={r['searches']} ctx_evidence={ctx_hit} "
                  f"cost=${r['usage']['cost']:.5f} {r['seconds']}s", flush=True)
        traces.sort(key=lambda t: t["query_id"])
        path.write_text(
            json.dumps({"config": cfg.__dict__, "traces": traces}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
