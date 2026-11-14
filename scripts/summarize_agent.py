"""Summarise automatic agent metrics per config/split (no semantic judgement; manual labels are separate).

  python scripts/summarize_agent.py dev
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RESULTS  # noqa: E402


def summarize(traces):
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if t["answerable"]]
    una = [t for t in ok if not t["answerable"]]
    n = len(ans)
    return {
        "runs": f"{len(ok)}/{len(traces)} ok",
        "tool_use": f"{sum(t['searches'] > 0 for t in ok)}/{len(ok)}",
        "searches_dist": dict(sorted(Counter(t["searches"] for t in ok).items())),
        "context_evidence_rate": f"{sum(bool(t['context_has_evidence']) for t in ans)}/{n}",
        "answered": f"{sum(t['status'] == 'answered' for t in ans)}/{n}",
        "refused_answerable": f"{sum(t['status'] == 'abstain' for t in ans)}/{n}",
        "abstain_correct_auto": f"{sum(t['status'] == 'abstain' for t in una)}/{len(una)}",
        "citation_in_context": f"{sum(len(t['citations']) for t in ok)}/{sum(len(t['citations']) + len(t['invalid_citations']) for t in ok)}",
        "avg_context_chars": round(sum(t["context_chars"] for t in ok) / len(ok)),
        "avg_seconds": round(sum(t["seconds"] for t in ok) / len(ok), 2),
        "total_cost_usd": round(sum(t["usage"]["cost"] for t in ok), 5),
        "avg_prompt_tokens": round(sum(t["usage"]["prompt_tokens"] for t in ok) / len(ok)),
    }


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    out = {}
    for f in sorted((RESULTS / "agent").glob(f"*_{split}.json")):
        out[f.stem.removesuffix(f"_{split}")] = summarize(json.loads(f.read_text(encoding="utf-8"))["traces"])
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False))
    (RESULTS / f"agent_summary_{split}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
