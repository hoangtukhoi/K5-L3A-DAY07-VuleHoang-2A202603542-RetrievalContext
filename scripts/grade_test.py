"""Hand-graded labels for the TEST split (base vs two_tools) and aggregation with starter/agent_metrics.py.

Labels were produced by reading every answer and opening each cited chunk (data/chunks.json):
  answer_grade: 0 wrong/missing evidence, 1 partly right, 2 right and complete (BENCHMARK.md)
  citation_support: one bool per citation, in the order of trace['citations']; True only if the cited chunk supports the claim
  context_grade: 0 no evidence, 1 evidence but missing a condition/source, 2 enough evidence and source
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
from agent_metrics import aggregate  # noqa: E402
from src.config import RESULTS  # noqa: E402
from src.data import load_questions  # noqa: E402

T, F = True, False
# qid: (answer_grade, [citation_support...], context_grade, note)
LABELS = {
    "base": {
        "T01": (2, [T, T, T], 2, ""),
        "T02": (2, [T, T], 2, ""),
        "T03": (2, [T, T], 2, ""),
        "T04": (2, [T, F], 2, "p193-B01 là bài tập đúng/sai, không nói finally luôn chạy -> citation không hỗ trợ"),
        "T05": (2, [T, T, F], 2, "p194-B00 là đề bài tập, không chứng minh quy tắc thứ tự catch -> citation không hỗ trợ"),
        "T06": (2, [T, T], 1, "Context có p192-B00 (RuntimeException không bắt buộc) nhưng thiếu câu 'và các lớp con'"),
        "T07": (2, [T, T], 2, "Evidence tương đương nằm ở trang 201 (gold ghi trang 199); đoạn trích gold không có trong context"),
        "T08": (2, [T, T, T], 2, ""),
        "T09": (2, [T, T, T], 2, ""),
        "U03": (None, [T], None, "Từ chối đúng, nêu rõ interface chỉ có phương thức trừu tượng"),
        "U04": (None, [], None, "Từ chối đúng, không trích dẫn"),
    },
    "two_tools": {
        "T01": (2, [T, T, T], 2, ""),
        "T02": (2, [T, T], 2, ""),
        "T03": (2, [T, T], 2, ""),
        "T04": (2, [T, T], 2, ""),
        "T05": (2, [T, T, T], 2, ""),
        "T06": (2, [T, T], 2, ""),
        "T07": (2, [T, T], 2, "Evidence tương đương ở trang 201; đoạn gold trang 199 không có trong context"),
        "T08": (2, [T, T, T, T, T], 2, ""),
        "T09": (2, [T, T, T], 2, ""),
        "U03": (None, [T], None, "Từ chối đúng"),
        "U04": (None, [T, T], None, "Từ chối đúng; hai citation chỉ hỗ trợ các câu hẹp (test phần mềm nói chung)"),
    },
}


def main():
    cases = [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in load_questions("test")]
    out = {}
    for cfg, labels in LABELS.items():
        traces = json.loads((RESULTS / "agent" / f"{cfg}_test.json").read_text(encoding="utf-8"))["traces"]
        reviews = []
        for t in traces:
            grade, support, _, note = labels[t["query_id"]]
            r = {"query_id": t["query_id"], "citation_support": support, "note": note}
            if t["answerable"]:
                r["answer_grade"] = 0 if t["status"] == "abstain" else grade
            else:
                r["abstention_correct"] = t["status"] == "abstain"
            reviews.append(r)
        agg = aggregate(cases, traces, reviews)
        ctx = [labels[q["id"]][2] for q in load_questions("test") if q["answerable"]]
        agg["context_grade"] = {str(g): ctx.count(g) for g in (0, 1, 2)}
        agg["context_grade"]["mean"] = round(sum(ctx) / len(ctx), 2)
        out[cfg] = {"aggregate": agg, "reviews": reviews}
        print(cfg, json.dumps(agg, ensure_ascii=False))
    (RESULTS / "agent_test_manual_labels.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
