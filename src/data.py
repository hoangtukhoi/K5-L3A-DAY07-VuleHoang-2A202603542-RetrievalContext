"""Build/load Day 7 data: chunks exported from Day 6 pgvector (strategy B) and the 24+4 question benchmark."""
import json
import re
import unicodedata

from .config import DATA, DAY6

UNANSWERABLE = [
    {"id": "U01", "split": "dev", "question": "Lớp sealed trong Java dùng để làm gì và khai báo ra sao?",
     "absence_reason": "Giáo trình không đề cập sealed class; quét toàn bộ 620 chunk không có 'sealed' hay 'lớp kín'."},
    {"id": "U02", "split": "dev", "question": "Biểu thức lambda trong Java có cú pháp như thế nào?",
     "absence_reason": "Không có 'lambda' ở bất kỳ chunk nào; giáo trình dừng ở Java trước khi có lambda."},
    {"id": "U03", "split": "test", "question": "Phương thức mặc định (default method) trong interface Java hoạt động ra sao?",
     "absence_reason": "Giáo trình nói về interface nhưng không có 'default method'; chữ 'default' chỉ xuất hiện trong lệnh switch."},
    {"id": "U04", "split": "test", "question": "Dùng JUnit viết kiểm thử đơn vị cho một lớp Java như thế nào?",
     "absence_reason": "Không có 'JUnit' hoặc 'kiểm thử đơn vị' trong toàn bộ corpus."},
]


def squash(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text or ""))


def build_questions() -> list[dict]:
    qs = []
    for line in (DAY6 / "benchmark" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        qs.append({"id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
                   "expected_answer": q["expected_answer"], "answerable": True, "filter": q.get("filter"),
                   "gold": [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]})
    for u in UNANSWERABLE:
        qs.append({**u, "type": "unanswerable", "answerable": False, "expected_behavior": "search_and_abstain",
                   "filter": None})
    return qs


def load_chunks() -> list[dict]:
    return json.loads((DATA / "chunks.json").read_text(encoding="utf-8"))


def load_questions(split: str | None = None) -> list[dict]:
    qs = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    return [q for q in qs if split in (None, "all", q["split"])]


def is_hit(question: dict, chunk: dict) -> bool:
    """Chunk is a hit when it is on a gold source+page and contains a whole gold evidence quote."""
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"]
               and squash(g["evidence"]) in squash(chunk["text"]) for g in question["gold"])
