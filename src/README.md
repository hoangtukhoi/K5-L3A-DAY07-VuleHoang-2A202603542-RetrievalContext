# Code Day 7 — Agentic RAG trên corpus Day 6

| File | Vai trò |
|---|---|
| `src/retrieval.py` | 3 phương pháp: dense (pgvector Day 6), BM25 (tokenizer âm tiết / bigram), hybrid RRF |
| `src/agent.py` | Agent tool-calling (OpenRouter), gộp context có ngân sách, citation, từ chối khi thiếu evidence |
| `src/llm.py` | Client OpenRouter: retry, `max_tokens`, đếm token/chi phí |
| `src/data.py` | Xuất câu hỏi từ benchmark Day 6 + 4 câu thiếu evidence; định nghĩa hit |
| `scripts/eval_retrieval.py` | Hit@1 / Evidence Hit@3 / MRR cho từng phương pháp |
| `scripts/run_agent.py` | Chạy agent thật cho từng cấu hình, lưu trace |
| `scripts/summarize_agent.py`, `scripts/grade_test.py` | Tổng hợp tự động và nhãn chấm tay (test) |

## Chạy lại

Cần: Postgres + pgvector của Day 6 đang chạy và đã nạp (`docker compose up -d`, `python scripts/ingest.py` trong repo Day 6), `OPENROUTER_API_KEY` và `DATABASE_URL` trong `starter/.env` của Day 6 (đường dẫn đặt bằng biến `DAY6_DIR`, mặc định là thư mục anh em), và Python của venv Day 6 (psycopg, sentence-transformers).

```bash
export PYTHONIOENCODING=utf-8 PYTHONPATH=.
# 1. xuất chunk từ pgvector -> data/chunks.json (xem report), dựng data/questions.json bằng src.data.build_questions()
python scripts/eval_retrieval.py --split dev
python scripts/run_agent.py --split dev                      # 5 cấu hình
python scripts/summarize_agent.py dev
# chốt cấu hình (results/selection_frozen.md) rồi mới chạy test:
python scripts/eval_retrieval.py --split test --final
python scripts/run_agent.py --split test --final --configs base two_tools
python scripts/grade_test.py
```
`data/chunks.json` (trích từ giáo trình Day 6) không nằm trong git; `data/questions.json` được commit.
