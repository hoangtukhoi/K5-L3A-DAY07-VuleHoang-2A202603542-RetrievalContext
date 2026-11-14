"""Shared config: Day 6 environment (DB url, embedding model) and OpenRouter settings."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
# Day 6 repo holds the .env (DATABASE_URL, OPENROUTER_API_KEY) and the pgvector service.
DAY6 = Path(os.getenv("DAY6_DIR", ROOT.parent / "K5-L3-DAY06-VuLeHoang-2A202603542-DataIngestionEngineering"))
load_dotenv(ROOT / ".env")
load_dotenv(DAY6 / "starter" / ".env")

STRATEGY = "B"  # Day 6 chosen chunking strategy (paragraph, max 800 chars)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "intfloat/multilingual-e5-small"
LLM_MODEL = os.getenv("LLM_MODEL") or "anthropic/claude-haiku-5.5"
