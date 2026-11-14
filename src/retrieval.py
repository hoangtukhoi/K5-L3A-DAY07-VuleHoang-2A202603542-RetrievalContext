"""Three retrieval methods over the Day 6 chunks: dense (pgvector), lexical (BM25), hybrid (RRF).

Every method returns dicts: id, text, source_id, source, page, metadata, score, rank.
The optional filter {'chapter': N} is user-provided context and is applied identically to all methods.
"""
import math
import os
import re
from collections import Counter

import psycopg

from .config import EMBEDDING_MODEL, STRATEGY

STOP = {"là", "và", "của", "có", "một", "các", "những", "được", "trong", "cho", "với", "để", "này", "khi",
        "thì", "gì", "nào", "như", "thế", "không", "bị", "ra", "đã", "sẽ", "cũng", "hay", "hoặc", "vì", "sao"}


def tok_syllable(text: str) -> list[str]:
    """Baseline: each Vietnamese syllable is one token."""
    return re.findall(r"\w+", text.lower())


def tok_bigram(text: str) -> list[str]:
    """Improved: drop stopwords, add syllable bigrams so compounds ('đối tượng', 'cơ bản') survive."""
    syl = [t for t in tok_syllable(text) if t not in STOP]
    return syl + [f"{a}_{b}" for a, b in zip(syl, syl[1:])]


def _pack(chunk: dict, score: float, rank: int) -> dict:
    return {**{k: chunk[k] for k in ("id", "text", "source_id", "source", "page", "metadata")},
            "score": round(float(score), 5), "rank": rank}


def _filter_ok(chunk: dict, filters: dict | None) -> bool:
    return all(chunk["metadata"].get(k) == v for k, v in (filters or {}).items())


class BM25Retriever:
    def __init__(self, chunks: list[dict], tokenizer=tok_syllable, k1: float = 1.5, b: float = 0.75, name: str = "bm25"):
        self.chunks, self.tok, self.k1, self.b, self.name = chunks, tokenizer, k1, b, name
        self.docs = [Counter(tokenizer(c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(chunks)
        self.df = Counter(t for d in self.docs for t in d)

    def search(self, query: str, *, top_k: int = 3, filters: dict | None = None) -> list[dict]:
        n, q = len(self.chunks), set(self.tok(query))
        scored = []
        for i, (c, d) in enumerate(zip(self.chunks, self.docs)):
            if not _filter_ok(c, filters):
                continue
            s = 0.0
            for t in q:
                if t in d:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * d[t] * (self.k1 + 1) / (d[t] + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [_pack(self.chunks[i], s, r) for r, (s, i) in enumerate(scored[:top_k], 1) if s > 0]


class DenseRetriever:
    """Exact cosine search in the Day 6 pgvector table, with the same e5 model/prefixes used at ingestion."""
    name = "dense"

    def __init__(self, database_url: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.conn = psycopg.connect(database_url or os.environ["DATABASE_URL"], connect_timeout=5)

    def embed_query(self, q: str) -> list[float]:
        return self.model.encode(["query: " + q], normalize_embeddings=True, show_progress_bar=False)[0].tolist()

    def search(self, query: str, *, top_k: int = 3, filters: dict | None = None) -> list[dict]:
        vec = "[" + ",".join(f"{x:.8f}" for x in self.embed_query(query)) + "]"
        where, params = "strategy = %s", [STRATEGY]
        if filters:
            if set(filters) - {"chapter"}:
                raise ValueError("only the 'chapter' filter is supported")
            where += " AND chapter = %s"
            params.append(filters["chapter"])
        rows = self.conn.execute(
            f"SELECT chunk_id, document_id, source, page, chapter, chapter_title, text, 1 - (embedding <=> %s::vector) "
            f"FROM chunks WHERE {where} ORDER BY embedding <=> %s::vector LIMIT %s", [vec, *params, vec, top_k]).fetchall()
        return [_pack({"id": r[0], "source_id": r[1], "source": r[2], "page": r[3], "text": r[6],
                       "metadata": {"chapter": r[4], "chapter_title": r[5], "document_id": r[1], "page": r[3]}}, r[7], k)
                for k, r in enumerate(rows, 1)]


class HybridRetriever:
    """Reciprocal Rank Fusion of several retrievers: score = sum 1/(k + rank). Rank-based, so no score normalisation needed."""

    def __init__(self, retrievers: list, name: str = "hybrid", pool: int = 20, rrf_k: int = 60):
        self.retrievers, self.name, self.pool, self.rrf_k = retrievers, name, pool, rrf_k

    def search(self, query: str, *, top_k: int = 3, filters: dict | None = None) -> list[dict]:
        fused, by_id = Counter(), {}
        for r in self.retrievers:
            for h in r.search(query, top_k=self.pool, filters=filters):
                fused[h["id"]] += 1 / (self.rrf_k + h["rank"])
                by_id[h["id"]] = h
        order = sorted(fused, key=lambda i: (-fused[i], i))[:top_k]
        return [{**by_id[i], "score": round(fused[i], 5), "rank": k} for k, i in enumerate(order, 1)]
