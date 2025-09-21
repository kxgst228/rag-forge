"""Retrieval functions: dense, sparse (BM25), and hybrid.

All three take a query + corpus and return ranked chunk indices with scores.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi


def dense_search(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Cosine similarity search over pre-computed embeddings.

    Returns list of (chunk_index, score) sorted by relevance.
    """
    # embeddings should already be normalized, but just in case
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    corpus_norms = corpus_embeddings / (
        np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-9
    )

    scores = corpus_norms @ query_norm
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def bm25_search(
    query: str,
    corpus_texts: list[str],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """BM25 sparse retrieval. Old school but still hard to beat for keyword-heavy queries.

    Tokenization is just .lower().split() — good enough for English. I tried spacy
    tokenization once and it was 10x slower with marginal improvement.
    """
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    corpus_texts: list[str],
    corpus_embeddings: np.ndarray,
    top_k: int = 5,
    dense_weight: float = 0.7,
) -> list[tuple[int, float]]:
    """Weighted combination of dense + BM25 scores.

    The 0.7/0.3 split worked best in my testing at Observe.AI. Dense handles
    semantic similarity, BM25 catches exact keyword matches that embeddings miss.
    """
    dense_results = dense_search(query_embedding, corpus_embeddings, top_k=len(corpus_texts))
    bm25_results = bm25_search(query, corpus_texts, top_k=len(corpus_texts))

    # normalize scores to [0, 1] for fair combination
    dense_scores = _normalize_scores(dense_results, len(corpus_texts))
    bm25_scores = _normalize_scores(bm25_results, len(corpus_texts))

    combined = {}
    for idx in range(len(corpus_texts)):
        d_score = dense_scores.get(idx, 0.0)
        b_score = bm25_scores.get(idx, 0.0)
        combined[idx] = dense_weight * d_score + (1 - dense_weight) * b_score

    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


def _normalize_scores(results: list[tuple[int, float]], n: int) -> dict[int, float]:
    """Min-max normalize scores to [0, 1]."""
    if not results:
        return {}
    scores = {idx: score for idx, score in results}
    min_s = min(scores.values())
    max_s = max(scores.values())
    rng = max_s - min_s
    if rng == 0:
        return {idx: 1.0 for idx in scores}
    return {idx: (s - min_s) / rng for idx, s in scores.items()}


RETRIEVERS = {
    "dense": "dense",
    "bm25": "bm25",
    "hybrid": "hybrid",
}
