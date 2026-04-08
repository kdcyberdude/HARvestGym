"""
search_episode_data tool — BM25 + semantic search over accumulated episode response data.

Searches all request/response bodies from prior curl_exec calls in this episode.
"""

from __future__ import annotations

import json
import re
from typing import Any


def search_episode_data(query: str, episode_store: dict) -> list[dict]:
    """
    Hybrid BM25 + keyword search over episode accumulated response bodies.

    Args:
        query: Keyword or natural language query (e.g. "Radiant Tee sku", "_csrf_token")
        episode_store: Per-episode store containing bm25_corpus and bm25_metadata

    Returns:
        Top-5 matching JSON objects from episode history, annotated with step info
    """
    corpus: list[str] = episode_store.get("bm25_corpus", [])
    metadata: list[dict] = episode_store.get("bm25_metadata", [])

    if not corpus:
        return [{"note": "No episode data yet. Make API calls with curl_exec() first."}]

    # Try BM25 ranking
    try:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [_tokenize(doc) for doc in corpus]
        tokenized_query = _tokenize(query)
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        # Get top 5 by BM25 score
        import numpy as np
        top_k = min(5, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                meta = metadata[idx]
                result = {
                    "step": idx + 1,
                    "url": meta.get("url", ""),
                    "method": meta.get("method", ""),
                    "status_code": meta.get("status_code", 0),
                    "data": meta.get("response_body"),
                }
                results.append(result)

        if results:
            return results

    except ImportError:
        pass
    except Exception as e:
        print(f"[search_episode_data] BM25 error: {e}", flush=True)

    # Fallback: keyword match
    query_lower = query.lower()
    query_terms = query_lower.split()
    results = []
    for idx, doc in enumerate(corpus):
        if any(term in doc.lower() for term in query_terms):
            meta = metadata[idx]
            results.append({
                "step": idx + 1,
                "url": meta.get("url", ""),
                "method": meta.get("method", ""),
                "status_code": meta.get("status_code", 0),
                "data": meta.get("response_body"),
            })
    return results[:5] if results else [{"note": f"No results found for: {query}"}]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_\-\.]+", text)
    return tokens if tokens else [""]
