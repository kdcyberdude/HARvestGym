"""
search_endpoints tool — semantic search over endpoint embeddings from browser_agent.

Embeds the query via the HuggingFace Inference API (same model used by browser_agent
to embed the endpoint catalog), then returns the top-k matches by cosine similarity.
Falls back to keyword (BM25-style term overlap) if embeddings are unavailable.
"""

from __future__ import annotations

import numpy as np


def search_endpoints(query: str, episode_store: dict) -> list[str]:
    """
    Semantic search over endpoint embeddings built by browser_agent.

    Args:
        query: Natural language query (e.g. "create guest cart", "add item to cart")
        episode_store: Mutable dict containing embeddings + chunks from browser_agent.

    Returns:
        List of up to 3 endpoint schema text strings.
    """
    chunks: list[str] = episode_store.get("endpoint_chunks", [])
    embeddings = episode_store.get("endpoint_embeddings")

    if not chunks:
        return ["No endpoint index available. Call browser_agent(task, url) first."]

    # Semantic search path — requires embeddings from HF API
    if embeddings is not None and hasattr(embeddings, "__len__") and len(embeddings) > 0:
        try:
            from .browser_agent import embed_query_via_api
            q_emb = embed_query_via_api(query)  # shape (1, D) or None

            if q_emb is not None:
                # Cosine similarity (both sides already L2-normalized)
                scores = (embeddings @ q_emb.T).flatten()
                top_k = min(3, len(scores))
                top_indices = np.argsort(scores)[::-1][:top_k]
                results = [chunks[int(i)] for i in top_indices]
                print(
                    f"[search_endpoints] Semantic: top scores "
                    f"{[round(float(scores[i]), 3) for i in top_indices]}",
                    flush=True,
                )
                return results
        except Exception as e:
            print(f"[search_endpoints] Semantic search failed: {e}. Using keyword fallback.", flush=True)

    # Keyword fallback — term overlap scoring
    print("[search_endpoints] Using keyword fallback.", flush=True)
    query_terms = query.lower().split()
    scored: list[tuple[float, str]] = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1.0 for t in query_terms if t in chunk_lower)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [c for _, c in scored[:3]]
    return results if results else chunks[:3]
