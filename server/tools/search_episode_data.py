"""
search_episode_data — semantic + BM25 search over accumulated episode API responses.

Each curl_exec call stores its full, untruncated response body in episode_store under
``episode_raw_bodies``.  This tool embeds those bodies (via the same HF API used by
browser_agent) and performs cosine-similarity search against the model's query, falling
back to BM25 keyword search when embeddings are unavailable.

Results are returned as compact previews so they fit in the LLM context window:
- Nested trees (e.g. category trees with children_data) are flattened to id+name pairs.
- Large item arrays are shown as a short sample with a total-count note.
- The model can issue more specific queries to drill into any result.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


# ---------------------------------------------------------------------------
# Compact preview helpers
# ---------------------------------------------------------------------------

def _flatten_tree(obj: Any, id_key: str = "id", name_key: str = "name") -> list[dict]:
    """Recursively flatten any nested tree structure into [{id, name}] pairs."""
    results: list[dict] = []
    if isinstance(obj, dict):
        if id_key in obj and name_key in obj:
            results.append({id_key: obj[id_key], name_key: obj[name_key]})
        for v in obj.values():
            results.extend(_flatten_tree(v, id_key, name_key))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(_flatten_tree(item, id_key, name_key))
    return results


def _compact_preview(response_body: Any, max_items: int = 3) -> dict:
    """
    Return a compact, context-friendly preview of a response body.

    - Distilled HTML (has page_type key) → structured summary with forms/products.
    - Nested trees with children_data → flat {id, name} list.
    - Lists / items arrays → short sample + total count.
    - Scalars / errors → returned as-is.
    - The preview always includes a note showing how many objects exist in total.
    """
    if not isinstance(response_body, (dict, list)):
        return {"value": response_body}

    # --- distilled HTML page (from html_distiller) ---
    if isinstance(response_body, dict) and "page_type" in response_body and "forms" in response_body:
        result: dict = {}
        if response_body.get("title"):
            result["page_title"] = response_body["title"]
        # Forms — most actionable: show action URL, method, and fields (strip base64 uenc)
        forms = response_body.get("forms", [])
        if forms:
            clean_forms = []
            for form in forms[:8]:
                fields = {k: v for k, v in form.get("fields", {}).items()
                          if k not in ("uenc",) and len(str(v)) < 100}
                clean_forms.append({
                    "action": form.get("action", ""),
                    "method": form.get("method", "GET"),
                    "fields": fields,
                })
            result["forms"] = clean_forms
        # Data blobs — show top-level keys and compact preview of small blobs
        blobs = response_body.get("data_blobs", [])
        if blobs:
            blob_summary = []
            for blob in blobs[:3]:
                data = blob.get("data")
                if isinstance(data, (dict, list)):
                    s = json.dumps(data)
                    blob_summary.append({"source": blob.get("source"), "preview": s[:300]})
                else:
                    blob_summary.append({"source": blob.get("source"), "keys": blob.get("keys", [])})
            result["data_blobs"] = blob_summary
        # Visible text — first 600 chars
        text = response_body.get("text")
        if text:
            result["page_text"] = text[:600]
        return result

    # --- nested tree (e.g. category tree) ---
    if isinstance(response_body, dict) and "children_data" in response_body:
        flat = _flatten_tree(response_body)
        sample = flat[:max_items]
        note = (
            f"Flattened tree — {len(flat)} total entries. "
            f"Showing first {len(sample)}. "
            "Use search_episode_data with a more specific name/id query to find a particular entry."
        )
        return {"entries_sample": sample, "total": len(flat), "note": note}

    # --- top-level list ---
    if isinstance(response_body, list):
        total = len(response_body)
        sample = [_pick_key_fields(i) for i in response_body[:max_items]]
        note = (
            f"{total} item(s) total. Showing first {len(sample)}. "
            "Refine your search_episode_data query to find a specific item."
        ) if total > max_items else f"{total} item(s)."
        return {"items_sample": sample, "total": total, "note": note}

    # --- dict with an "items" array (common paginated response) ---
    if isinstance(response_body, dict) and "items" in response_body:
        items = response_body.get("items", [])
        total = response_body.get("total_count", len(items))
        sample = [_pick_key_fields(i) for i in items[:max_items]]
        note = (
            f"{total} item(s) total. Showing first {len(sample)}. "
            "Refine your search_episode_data query to find a specific item."
        ) if len(items) > max_items else f"{len(items)} item(s)."
        result = dict(response_body)
        result["items"] = sample
        result["_preview_note"] = note
        result["total_count"] = total
        return result

    # --- plain dict — return as-is (usually already small) ---
    return response_body


def _pick_key_fields(item: Any) -> Any:
    """For list items, keep only the most useful fields to reduce context size."""
    if not isinstance(item, dict):
        return item
    KEEP = {"id", "sku", "name", "price", "category_id", "title", "slug",
            "item_id", "quote_id", "qty", "status", "order_id", "email",
            "username", "token", "cartId", "cart_id"}
    kept = {k: v for k, v in item.items() if k in KEEP}
    return kept if kept else item  # fallback: return full item if no key fields match


# ---------------------------------------------------------------------------
# Text representation for embedding / BM25
# ---------------------------------------------------------------------------

def _body_to_search_text(url: str, method: str, status_code: int,
                          response_body: Any) -> str:
    """
    Produce a searchable text string that represents a stored API response.
    We embed this text so the model can find responses by semantic query.
    The full body is stored separately (in episode_raw_bodies) for retrieval.
    """
    try:
        body_str = json.dumps(response_body) if not isinstance(response_body, str) else response_body
    except Exception:
        body_str = str(response_body)

    # Truncate for embedding (model has 512-token limit; 2000 chars is ~400 tokens)
    if len(body_str) > 2000:
        body_str = body_str[:2000]

    return f"url: {url} method: {method} status: {status_code} response: {body_str}"


# ---------------------------------------------------------------------------
# Semantic embedding search
# ---------------------------------------------------------------------------

def _get_episode_embeddings(episode_store: dict) -> tuple[Any, list[str]] | None:
    """
    Build or retrieve embeddings for all stored episode responses.

    Returns (embeddings_array, text_list) or None if embeddings unavailable.
    Embeddings are cached in episode_store["response_embeddings"] after first build.
    New responses added since last build are embedded incrementally.
    """
    try:
        import numpy as np
        from .browser_agent import _embed_with_cache
    except ImportError:
        return None

    texts: list[str] = episode_store.get("bm25_corpus", [])
    if not texts:
        return None

    cached_embs = episode_store.get("response_embeddings")
    cached_count = len(cached_embs) if cached_embs is not None else 0

    if cached_count == len(texts):
        # All texts already embedded
        return cached_embs, texts

    # Embed any new texts added since last call
    new_texts = texts[cached_count:]
    new_embs = _embed_with_cache(new_texts)
    if new_embs is None:
        return None

    if cached_embs is not None and len(cached_embs) > 0:
        combined = np.vstack([cached_embs, new_embs])
    else:
        combined = new_embs

    episode_store["response_embeddings"] = combined
    return combined, texts


def _semantic_search(query: str, episode_store: dict,
                     top_k: int = 5) -> list[int] | None:
    """
    Return top_k indices ranked by cosine similarity to the query.
    Returns None if embeddings are unavailable (fall back to BM25).
    """
    try:
        import numpy as np
        from .browser_agent import _embed_with_cache
    except ImportError:
        return None

    result = _get_episode_embeddings(episode_store)
    if result is None:
        return None

    embs, _ = result
    query_emb = _embed_with_cache([query])
    if query_emb is None:
        return None

    scores = embs @ query_emb[0]  # dot product = cosine sim (both L2-normalised)
    top_k = min(top_k, len(scores))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# BM25 fallback
# ---------------------------------------------------------------------------

def _bm25_search(query: str, corpus: list[str], top_k: int = 5) -> list[int]:
    """Return top_k indices by BM25 score, or keyword-match fallback."""
    try:
        from rank_bm25 import BM25Okapi
        import numpy as np

        tokenized = [_tokenize(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(_tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in top[:top_k] if scores[i] > 0]
    except Exception:
        pass

    # Keyword fallback
    q_lower = query.lower()
    terms = q_lower.split()
    hits = [i for i, doc in enumerate(corpus) if any(t in doc.lower() for t in terms)]
    return hits[:top_k]


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9_\-\.]+", text)
    return tokens if tokens else [""]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_episode_data(query: str, episode_store: dict) -> list[dict]:
    """
    Semantic + BM25 search over all API responses collected during this episode.

    Each response is stored in full (untruncated) in the episode store.
    Results are returned as compact previews so they fit the LLM context window:
    - Nested trees are flattened to {id, name} pairs with a total-count note.
    - Large arrays show a short sample with a note like "47 items total".
    - Use more specific queries to drill into a particular response.

    Args:
        query: Natural language or keyword query (e.g. "category id for Pants",
               "cart id", "SKU for Radiant Tee", "_csrf_token").
        episode_store: Per-episode mutable store populated by curl_exec.

    Returns:
        List of up to 5 matching results, each with:
          step, url, method, status_code, data (compact preview).
    """
    corpus: list[str] = episode_store.get("bm25_corpus", [])
    metadata: list[dict] = episode_store.get("bm25_metadata", [])

    if not corpus:
        return [{"note": "No episode data yet. Make API calls with curl_exec() first."}]

    # Try semantic search first
    indices = _semantic_search(query, episode_store, top_k=5)

    # Fall back to BM25 if semantic unavailable
    if indices is None:
        indices = _bm25_search(query, corpus, top_k=5)

    if not indices:
        return [{"note": f"No results found for: {query!r}. "
                         "Try a different query or check your curl_exec call history."}]

    results = []
    for idx in indices:
        if idx >= len(metadata):
            continue
        meta = metadata[idx]
        # Full untruncated body is in episode_raw_bodies; metadata holds it too
        raw_body = episode_store.get("episode_raw_bodies", {}).get(idx, meta.get("response_body"))
        results.append({
            "step": idx + 1,
            "url": meta.get("url", ""),
            "method": meta.get("method", ""),
            "status_code": meta.get("status_code", 0),
            "data": _compact_preview(raw_body),
        })

    return results
