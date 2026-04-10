"""
browser_agent tool — HAR-based API surface discovery.

At step 1, loads a pre-recorded HAR file for the target application,
extracts an OpenAPI-like spec from the observed network traffic (XHR/fetch
calls, REST endpoints, form submissions), and builds embeddings via the
HuggingFace Inference API for semantic search_endpoints().

Architecture:
  - The HAR file is the sole source of the agent's API knowledge.
    The agent discovers endpoints only from what was recorded in the HAR.
    If the HAR is sparse, the browser agent recording needs to be improved —
    the product does not patch this by injecting other data sources.

  - The API catalog (catalogs/*.json) is used exclusively by the judge
    for parameter-sourcing grading.  It plays no role in the training loop.

  - Embeddings are cached on disk via embed_cache.py (max 2000 entries).
    First run: calls HF Inference API.  All subsequent episodes in the same
    training run are pure cache hits — zero API cost.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

HARS_DIR = Path(__file__).parent.parent.parent / "hars"

HAR_MAP: dict[str, str] = {
    ":7770": "shopping.har",
    ":7780": "shopping_admin.har",
    ":9999": "forum.har",
    ":3000": "osm.har",
    ":8888": "wikipedia.har",
}

APP_NAME_MAP: dict[str, str] = {
    ":7770": "shopping",
    ":7780": "shopping_admin",
    ":9999": "forum",
    ":3000": "osm",
    ":8888": "wikipedia",
}

# ---------------------------------------------------------------------------
# HAR filtering helpers
# ---------------------------------------------------------------------------

# Hard static asset extensions — always skip
_STATIC_RE = re.compile(
    r"\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot|map|webp|avif|otf|gz|zip)(\?|$)",
    re.IGNORECASE,
)

# Third-party analytics / CDN domains — always skip
_SKIP_HOSTS = {
    "google-analytics.com", "doubleclick.net", "googletagmanager.com",
    "cdn.jsdelivr.net", "cdnjs.cloudflare.com", "fonts.googleapis.com",
    "fonts.gstatic.com",
}

# Path prefixes that are definitely NOT API endpoints
_SKIP_PATH_PREFIXES = (
    "/static/", "/_next/", "/assets/", "/__webpack",
    "/media/catalog/", "/media/logo/", "/media/wysiwyg/",
)

# Path keywords that indicate a real API or data endpoint
_API_PATH_HINTS = (
    "/rest/", "/api/", "/ajax/", "/graphql", ".json",
    "/index.php/customer/", "/index.php/checkout/",
    "/customer/", "/checkout/", "/catalog/", "/mui/",
    "/login_check", "/f/", "/submission/", "/search",
)

# ID normalisation (replace dynamic segments with {id})
_ID_PATTERNS = [
    (re.compile(r"/[0-9a-f]{32,}(?=/|$)"), "/{id}"),
    (re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=/|$)"), "/{id}"),
    (re.compile(r"/\d+(?=/|$)"), "/{id}"),
]

# Max chars for request/response body samples in the embedded text
_BODY_SAMPLE_CHARS = 300


def _is_static_asset(url: str) -> bool:
    parsed = urlparse(url)
    if _STATIC_RE.search(parsed.path):
        return True
    host = parsed.netloc.split(":")[0]
    if any(skip in host for skip in _SKIP_HOSTS):
        return True
    if any(parsed.path.startswith(pfx) for pfx in _SKIP_PATH_PREFIXES):
        return True
    return False


def _is_api_like(path: str, method: str, resp_ct: str, req_ct: str) -> bool:
    """
    Decide whether an entry is worth including in the agent's search index.

    Rules (any match → include):
      1. Non-GET write operation (POST/PUT/PATCH/DELETE) — always interesting
      2. Path contains an API hint keyword
      3. Response content-type is JSON or XML
      4. Request content-type is JSON (even if GET — REST-ish)
    """
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        return True
    if any(hint in path for hint in _API_PATH_HINTS):
        return True
    if "json" in resp_ct or "xml" in resp_ct:
        return True
    if "json" in req_ct:
        return True
    return False


def _is_html_page(method: str, resp_ct: str) -> bool:
    """Return True for HTML GET responses that may contain SSR data."""
    return method == "GET" and "text/html" in resp_ct


def _normalise_path(path: str) -> str:
    for pattern, replacement in _ID_PATTERNS:
        path = pattern.sub(replacement, path)
    return path


def _get_content_type(entry: dict, which: str) -> str:
    obj = entry.get("request" if which == "request" else "response", {})
    for h in obj.get("headers", []):
        if h.get("name", "").lower() == "content-type":
            return h.get("value", "").lower()
    if which == "response":
        return obj.get("content", {}).get("mimeType", "").lower()
    return ""


def _extract_body(req: dict) -> Any:
    post_data = req.get("postData", {})
    if not post_data:
        return None
    text = post_data.get("text", "")
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text[:200] if text else None


def _truncate_response_sample(resp: dict) -> Any:
    content = resp.get("content", {})
    text = content.get("text", "")
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 2:
            return parsed[:2]
        if isinstance(parsed, dict):
            truncated = {}
            for k, v in parsed.items():
                if isinstance(v, list) and len(v) > 2:
                    truncated[k] = v[:2]
                else:
                    truncated[k] = v
            return truncated
        return parsed
    except Exception:
        return text[:_BODY_SAMPLE_CHARS] if text else None


def extract_openapi_spec(har_data: dict, app_base_url: str) -> list[dict]:
    """
    Extract an OpenAPI-like spec from HAR data.

    Includes:
      - REST calls, XHR/fetch, form POSTs, any JSON-responding GET
      - HTML GET pages that have a non-empty response body (distilled via html_distiller)

    Excludes: static assets (JS/CSS/images/fonts), analytics, CDN.
    """
    from .html_distiller import distill_html

    entries = har_data.get("log", {}).get("entries", [])
    seen: set[str] = set()
    spec_entries = []

    for entry in entries:
        req = entry.get("request", {})
        resp = entry.get("response", {})
        raw_url = req.get("url", "")
        method = req.get("method", "GET").upper()

        if not raw_url:
            continue
        if _is_static_asset(raw_url):
            continue

        resp_ct = _get_content_type(entry, "response")
        req_ct = _get_content_type(entry, "request")

        parsed_url = urlparse(raw_url)
        path = parsed_url.path

        is_html = _is_html_page(method, resp_ct)
        is_api = _is_api_like(path, method, resp_ct, req_ct)

        if not is_api and not is_html:
            continue

        path_norm = _normalise_path(path)
        key = f"{method} {path_norm}"
        if key in seen:
            continue
        seen.add(key)

        has_auth = any(
            h.get("name", "").lower() in ("authorization", "x-api-key", "cookie")
            for h in req.get("headers", [])
        )

        if is_html:
            # Attempt to distil the HTML body captured in the HAR
            html_body = entry.get("response", {}).get("content", {}).get("text", "") or ""
            if not html_body:
                # HAR was recorded without "Save response body" — still include the
                # page as a stub so the agent knows the route exists
                distilled = None
                distilled_summary = None
            else:
                distilled = distill_html(html_body, base_url=raw_url)
                # Build a short summary for the spec text (used for embeddings)
                blob_count = len(distilled.get("data_blobs", []))
                form_count = len(distilled.get("forms", []))
                blob_keys = []
                for b in distilled.get("data_blobs", [])[:3]:
                    blob_keys.extend(b.get("keys", [])[:5])
                distilled_summary = {
                    "page_type": distilled.get("page_type"),
                    "title": distilled.get("title"),
                    "data_blobs": blob_count,
                    "forms": form_count,
                    "blob_top_keys": blob_keys[:20],
                    "text_preview": (distilled.get("text") or "")[:200],
                }

            spec_entries.append({
                "method": method,
                "path": path_norm,
                "query_params": parsed_url.query or None,
                "request_body": None,
                "status_code": resp.get("status", 0),
                "response_content_type": resp_ct,
                "response_body_sample": distilled_summary,
                "auth_observed": has_auth,
                "is_html_page": True,
                # Store full distilled dict so the agent can retrieve it via search_endpoints
                "_distilled": distilled,
            })
        else:
            spec_entries.append({
                "method": method,
                "path": path_norm,
                "query_params": parsed_url.query or None,
                "request_body": _extract_body(req),
                "status_code": resp.get("status", 0),
                "response_content_type": resp_ct,
                "response_body_sample": _truncate_response_sample(resp),
                "auth_observed": has_auth,
                "is_html_page": False,
            })

    return spec_entries


def spec_entry_to_text(entry: dict, app_name: str) -> str:
    """Convert a spec entry to a searchable text string for embedding."""
    parts = [
        f"app: {app_name}",
        f"endpoint: {entry['method']} {entry['path']}",
        f"status: {entry['status_code']}",
        f"auth: {'required' if entry['auth_observed'] else 'none'}",
    ]
    if entry.get("is_html_page"):
        parts.append("type: html_page")
        sample = entry.get("response_body_sample") or {}
        if sample.get("title"):
            parts.append(f"title: {sample['title']}")
        if sample.get("blob_top_keys"):
            parts.append(f"data_keys: {' '.join(sample['blob_top_keys'][:15])}")
        if sample.get("text_preview"):
            parts.append(f"text: {sample['text_preview'][:200]}")
    else:
        if entry.get("query_params"):
            parts.append(f"query: {entry['query_params']}")
        if entry.get("request_body"):
            body = entry["request_body"]
            body_str = json.dumps(body)[:_BODY_SAMPLE_CHARS] if not isinstance(body, str) else body[:_BODY_SAMPLE_CHARS]
            parts.append(f"body: {body_str}")
        if entry.get("response_body_sample") is not None:
            rsp = entry["response_body_sample"]
            rsp_str = json.dumps(rsp)[:_BODY_SAMPLE_CHARS] if not isinstance(rsp, str) else str(rsp)[:_BODY_SAMPLE_CHARS]
            parts.append(f"response_sample: {rsp_str}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# HuggingFace Inference API — with persistent cache
# ---------------------------------------------------------------------------

_HF_FEATURE_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "google/embeddinggemma-300m/pipeline/feature-extraction"
)
_EMBED_BATCH_SIZE = 64  # max sentences per API call to avoid HTTP 413


def _call_hf_api(sentences: list[str]) -> np.ndarray | None:
    """
    Raw HF Inference API call for a list of sentences.
    Returns L2-normalized float32 array of shape (N, D), or None on failure.
    """
    import requests as req_lib

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    all_vecs: list[list[float]] = []
    for i in range(0, len(sentences), _EMBED_BATCH_SIZE):
        batch = sentences[i : i + _EMBED_BATCH_SIZE]
        try:
            resp = req_lib.post(
                _HF_FEATURE_URL,
                headers=headers,
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data:
                # feature-extraction can return per-token or sentence-level vectors
                if item and isinstance(item[0], list):
                    # Per-token: mean-pool over tokens
                    vec = np.mean(np.array(item, dtype=np.float32), axis=0)
                    all_vecs.append(vec.tolist())
                else:
                    all_vecs.append(item)
        except Exception as e:
            print(f"[browser_agent] HF API batch {i}:{i+_EMBED_BATCH_SIZE} failed: {e}", flush=True)
            return None

    if not all_vecs:
        return None

    emb = np.array(all_vecs, dtype=np.float32)
    # L2-normalize so dot product == cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return emb / norms


def _embed_with_cache(sentences: list[str]) -> np.ndarray | None:
    """
    Embed a list of sentences, using the persistent cache as a proxy.

    Flow:
      1. Check cache for all sentences (O(n) dict lookup)
      2. For cache misses, call HF API in batches
      3. Store new embeddings in cache (single disk write)
      4. Return the full (N, D) array — all L2-normalized

    On the second+ call for any given sentence, cost = 0 API calls.
    """
    from .embed_cache import get_cache

    cache = get_cache()
    results, miss_indices = cache.get_batch(sentences)

    n_hits = len(sentences) - len(miss_indices)
    if miss_indices:
        print(
            f"[browser_agent] Embed cache: {n_hits} hits, {len(miss_indices)} misses — "
            f"calling HF API for {len(miss_indices)} sentences",
            flush=True,
        )
        missing_texts = [sentences[i] for i in miss_indices]
        new_embs = _call_hf_api(missing_texts)
        if new_embs is None:
            if n_hits == 0:
                return None  # total failure
            # Partial failure: fill misses with zeros (keyword fallback will handle them)
            for i in miss_indices:
                results[i] = np.zeros(768, dtype=np.float32)
        else:
            # Store new embeddings in cache (single disk write)
            cache.put_batch(
                [(missing_texts[j], new_embs[j]) for j in range(len(missing_texts))]
            )
            for j, i in enumerate(miss_indices):
                results[i] = new_embs[j]
    else:
        print(
            f"[browser_agent] Embed cache: {n_hits} hits, 0 misses — no API call needed",
            flush=True,
        )

    return np.array(results, dtype=np.float32)  # shape (N, D), all L2-normalized


def embed_query_via_api(query: str) -> np.ndarray | None:
    """
    Embed a single query string.  Returns shape (1, D) or None.
    Also goes through the cache — repeated queries cost nothing.
    """
    return _embed_with_cache([query])


def build_endpoint_embeddings(spec_entries: list[dict], app_name: str):
    """
    Build embeddings for HAR-extracted spec entries.
    Returns (embeddings_array, text_chunks).
    Embeddings are retrieved from or saved to the persistent cache.
    """
    chunks = [spec_entry_to_text(e, app_name) for e in spec_entries]
    if not chunks:
        return np.array([]), []

    print(f"[browser_agent] Building embeddings for {len(chunks)} endpoints...", flush=True)
    embeddings = _embed_with_cache(chunks)
    if embeddings is None:
        print("[browser_agent] Embedding unavailable — keyword search only.", flush=True)
        return None, chunks

    print(f"[browser_agent] Embeddings ready: shape {embeddings.shape}", flush=True)
    return embeddings, chunks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_browser_agent(task: str, url: str, episode_store=None) -> dict:
    """
    Load the pre-recorded HAR for the app inferred from the URL,
    extract an OpenAPI-like spec from the observed traffic,
    build/retrieve embeddings (via cache → HF API),
    and store everything in episode_store for search_endpoints().

    Returns a summary of discovered endpoints (method + path only).
    The agent must call search_endpoints() to get full schemas.
    """
    # Detect app from URL port
    app_name = "unknown"
    har_filename = None
    for port_suffix, fname in HAR_MAP.items():
        if port_suffix in url:
            har_filename = fname
            app_name = APP_NAME_MAP[port_suffix]
            break

    # Fallback app detection
    if har_filename is None:
        if "7780" in url or "shopping_admin" in url.lower():
            har_filename, app_name = "shopping_admin.har", "shopping_admin"
        elif "7770" in url or "shopping" in url.lower():
            har_filename, app_name = "shopping.har", "shopping"
        elif "9999" in url or "forum" in url.lower():
            har_filename, app_name = "forum.har", "forum"
        elif "8888" in url or "wiki" in url.lower():
            har_filename, app_name = "wikipedia.har", "wikipedia"
        elif "3000" in url or "osm" in url.lower():
            har_filename, app_name = "osm.har", "osm"
        else:
            har_filename, app_name = "shopping.har", "shopping"

    har_path = HARS_DIR / har_filename
    if not har_path.exists():
        _store_empty(episode_store, app_name)
        return {
            "app": app_name,
            "endpoints": [],
            "total_endpoints": 0,
            "error": f"HAR file not found: {har_filename}",
            "note": "No HAR available. Use search_endpoints() with keyword queries.",
        }

    with open(har_path) as f:
        har_data = json.load(f)

    # Extract spec from HAR traffic
    spec_entries = extract_openapi_spec(har_data, url)
    print(
        f"[browser_agent] HAR '{har_filename}' → {len(spec_entries)} unique endpoints extracted",
        flush=True,
    )

    # Build / retrieve embeddings via cache
    if spec_entries and episode_store is not None:
        try:
            embeddings, chunks = build_endpoint_embeddings(spec_entries, app_name)
            episode_store["endpoint_embeddings"] = embeddings
            episode_store["endpoint_chunks"] = chunks
            episode_store["spec_entries"] = spec_entries
            episode_store["app_name"] = app_name
        except Exception as e:
            print(f"[browser_agent] Embedding error: {e} — using keyword fallback.", flush=True)
            chunks = [spec_entry_to_text(e, app_name) for e in spec_entries]
            episode_store["endpoint_embeddings"] = None
            episode_store["endpoint_chunks"] = chunks
            episode_store["spec_entries"] = spec_entries
            episode_store["app_name"] = app_name
    elif episode_store is not None:
        _store_empty(episode_store, app_name)

    summary = [{"method": e["method"], "path": e["path"]} for e in spec_entries]
    api_count = sum(1 for e in spec_entries if not e.get("is_html_page"))
    html_count = sum(1 for e in spec_entries if e.get("is_html_page"))
    return {
        "app": app_name,
        "endpoints": summary,
        "total_endpoints": len(summary),
        "api_endpoints": api_count,
        "html_pages": html_count,
        "note": (
            f"Discovered {api_count} API endpoints and {html_count} HTML page(s) "
            f"from recorded traffic. "
            "Use search_endpoints(query) to get full schema, parameters, auth details, "
            "and page content (for HTML pages: embedded data blobs, forms, CSRF tokens)."
        ),
    }


def _store_empty(episode_store, app_name: str) -> None:
    if episode_store is not None:
        episode_store["spec_entries"] = []
        episode_store["endpoint_chunks"] = []
        episode_store["endpoint_embeddings"] = None
        episode_store["app_name"] = app_name
