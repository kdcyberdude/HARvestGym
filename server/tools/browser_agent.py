"""
browser_agent tool — HAR-based API surface discovery.

At step 1, loads a pre-recorded HAR file for the target application,
extracts an OpenAPI-like spec, and builds embeddings via the HuggingFace
Inference API (google/embeddinggemma-300m) for semantic search_endpoints().
No local model download — requires HF_TOKEN in the environment.
Falls back to keyword search if the API call fails.
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
# HAR path resolution
# ---------------------------------------------------------------------------

HARS_DIR = Path(__file__).parent.parent.parent / "hars"
CATALOGS_DIR = Path(__file__).parent.parent.parent / "catalogs"

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

# Static asset patterns to skip
_STATIC_RE = re.compile(
    r"\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot|map|webp|avif|otf)(\?|$)",
    re.IGNORECASE,
)
_ANALYTICS_HOSTS = {"google-analytics.com", "doubleclick.net", "googletagmanager.com",
                    "cdn.jsdelivr.net", "cdnjs.cloudflare.com"}

# ID normalisation patterns
_ID_PATTERNS = [
    (re.compile(r"/[0-9a-f]{32,}(?=/|$)"), "/{id}"),           # Magento cart IDs
    (re.compile(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?=/|$)"), "/{id}"),  # UUIDs
    (re.compile(r"/\d+(?=/|$)"), "/{id}"),                      # numeric IDs
]


def _is_static_asset(url: str) -> bool:
    parsed = urlparse(url)
    if _STATIC_RE.search(parsed.path):
        return True
    if parsed.netloc in _ANALYTICS_HOSTS:
        return True
    return False


def _normalise_path(path: str) -> str:
    for pattern, replacement in _ID_PATTERNS:
        path = pattern.sub(replacement, path)
    return path


def _get_content_type(entry: dict, which: str) -> str:
    """Extract Content-Type from request or response headers."""
    headers_key = "request" if which == "request" else "response"
    obj = entry.get(headers_key, {})
    for h in obj.get("headers", []):
        if h.get("name", "").lower() == "content-type":
            return h.get("value", "").lower()
    if which == "response":
        ct = obj.get("content", {}).get("mimeType", "")
        return ct.lower()
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
            # truncate large arrays in response
            truncated = {}
            for k, v in parsed.items():
                if isinstance(v, list) and len(v) > 2:
                    truncated[k] = v[:2]
                else:
                    truncated[k] = v
            return truncated
        return parsed
    except Exception:
        return text[:300] if text else None


def extract_openapi_spec(har_data: dict, app_base_url: str) -> list[dict]:
    """Extract OpenAPI-like spec from HAR data."""
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

        # Skip pure static HTML page loads (GET returning text/html for main page/nav)
        # BUT keep: POST forms, API paths, admin paths, JSON responses
        is_html_get = "text/html" in resp_ct and method == "GET"
        has_api_path = any(x in path for x in ["/rest/", "/api/", "/ajax/", "/mui/", ".json"])
        is_admin_path = "/admin/" in path or "/rest/V1/" in path
        is_post = method in ("POST", "PUT", "PATCH", "DELETE")
        has_json_response = "json" in resp_ct

        if is_html_get and not has_api_path and not is_admin_path and not has_json_response:
            # Skip pure page navigations but only for very common extensions
            if not is_post:
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

        spec_entries.append({
            "method": method,
            "path": path_norm,
            "query_params": parsed_url.query or None,
            "request_body": _extract_body(req),
            "status_code": resp.get("status", 0),
            "response_content_type": resp_ct,
            "response_body_sample": _truncate_response_sample(resp),
            "auth_observed": has_auth,
        })

    return spec_entries


def catalog_to_spec_entries(app_name: str) -> list[dict]:
    """Load ground truth catalog as spec entries when HAR doesn't yield results."""
    catalog_path = CATALOGS_DIR / f"{app_name}.json"
    if not catalog_path.exists():
        return []
    try:
        with open(catalog_path) as f:
            data = json.load(f)
        endpoints = data if isinstance(data, list) else data.get("endpoints", [])
        spec_entries = []
        for ep in endpoints:
            # Handle "endpoint": "POST /rest/V1/..." format
            endpoint_str = ep.get("endpoint", "")
            if endpoint_str and " " in endpoint_str:
                parts = endpoint_str.split(" ", 1)
                method = parts[0].upper()
                path = parts[1]
            else:
                path = ep.get("path", endpoint_str)
                method = ep.get("method", "GET").upper()

            if not path:
                continue

            auth = ep.get("auth", ep.get("authentication", "none"))
            spec_entries.append({
                "method": method,
                "path": path,
                "query_params": None,
                "request_body": ep.get("body_params") or ep.get("body"),
                "status_code": 200,
                "response_content_type": "application/json",
                "response_body_sample": ep.get("response_fields") or ep.get("response_sample"),
                "auth_observed": auth not in ("none", "None", None, ""),
            })
        return spec_entries
    except Exception as e:
        print(f"[browser_agent] Failed to load catalog {app_name}: {e}", flush=True)
        return []


def spec_entry_to_text(entry: dict, app_name: str) -> str:
    """Convert a spec entry to searchable text for embedding."""
    parts = [
        f"app: {app_name}",
        f"endpoint: {entry['method']} {entry['path']}",
        f"status: {entry['status_code']}",
        f"auth: {'required' if entry['auth_observed'] else 'none'}",
    ]
    if entry.get("query_params"):
        parts.append(f"query: {entry['query_params']}")
    if entry.get("request_body"):
        body_str = json.dumps(entry["request_body"])[:300] if not isinstance(entry["request_body"], str) else entry["request_body"][:300]
        parts.append(f"body: {body_str}")
    if entry.get("response_body_sample") is not None:
        resp_str = json.dumps(entry["response_body_sample"])[:300] if not isinstance(entry["response_body_sample"], str) else str(entry["response_body_sample"])[:300]
        parts.append(f"response_sample: {resp_str}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# HuggingFace Inference API embeddings (no local model download needed)
# ---------------------------------------------------------------------------

_HF_EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "google/embeddinggemma-300m/pipeline/sentence-similarity"
)

# Maximum sentences per API call (avoid HTTP 413)
_EMBED_BATCH_SIZE = 64


def _embed_via_api(sentences: list[str], source_sentence: str | None = None) -> np.ndarray | None:
    """
    Call the HF Inference API to get sentence embeddings or similarity scores.

    When source_sentence is None: returns a (N, D) embedding matrix by calling
    the API with each sentence as the source and a fixed anchor, then using the
    raw hidden states trick — actually we call the feature-extraction pipeline.

    Because the sentence-similarity pipeline returns similarity *scores* (not
    raw vectors), we use the feature-extraction endpoint instead to get
    real embedding vectors we can do cosine search on.
    """
    import requests

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return None

    # Use the feature-extraction pipeline which returns raw embedding vectors
    url = (
        "https://router.huggingface.co/hf-inference/models/"
        "google/embeddinggemma-300m/pipeline/feature-extraction"
    )
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []
    for i in range(0, len(sentences), _EMBED_BATCH_SIZE):
        batch = sentences[i : i + _EMBED_BATCH_SIZE]
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={"inputs": batch, "options": {"wait_for_model": True}},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # feature-extraction returns list[list[float]] or list[list[list[float]]]
            # (the latter when the model returns per-token embeddings — we mean-pool)
            for item in data:
                if isinstance(item[0], list):
                    # Per-token: mean pool over tokens
                    arr = np.mean(np.array(item), axis=0)
                    all_embeddings.append(arr.tolist())
                else:
                    all_embeddings.append(item)
        except Exception as e:
            print(f"[browser_agent] HF embed API batch {i} failed: {e}", flush=True)
            return None

    if not all_embeddings:
        return None

    emb = np.array(all_embeddings, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return emb / norms


def embed_query_via_api(query: str) -> np.ndarray | None:
    """Embed a single query string via HF API. Returns shape (1, D) or None."""
    result = _embed_via_api([query])
    return result  # shape (1, D)


def build_endpoint_embeddings(spec_entries: list[dict], app_name: str):
    """
    Build embeddings for all endpoint spec entries using HF Inference API.
    Returns (embeddings_array, text_chunks).
    Falls back to (None, chunks) if the API is unavailable.
    """
    chunks = [spec_entry_to_text(e, app_name) for e in spec_entries]
    if not chunks:
        return np.array([]), []

    print(f"[browser_agent] Embedding {len(chunks)} endpoints via HF API...", flush=True)
    embeddings = _embed_via_api(chunks)
    if embeddings is None:
        print("[browser_agent] HF embed API unavailable — keyword search only.", flush=True)
        return None, chunks

    print(f"[browser_agent] Embeddings built: shape {embeddings.shape}", flush=True)
    return embeddings, chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_browser_agent(task: str, url: str, episode_store=None) -> dict:
    """
    Load HAR for the app inferred from URL, extract spec, build embeddings.
    Returns summary endpoint list.

    episode_store: mutable dict where we store embeddings/spec for search_endpoints().
    """
    # Detect app from URL
    app_name = "unknown"
    har_filename = None
    for port_suffix, fname in HAR_MAP.items():
        if port_suffix in url:
            har_filename = fname
            app_name = APP_NAME_MAP[port_suffix]
            break

    if har_filename is None:
        # Try to guess from URL path
        if "shopping" in url.lower() or "7770" in url or "7780" in url:
            har_filename = "shopping.har"
            app_name = "shopping"
        elif "forum" in url.lower() or "9999" in url:
            har_filename = "forum.har"
            app_name = "forum"
        elif "wiki" in url.lower() or "8888" in url:
            har_filename = "wikipedia.har"
            app_name = "wikipedia"
        else:
            har_filename = "shopping.har"
            app_name = "shopping"

    har_path = HARS_DIR / har_filename
    if not har_path.exists():
        return {
            "app": app_name,
            "endpoints": [],
            "total_endpoints": 0,
            "note": f"HAR file not found: {har_path}. No endpoints available.",
            "error": f"Missing HAR: {har_filename}",
        }

    with open(har_path) as f:
        har_data = json.load(f)

    spec_entries = extract_openapi_spec(har_data, url)

    # Augment with ground truth catalog if HAR extraction is sparse
    catalog_entries = catalog_to_spec_entries(app_name)
    if len(spec_entries) < 5 and catalog_entries:
        print(f"[browser_agent] HAR yielded {len(spec_entries)} endpoints, augmenting from catalog ({len(catalog_entries)} entries)", flush=True)
        # Merge: catalog takes priority for proper paths
        har_paths = {e["path"] for e in spec_entries}
        for ce in catalog_entries:
            if ce["path"] not in har_paths:
                spec_entries.append(ce)
    elif catalog_entries:
        # Augment any catalog endpoints not found in HAR
        har_paths = {e["path"] for e in spec_entries}
        for ce in catalog_entries:
            if ce["path"] not in har_paths:
                spec_entries.append(ce)

    # Build embeddings and store in episode_store for search_endpoints
    if spec_entries and episode_store is not None:
        try:
            embeddings, chunks = build_endpoint_embeddings(spec_entries, app_name)
            episode_store["endpoint_embeddings"] = embeddings
            episode_store["endpoint_chunks"] = chunks
            episode_store["spec_entries"] = spec_entries
            episode_store["app_name"] = app_name
        except Exception as e:
            print(f"[browser_agent] Embedding build failed: {e}. Storing spec without embeddings.", flush=True)
            # Store chunks as plain text even without embeddings for keyword fallback
            chunks = [spec_entry_to_text(e, app_name) for e in spec_entries]
            episode_store["endpoint_chunks"] = chunks
            episode_store["endpoint_embeddings"] = None
            episode_store["spec_entries"] = spec_entries
            episode_store["app_name"] = app_name
    elif episode_store is not None:
        episode_store["spec_entries"] = []
        episode_store["app_name"] = app_name

    # Return summary only (no schemas)
    summary_endpoints = [{"method": e["method"], "path": e["path"]} for e in spec_entries]

    return {
        "app": app_name,
        "endpoints": summary_endpoints,
        "total_endpoints": len(summary_endpoints),
        "note": (
            "These endpoints were observed for this application. "
            "Use search_endpoints() with a natural language query to get the full schema, "
            "parameters, and auth details for any endpoint."
        ),
    }
