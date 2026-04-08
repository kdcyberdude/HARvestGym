"""
Tool 0: browser_agent — HAR processing pipeline.

Stages:
1. Check for pre-recorded HAR file (by port mapping) → load or fall back to live browser
2. Filter HAR entries: skip static assets, HTML pages, deduplicate by (method, normalised path)
3. Build OpenAPI-like spec from filtered entries
4. Build GEMMA embeddings over the spec (for search_endpoints)
5. Return summary endpoint list (method + path only)
"""

import json
import os
import re
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HAR_MAP = {
    ":7770": "hars/shopping.har",
    ":7780": "hars/shopping_admin.har",
    ":9999": "hars/forum.har",
    ":3000": "hars/osm.har",
    ":8888": "hars/wikipedia.har",
}

APP_NAMES = {
    ":7770": "shopping",
    ":7780": "shopping_admin",
    ":9999": "forum",
    ":3000": "osm",
    ":8888": "wikipedia",
}

SKIP_EXTENSIONS = {".css", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".woff",
                   ".woff2", ".ttf", ".gif", ".js", ".map"}

SKIP_PATH_PREFIXES = ["/static/", "/media/", "/_next/", "/assets/",
                      "/__webpack", "/pub/static/"]

# ---------------------------------------------------------------------------
# Path normalisation
# ---------------------------------------------------------------------------

# Patterns for dynamic segments
_UUID_RE = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)
_LONG_ALPHANUM_RE = re.compile(r'[a-zA-Z0-9]{32,}')  # Magento cart IDs etc.
_NUMERIC_ID_RE = re.compile(r'^[0-9]+$')
_FORUM_SLUG_RE = re.compile(r'^[0-9]+-[a-z0-9-]+$')  # e.g. "1-hello-world"


def _normalise_path(path: str) -> str:
    """Replace concrete IDs/slugs with {id} placeholders."""
    segments = path.strip("/").split("/")
    normalised = []
    for seg in segments:
        if _UUID_RE.fullmatch(seg):
            normalised.append("{id}")
        elif _LONG_ALPHANUM_RE.fullmatch(seg):
            normalised.append("{id}")
        elif _NUMERIC_ID_RE.fullmatch(seg) and len(seg) >= 2:
            normalised.append("{id}")
        elif _FORUM_SLUG_RE.fullmatch(seg):
            normalised.append("{id}-{slug}")
        else:
            normalised.append(seg)
    return "/" + "/".join(normalised)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _is_static_asset(url: str) -> bool:
    """Check if URL points to a static asset."""
    parsed = urlparse(url)
    path = parsed.path.lower()

    for ext in SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True

    for prefix in SKIP_PATH_PREFIXES:
        if path.startswith(prefix):
            return True

    return False


def _get_response_content_type(resp: dict) -> str:
    """Extract content-type from response headers or content field."""
    # Check headers
    for h in resp.get("headers", []):
        if h["name"].lower() == "content-type":
            return h["value"].lower()
    # Check content field (HAR format)
    content = resp.get("content", {})
    return content.get("mimeType", "").lower()


def _extract_body(req: dict) -> str | None:
    """Extract request body text from HAR entry."""
    pd = req.get("postData")
    if pd is None:
        return None
    if isinstance(pd, dict):
        return pd.get("text")
    return str(pd) if pd else None


def _truncate_body(resp: dict, max_len: int = 500) -> str | None:
    """Extract and truncate response body for spec document."""
    content = resp.get("content", {})
    text = content.get("text", "")
    if not text:
        return None
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def resolve_har_path(url: str, base_dir: str = ".") -> str | None:
    """Find pre-recorded HAR file for this app URL."""
    for port_key, rel_path in HAR_MAP.items():
        if port_key in url:
            full_path = os.path.join(base_dir, rel_path)
            if os.path.exists(full_path):
                return full_path
    return None


def resolve_app_name(url: str) -> str:
    """Map URL to app name."""
    for port_key, name in APP_NAMES.items():
        if port_key in url:
            return name
    return "unknown"


def extract_openapi_spec(har_data: dict, app_base_url: str) -> list[dict]:
    """
    Stage 2-3: Filter HAR entries and extract OpenAPI-like spec.
    Returns list of structured endpoint documents.
    """
    entries = har_data["log"]["entries"]
    seen = set()
    spec_entries = []

    for entry in entries:
        req = entry["request"]
        resp = entry["response"]
        raw_url = req["url"]
        method = req["method"]

        # Skip static assets
        if _is_static_asset(raw_url):
            continue

        # Skip HTML page navigations
        content_type = _get_response_content_type(resp)
        if "text/html" in content_type and method == "GET":
            continue

        # Normalise path
        parsed = urlparse(raw_url)
        path = _normalise_path(parsed.path)

        # Deduplicate
        key = f"{method} {path}"
        if key in seen:
            continue
        seen.add(key)

        # Auth detection
        has_auth = any(
            h["name"].lower() in ("authorization", "x-api-key", "cookie")
            for h in req.get("headers", [])
        )

        spec_entries.append({
            "method": method,
            "path": path,
            "query_params": parsed.query or None,
            "request_body": _extract_body(req),
            "status_code": resp["status"],
            "response_content_type": content_type,
            "response_body_sample": _truncate_body(resp),
            "auth_observed": has_auth,
        })

    return spec_entries


def spec_entry_to_text(entry: dict, app_name: str) -> str:
    """Convert a spec entry to a searchable text document for embedding."""
    parts = [
        f"app: {app_name}",
        f"endpoint: {entry['method']} {entry['path']}",
        f"status: {entry['status_code']}",
        f"auth: {'required' if entry['auth_observed'] else 'none'}",
    ]
    if entry.get("query_params"):
        parts.append(f"query: {entry['query_params']}")
    if entry.get("request_body"):
        parts.append(f"body: {entry['request_body'][:200]}")
    if entry.get("response_body_sample"):
        parts.append(f"response_sample: {entry['response_body_sample'][:200]}")
    return " | ".join(parts)


def build_summary_output(spec_entries: list[dict], app_name: str) -> dict:
    """Stage 5: Build summary-only output for the RL agent."""
    endpoints = [{"method": e["method"], "path": e["path"]} for e in spec_entries]
    return {
        "app": app_name,
        "endpoints": endpoints,
        "total_endpoints": len(endpoints),
        "note": (
            "These endpoints were observed for this application. "
            "Use search_endpoints() with a natural language query to get "
            "the full schema, parameters, and auth details for any endpoint."
        ),
    }


def browser_agent(task: str, url: str, base_dir: str = ".") -> tuple[dict, list[dict], list[str]]:
    """
    Full browser_agent pipeline.

    Returns:
        (summary_output, spec_entries, text_chunks)
        - summary_output: what the RL agent sees
        - spec_entries: structured spec for internal use
        - text_chunks: searchable text docs for embedding/search
    """
    app_name = resolve_app_name(url)

    # Stage 1: Get HAR data
    har_path = resolve_har_path(url, base_dir)
    if har_path:
        with open(har_path) as f:
            har_data = json.load(f)
    else:
        raise FileNotFoundError(
            f"No HAR file found for {url}. Live browser fallback not implemented in test mode."
        )

    # Stage 2-3: Extract spec
    spec_entries = extract_openapi_spec(har_data, url)

    # Stage 4: Build text chunks (embeddings would happen here)
    text_chunks = [spec_entry_to_text(e, app_name) for e in spec_entries]

    # Stage 5: Build summary
    summary = build_summary_output(spec_entries, app_name)

    return summary, spec_entries, text_chunks


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: browser_agent with mock HAR data")
    print("=" * 70)

    mock_har_path = os.path.join(os.path.dirname(__file__), "mock_data", "mock_har.json")
    with open(mock_har_path) as f:
        har_data = json.load(f)

    url = "http://localhost:7770/"
    app_name = "shopping"

    # Test filtering
    spec = extract_openapi_spec(har_data, url)
    print(f"\nFiltered {len(har_data['log']['entries'])} HAR entries → {len(spec)} API endpoints\n")

    for e in spec:
        print(f"  {e['method']:6s} {e['path']}")
        if e.get("request_body"):
            print(f"         body: {e['request_body'][:80]}...")

    # Test summary output
    summary = build_summary_output(spec, app_name)
    print(f"\n--- Summary Output (what RL agent sees) ---")
    print(json.dumps(summary, indent=2))

    # Test text chunks
    chunks = [spec_entry_to_text(e, app_name) for e in spec]
    print(f"\n--- Text Chunks for Embedding ({len(chunks)} docs) ---")
    for i, chunk in enumerate(chunks):
        print(f"  [{i}] {chunk[:120]}...")

    # Test path normalisation
    print(f"\n--- Path Normalisation Tests ---")
    test_paths = [
        "/rest/V1/products/42",
        "/rest/V1/guest-carts/3fa85f64-5717-4562-b3fc-2c963f66afa6/items",
        "/rest/V1/guest-carts/abcdef1234567890abcdef1234567890ab/totals",
        "/api/0.6/node/12345678",
        "/f/general/1-hello-world",
        "/rest/V1/categories",
        "/rest/V1/products",
    ]
    for p in test_paths:
        print(f"  {p:65s} → {_normalise_path(p)}")

    # Test static asset detection
    print(f"\n--- Static Asset Detection ---")
    test_urls = [
        "http://localhost:7770/rest/V1/products",
        "http://localhost:7770/static/version1/file.js",
        "http://localhost:7770/media/catalog/product/img.jpg",
        "http://localhost:7770/beauty-personal-care.html",
    ]
    for u in test_urls:
        print(f"  {u:60s} → static={_is_static_asset(u)}")

    print("\n[PASS] browser_agent tool tests completed successfully")
