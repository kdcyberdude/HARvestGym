"""
html_distiller — technology-agnostic HTML distillation for the RL agent.

Converts an HTML response body into a compact, structured dict that the agent
and the embedding index can work with.  Raw HTML is never returned as-is —
it is expensive (200 KB+) and mostly noise (CSS classes, JS bundles, nav chrome).

What is extracted (in priority order):
  1. Embedded JSON data blobs  — server-injected structured data that is the
     *actual payload* for SSR pages:
       • <script type="application/json">                (Next.js, generic)
       • <script type="text/x-magento-init">            (Magento 2)
       • window.__INITIAL_STATE__ = {...}               (Redux-style SSR)
       • window.__NEXT_DATA__ = {...}                   (Next.js legacy)
       • window.__nuxt__ = {...} / window.__NUXT__ = {} (Nuxt.js)
       • <script id="__NEXT_DATA__">                    (Next.js)
       • Any <script> tag containing only valid JSON
     These are technology-specific patterns, but the extraction logic is written
     generically — it looks for the common conventions rather than hardcoding
     Magento.  A React/Next.js app will be handled by the same code path.

  2. HTML forms  — discovers new POST endpoints (form.action) and captures
     auth-critical fields (CSRF tokens, hidden inputs).

  3. Visible text content  — the human-readable body after stripping all
     scripts, styles, and nav/header/footer chrome.  Capped at MAX_TEXT_CHARS.

Output schema (always a dict with the same keys — absent items are None/[]):
  {
    "page_type": str,          # "data_page" | "form_page" | "text_page"
    "title": str | None,       # <title> text
    "description": str | None, # <meta name="description">
    "data_blobs": [            # extracted JSON payloads
      {"source": str, "data": any, "keys": [str]}  # keys = top-level keys
    ],
    "forms": [
      {
        "action": str,         # endpoint URL (relative or absolute)
        "method": str,         # GET | POST
        "fields": {            # name → value (includes hidden inputs)
          "field_name": "field_value_or_type"
        }
      }
    ],
    "text": str | None,        # stripped visible text (capped)
    "raw_truncated": str,      # first RAW_PREVIEW_CHARS of raw HTML (fallback)
  }

Usage:
  from server.tools.html_distiller import distill_html

  result = distill_html(html_string, base_url="http://example.com/page")
  # result["data_blobs"] — structured data, e.g. product listings
  # result["forms"]      — form actions + CSRF tokens
  # result["text"]       — stripped readable text
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urljoin

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TEXT_CHARS = 20000      # max chars of stripped visible text to keep
MAX_BLOB_KEYS = 40         # max top-level keys to surface from a JSON blob
MAX_BLOB_DEPTH_PREVIEW = 2 # how many levels of nesting to summarise
RAW_PREVIEW_CHARS = 1000   # fallback raw HTML preview if BS4 unavailable
MAX_BLOBS = 10             # max embedded JSON blobs to extract
MAX_FORMS = 5              # max forms to extract
MAX_ITEMS_IN_ARRAY = 3     # preview items for large arrays in blobs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def distill_html(html: str, base_url: str = "") -> dict:
    """
    Distil an HTML page into a structured, compact representation.

    Args:
        html:      Raw HTML string (may be very large).
        base_url:  The URL this page was fetched from, used to resolve
                   relative URLs in form actions.

    Returns:
        Distilled dict (see module docstring for schema).
    """
    if not html:
        return _empty_result()

    if not _BS4_AVAILABLE:
        return {
            **_empty_result(),
            "raw_truncated": html[:RAW_PREVIEW_CHARS],
            "_note": "beautifulsoup4 not installed; only raw preview returned.",
        }

    try:
        # lxml is faster and more forgiving than html.parser for large pages
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    title = _extract_title(soup)
    description = _extract_meta_description(soup)
    data_blobs = _extract_data_blobs(soup)
    forms = _extract_forms(soup, base_url)
    text = _extract_visible_text(soup)

    # Determine page_type based on what we found
    if data_blobs:
        page_type = "data_page"
    elif forms:
        page_type = "form_page"
    else:
        page_type = "text_page"

    return {
        "page_type": page_type,
        "title": title,
        "description": description,
        "data_blobs": data_blobs,
        "forms": forms,
        "text": text,
        "raw_truncated": html[:RAW_PREVIEW_CHARS],
    }


def distill_html_compact(html: str, base_url: str = "") -> str:
    """
    Return a compact text representation of the distilled HTML,
    suitable for returning to the agent in curl_exec responses.

    Aims for < 3000 chars while preserving all actionable information.
    """
    d = distill_html(html, base_url)

    parts: list[str] = []

    if d["title"]:
        parts.append(f"[Page: {d['title']}]")

    if d["description"]:
        parts.append(f"[Description: {d['description']}]")

    if d["data_blobs"]:
        parts.append(f"[Embedded data — {len(d['data_blobs'])} block(s)]")
        for i, blob in enumerate(d["data_blobs"]):
            src = blob.get("source", "?")
            data = blob.get("data")
            preview = _compact_blob_preview(data)
            parts.append(f"  blob[{i}] from <{src}>: {preview}")

    if d["forms"]:
        parts.append(f"[Forms — {len(d['forms'])} found]")
        for form in d["forms"]:
            action = form["action"] or "(current page)"
            method = form["method"]
            fields = form["fields"]
            # Strip noisy base64-encoded redirect fields; keep actionable fields only
            _SKIP_FIELDS = {"uenc"}
            clean_fields = {k: v for k, v in fields.items() if k not in _SKIP_FIELDS}
            csrf = {k: v for k, v in clean_fields.items()
                    if "csrf" in k.lower() or "token" in k.lower()
                    or k.startswith("_") or clean_fields.get(k, "") == "hidden"}
            field_summary = ", ".join(f"{k}={repr(v)}" for k, v in list(clean_fields.items())[:6])
            parts.append(f"  {method} {action}")
            parts.append(f"    fields: {field_summary}")
            if csrf:
                parts.append(f"    csrf/hidden: {csrf}")

    if d["text"]:
        parts.append(f"[Text content]\n{d['text'][:800]}")

    result = "\n".join(parts)
    if not result:
        # Absolute fallback: raw preview
        return html[:RAW_PREVIEW_CHARS]
    return result


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_title(soup) -> str | None:
    tag = soup.find("title")
    if tag:
        return tag.get_text(strip=True) or None
    return None


def _extract_meta_description(soup) -> str | None:
    tag = soup.find("meta", attrs={"name": "description"})
    if tag and tag.get("content"):
        return tag["content"].strip() or None
    return None


# Patterns for window.X = {...} assignments in inline scripts
_WINDOW_ASSIGN_RE = re.compile(
    r'window\.__?([A-Za-z0-9_]+)__?\s*=\s*(\{.*?\}|\[.*?\])',
    re.DOTALL,
)

# Known SSR data script types
_DATA_SCRIPT_TYPES = {
    "application/json",
    "text/x-magento-init",
    "application/ld+json",     # structured data / schema.org
}

# Known SSR script IDs
_DATA_SCRIPT_IDS = {
    "__next_data__",
    "__nuxt__",
    "initial-state",
    "redux-state",
    "app-state",
    "page-data",
    "server-data",
    "bootstrap-data",
}


def _try_parse_json(text: str) -> tuple[bool, Any]:
    """Returns (success, parsed_value)."""
    text = text.strip()
    if not text:
        return False, None
    try:
        return True, json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False, None


def _summarise_json_keys(obj: Any, depth: int = 0) -> list[str]:
    """Return top-level keys (and one level of nested keys) for a JSON object."""
    if not isinstance(obj, dict):
        if isinstance(obj, list) and obj:
            return _summarise_json_keys(obj[0], depth)
        return []
    keys = list(obj.keys())
    if depth < 1:
        nested = []
        for k in keys[:5]:
            v = obj[k]
            if isinstance(v, dict):
                sub = list(v.keys())[:5]
                nested.append(f"{k}.{{{','.join(sub)}}}")
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                sub = list(v[0].keys())[:4]
                nested.append(f"{k}[].{{{','.join(sub)}}}")
        return keys + nested
    return keys


def _extract_data_blobs(soup) -> list[dict]:
    """
    Extract all embedded JSON data blobs from <script> tags and window.X = {...} patterns.
    """
    blobs: list[dict] = []
    seen_sources: set[str] = set()

    # 1. <script type="..."> tags with known data types
    for script in soup.find_all("script"):
        if len(blobs) >= MAX_BLOBS:
            break

        script_type = (script.get("type") or "").lower().strip()
        script_id = (script.get("id") or "").lower().strip()
        text = script.string or ""

        source = None
        if script_type in _DATA_SCRIPT_TYPES:
            source = script_type
        elif script_id in _DATA_SCRIPT_IDS:
            source = f"id={script.get('id')}"
        elif script_type in ("", "text/javascript", "module"):
            # Check for window.X = {...} patterns
            for m in _WINDOW_ASSIGN_RE.finditer(text):
                var_name = f"window.__{m.group(1)}__"
                ok, data = _try_parse_json(m.group(2))
                if ok and isinstance(data, (dict, list)):
                    source_key = var_name
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        blobs.append({
                            "source": var_name,
                            "data": _preview_blob(data),
                            "keys": _summarise_json_keys(data)[:MAX_BLOB_KEYS],
                        })
            continue  # already handled window patterns above
        else:
            continue

        if not text.strip():
            continue

        ok, data = _try_parse_json(text)
        if not ok:
            continue

        # Skip tiny or trivially small blobs (no useful data)
        if isinstance(data, dict) and len(data) <= 1 and not any(
            isinstance(v, (dict, list)) for v in data.values()
        ):
            continue

        source_key = f"{source}:{script_id or 'anon'}"
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        blobs.append({
            "source": source,
            "data": _preview_blob(data),
            "keys": _summarise_json_keys(data)[:MAX_BLOB_KEYS],
        })

    return blobs


def _preview_blob(data: Any) -> Any:
    """
    Return a compact preview of a JSON blob — large arrays are trimmed,
    deeply nested objects are summarised.
    """
    if isinstance(data, list):
        if len(data) > MAX_ITEMS_IN_ARRAY:
            return {
                "sample": [_preview_blob(item) for item in data[:MAX_ITEMS_IN_ARRAY]],
                "total": len(data),
                "_note": f"{len(data)} items total. Use search_episode_data() for specifics.",
            }
        return [_preview_blob(item) for item in data]

    if isinstance(data, dict):
        result = {}
        for k, v in list(data.items())[:MAX_BLOB_KEYS]:
            if isinstance(v, list) and len(v) > MAX_ITEMS_IN_ARRAY:
                result[k] = {
                    "sample": [_preview_blob(i) for i in v[:MAX_ITEMS_IN_ARRAY]],
                    "total": len(v),
                    "_note": f"{len(v)} items. Use search_episode_data() for specifics.",
                }
            elif isinstance(v, dict) and len(v) > 30:
                # Only collapse very large dicts — preserve small-to-medium ones fully
                # since they often contain critical IDs (e.g. product option configs)
                result[k] = {
                    "_keys": list(v.keys())[:20],
                    "_note": "large nested object — call search_episode_data() for full content",
                }
            else:
                result[k] = v
        return result

    return data


def _extract_forms(soup, base_url: str) -> list[dict]:
    """
    Extract all forms: action URL, method, and all named fields with their values.
    Hidden inputs (CSRF tokens, form_key, etc.) are included.
    """
    forms = []
    for form in soup.find_all("form")[:MAX_FORMS]:
        action = form.get("action", "") or ""
        if base_url and action and not action.startswith("http"):
            action = urljoin(base_url, action)
        method = (form.get("method") or "GET").upper()

        fields: dict[str, str] = {}
        for inp in form.find_all(["input", "select", "textarea"]):
            name = inp.get("name")
            if not name:
                continue
            inp_type = (inp.get("type") or "text").lower()
            value = inp.get("value", "")
            if inp_type == "hidden":
                # Hidden inputs: store actual value (CSRF tokens etc.)
                fields[name] = value
            elif inp_type in ("submit", "button", "reset"):
                continue
            elif inp_type == "checkbox":
                fields[name] = "checkbox"
            elif inp_type == "radio":
                if name not in fields:
                    fields[name] = "radio"
            else:
                # text, email, password, number, etc.
                fields[name] = inp_type if not value else value

        forms.append({
            "action": action,
            "method": method,
            "fields": fields,
        })

    return forms


# Tags whose text content is irrelevant noise
_NOISE_TAGS = {
    "script", "style", "noscript", "head", "meta", "link",
    "header", "footer", "nav", "aside",
    "svg", "path", "symbol",
    "[document]",
}


def _extract_visible_text(soup) -> str | None:
    """
    Extract visible text content from the page.

    Strips scripts, styles, navigation, and other noise.
    Returns plain text, capped at MAX_TEXT_CHARS.
    """
    # Remove noise tags in-place
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Get text from what's left — use separator so words don't jam together
    text = soup.get_text(separator=" ", strip=True)

    # Collapse whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()

    if not text:
        return None

    return text[:MAX_TEXT_CHARS]


def _compact_blob_preview(data: Any) -> str:
    """One-line preview of a JSON blob for the compact text representation."""
    if data is None:
        return "null"
    if isinstance(data, bool):
        return str(data).lower()
    if isinstance(data, (int, float)):
        return str(data)
    if isinstance(data, str):
        return repr(data[:80])
    if isinstance(data, list):
        total = data.get("total") if isinstance(data, dict) else len(data)
        sample = data.get("sample") if isinstance(data, dict) else data[:1]
        if sample:
            first_keys = list(sample[0].keys())[:4] if isinstance(sample[0], dict) else []
            return f"array({total} items), first keys: {first_keys}"
        return f"array({len(data)} items)"
    if isinstance(data, dict):
        # If it has a "total" note it's our preview wrapper
        if "_note" in data and "total" in data:
            sample = data.get("sample", [])
            keys = list(sample[0].keys())[:4] if sample and isinstance(sample[0], dict) else []
            return f"array({data['total']} items), first item keys: {keys}"
        keys = list(data.keys())[:8]
        return f"object({len(data)} keys): {keys}"
    return str(data)[:100]


def _empty_result() -> dict:
    return {
        "page_type": "text_page",
        "title": None,
        "description": None,
        "data_blobs": [],
        "forms": [],
        "text": None,
        "raw_truncated": "",
    }
