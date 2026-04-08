"""
curl_exec tool — execute HTTP calls via subprocess, index responses, return truncated result.

Parses curl command string, executes against live EC2 server, auto-injects session cookies,
indexes full response into episode BM25 store, returns smart-truncated observation.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Truncation constants
# ---------------------------------------------------------------------------

NONJSON_MAX_CHARS = 3000      # HTML / plain text truncation (raised for CSRF token visibility)
ARRAY_PREVIEW_ITEMS = 2       # How many items to show in large arrays
ARRAY_LARGE_THRESHOLD = 3     # Arrays >= this size are truncated


# ---------------------------------------------------------------------------
# Curl command parser
# ---------------------------------------------------------------------------

def parse_curl_command(command: str) -> dict:
    """
    Parse a curl command string into components.

    Returns dict with keys: method, url, headers, body, data_type
    """
    # Normalize: remove newline continuations
    command = re.sub(r"\\\s*\n\s*", " ", command)

    try:
        tokens = shlex.split(command)
    except ValueError:
        # Fall back to simple split if shlex fails
        tokens = command.split()

    if not tokens or tokens[0] != "curl":
        raise ValueError(f"Not a curl command: {command[:100]}")

    result: dict = {
        "method": "GET",
        "url": None,
        "headers": {},
        "body": None,
        "data_type": None,  # "json" | "form" | None
    }

    i = 1
    while i < len(tokens):
        tok = tokens[i]

        if tok in ("-X", "--request") and i + 1 < len(tokens):
            result["method"] = tokens[i + 1].upper()
            i += 2

        elif tok in ("-H", "--header") and i + 1 < len(tokens):
            header = tokens[i + 1]
            if ":" in header:
                name, _, value = header.partition(":")
                result["headers"][name.strip().lower()] = value.strip()
            i += 2

        elif tok in ("-d", "--data", "--data-raw", "--data-binary") and i + 1 < len(tokens):
            result["body"] = tokens[i + 1]
            if result["method"] == "GET":
                result["method"] = "POST"
            i += 2

        elif tok == "--data-urlencode" and i + 1 < len(tokens):
            # Append to existing body
            existing = result.get("body") or ""
            if existing:
                result["body"] = existing + "&" + tokens[i + 1]
            else:
                result["body"] = tokens[i + 1]
            if result["method"] == "GET":
                result["method"] = "POST"
            i += 2

        elif tok in ("-F", "--form") and i + 1 < len(tokens):
            existing = result.get("body") or ""
            if existing:
                result["body"] = existing + "&" + tokens[i + 1]
            else:
                result["body"] = tokens[i + 1]
            if result["method"] == "GET":
                result["method"] = "POST"
            i += 2

        elif tok in ("-u", "--user") and i + 1 < len(tokens):
            i += 2  # skip basic auth for now

        elif tok in ("-L", "--location", "-s", "--silent", "-v", "--verbose",
                     "-k", "--insecure", "--compressed", "-g", "--globoff"):
            i += 1

        elif tok in ("-o", "--output", "--max-time", "--connect-timeout",
                     "--retry", "-A", "--user-agent", "-e", "--referer"):
            i += 2  # skip flag + value

        elif not tok.startswith("-") and result["url"] is None:
            result["url"] = tok.strip("'\"")
            i += 1

        elif tok.startswith("http"):
            result["url"] = tok.strip("'\"")
            i += 1

        else:
            i += 1

    # Infer data_type from content-type header
    ct = result["headers"].get("content-type", "")
    if "application/json" in ct:
        result["data_type"] = "json"
    elif "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
        result["data_type"] = "form"
    elif result["body"]:
        # Guess from body
        if result["body"].strip().startswith("{") or result["body"].strip().startswith("["):
            result["data_type"] = "json"
        else:
            result["data_type"] = "form"

    return result


# ---------------------------------------------------------------------------
# Smart truncation
# ---------------------------------------------------------------------------

def smart_truncate(body_text: str, content_type: str = "") -> Any:
    """
    Apply truncation rules to a response body string.

    Rules (first match wins):
    1. Non-JSON → truncate to NONJSON_MAX_CHARS
    2. JSON primitive (str/int/bool/null) → never truncate
    3. Error (detected by content) → never truncate
    4. JSON object/array with no large arrays → return as-is
    5. JSON with large array → keep first ARRAY_PREVIEW_ITEMS, add _list_truncated note
    """
    if not body_text:
        return ""

    # Rule 1: non-JSON
    if "application/json" not in content_type and not _looks_like_json(body_text):
        return body_text[:NONJSON_MAX_CHARS]

    # Try to parse as JSON
    try:
        parsed = json.loads(body_text)
    except (json.JSONDecodeError, ValueError):
        return body_text[:NONJSON_MAX_CHARS]

    # Rule 2: JSON primitive
    if not isinstance(parsed, (dict, list)):
        return parsed

    # Rule 3: detect error (4xx/5xx already handled by caller; this checks body content)
    if isinstance(parsed, dict) and ("message" in parsed or "error" in parsed):
        return parsed  # never truncate errors

    # Rules 4 and 5
    return _truncate_json(parsed)


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[") or stripped.startswith('"')


def _truncate_json(obj: Any) -> Any:
    if isinstance(obj, list):
        if len(obj) >= ARRAY_LARGE_THRESHOLD:
            return {
                "items": obj[:ARRAY_PREVIEW_ITEMS],
                "_list_truncated": {
                    "shown": ARRAY_PREVIEW_ITEMS,
                    "total": len(obj),
                    "note": (
                        f"Showing {ARRAY_PREVIEW_ITEMS} of {len(obj)} items. "
                        "Use search_episode_data() to find a specific item from this response."
                    ),
                },
            }
        return obj

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, list) and len(v) >= ARRAY_LARGE_THRESHOLD:
                result[k] = v[:ARRAY_PREVIEW_ITEMS]
                result["_list_truncated"] = {
                    "field": k,
                    "shown": ARRAY_PREVIEW_ITEMS,
                    "total": len(v),
                    "note": (
                        f"Showing {ARRAY_PREVIEW_ITEMS} of {len(v)} items. "
                        "Use search_episode_data() to find a specific item from this response."
                    ),
                }
            else:
                result[k] = v
        return result

    return obj


# ---------------------------------------------------------------------------
# Cookie injection
# ---------------------------------------------------------------------------

def _inject_cookies(headers: dict, session_state: dict) -> dict:
    """Inject cookies from session_state into the request headers."""
    headers = dict(headers)  # copy

    # Collect cookie values
    cookie_parts = []
    for key, value in session_state.items():
        if key.lower() in ("phpsessid", "sessid", "session", "cookie",
                           "mage-cache-sessid", "private_content_version",
                           "form_key", "PHPSESSID"):
            cookie_parts.append(f"{key}={value}")

    # Check if there's a raw cookie header already
    existing = headers.get("cookie", "")
    if cookie_parts:
        all_cookies = existing + ("; " if existing else "") + "; ".join(cookie_parts)
        headers["cookie"] = all_cookies

    return headers


# ---------------------------------------------------------------------------
# Session state extraction
# ---------------------------------------------------------------------------

def _extract_set_cookies(response_headers: dict, session_state: dict) -> None:
    """Extract Set-Cookie headers into session_state."""
    for name, value in response_headers.items():
        if name.lower() == "set-cookie":
            # Parse "NAME=VALUE; Path=...; ..."
            cookies = value.split(";")
            if cookies:
                kv = cookies[0].strip()
                if "=" in kv:
                    k, _, v = kv.partition("=")
                    session_state[k.strip()] = v.strip()


def _extract_tokens_from_body(body: Any, session_state: dict) -> None:
    """Extract auth tokens from JSON response bodies into session_state."""
    if isinstance(body, str) and len(body) > 10 and len(body) < 500:
        # Likely a token (Magento returns bare quoted strings for auth tokens)
        stripped = body.strip('"').strip()
        if re.match(r"^[A-Za-z0-9_\-\.]{20,}$", stripped):
            session_state["_last_token"] = stripped

    if isinstance(body, dict):
        for key in ("access_token", "token", "cart_id", "form_key"):
            if key in body and body[key]:
                session_state[key] = body[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def curl_exec(command: str, session_state: dict, episode_store: dict,
              app_base_url: str = "") -> dict:
    """
    Parse and execute a curl command against the live app.

    Args:
        command: Full curl command string
        session_state: Current session state (cookies/tokens), mutated in place
        episode_store: Per-episode store for BM25 indexing, mutated in place
        app_base_url: Base URL to validate requests against

    Returns:
        {status_code, headers, body} with smart-truncated body
    """
    try:
        parsed = parse_curl_command(command)
    except Exception as e:
        return {"status_code": -1, "headers": {}, "body": f"curl parse error: {e}", "error": str(e)}

    if not parsed["url"]:
        return {"status_code": -1, "headers": {}, "body": "No URL in curl command", "error": "missing url"}

    # Inject session cookies
    parsed["headers"] = _inject_cookies(parsed["headers"], session_state)

    # Build actual curl args
    args = ["curl", "-s", "-i", "-L", "--max-time", "15"]
    args += ["-X", parsed["method"]]
    args += [parsed["url"]]

    for h_name, h_val in parsed["headers"].items():
        args += ["-H", f"{h_name}: {h_val}"]

    if parsed["body"]:
        args += ["-d", parsed["body"]]

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=20,
        )
        raw_output = result.stdout
    except subprocess.TimeoutExpired:
        return {"status_code": -1, "headers": {}, "body": "Request timed out (20s)", "error": "timeout"}
    except Exception as e:
        return {"status_code": -1, "headers": {}, "body": f"subprocess error: {e}", "error": str(e)}

    # Parse HTTP response: headers + body split at blank line
    status_code = 0
    resp_headers: dict[str, str] = {}
    body_text = ""

    if raw_output:
        # Find status line (handle redirects: multiple HTTP/ headers)
        lines = raw_output.split("\r\n") if "\r\n" in raw_output else raw_output.split("\n")
        header_lines = []
        body_lines = []
        in_body = False
        last_status = 0

        for line in lines:
            if in_body:
                body_lines.append(line)
            elif line.startswith("HTTP/"):
                # Could be redirect status; keep last
                parts = line.split(" ", 2)
                if len(parts) >= 2:
                    try:
                        last_status = int(parts[1])
                    except ValueError:
                        pass
                header_lines = []  # reset headers for this response
            elif line.strip() == "":
                if last_status:  # we've seen at least one status line
                    in_body = True
            else:
                header_lines.append(line)

        status_code = last_status
        body_text = "\n".join(body_lines).strip()

        for h_line in header_lines:
            if ":" in h_line:
                h_name, _, h_val = h_line.partition(":")
                resp_headers[h_name.strip().lower()] = h_val.strip()

    # Extract cookies / tokens into session_state
    _extract_set_cookies(resp_headers, session_state)

    # Try to parse body as JSON
    resp_ct = resp_headers.get("content-type", "")
    parsed_body: Any = body_text
    try:
        parsed_body = json.loads(body_text) if body_text else ""
    except (json.JSONDecodeError, ValueError):
        parsed_body = body_text

    # Extract tokens from body
    _extract_tokens_from_body(parsed_body, session_state)

    # Index into episode BM25 store BEFORE truncation
    _index_into_episode_store(
        episode_store=episode_store,
        request_body=parsed["body"],
        response_body=parsed_body,
        url=parsed["url"],
        method=parsed["method"],
        status_code=status_code,
    )

    # Apply smart truncation
    if status_code >= 400:
        # Never truncate errors
        truncated_body = parsed_body
    else:
        body_for_truncation = body_text if isinstance(parsed_body, str) else json.dumps(parsed_body)
        truncated_body = smart_truncate(body_for_truncation, resp_ct)

    return {
        "status_code": status_code,
        "headers": resp_headers,
        "body": truncated_body,
    }


# ---------------------------------------------------------------------------
# Episode store indexing
# ---------------------------------------------------------------------------

def _index_into_episode_store(episode_store: dict, request_body: Any,
                               response_body: Any, url: str, method: str,
                               status_code: int) -> None:
    """Index request/response into episode BM25 store for search_episode_data()."""
    if "bm25_corpus" not in episode_store:
        episode_store["bm25_corpus"] = []
        episode_store["bm25_metadata"] = []

    def _to_text(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        return json.dumps(obj)

    entry_text = f"url: {url} | method: {method} | status: {status_code} | " \
                 f"request: {_to_text(request_body)} | response: {_to_text(response_body)}"

    episode_store["bm25_corpus"].append(entry_text)
    episode_store["bm25_metadata"].append({
        "url": url,
        "method": method,
        "status_code": status_code,
        "response_body": response_body,
    })
