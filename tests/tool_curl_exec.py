"""
Tool 2: curl_exec — HTTP execution with truncation and episode indexing.

Pipeline:
1. Parse curl command string → extract method, URL, headers, body
2. Execute via subprocess (or mock in test mode)
3. Index full response into episode BM25 store (before truncation)
4. Truncate response body for context window
5. Return {status_code, headers, body}
"""

import json
import re
import shlex
from typing import Any

# ---------------------------------------------------------------------------
# Curl command parser
# ---------------------------------------------------------------------------


def parse_curl_command(command: str) -> dict:
    """
    Parse a curl command string into structured components.
    Returns: {method, url, headers: dict, body: str|None}
    """
    # Handle the command as a shell argument list
    try:
        parts = shlex.split(command)
    except ValueError:
        return {"error": "Failed to parse curl command"}

    # Remove 'curl' prefix if present
    if parts and parts[0] == "curl":
        parts = parts[1:]

    result = {
        "method": "GET",
        "url": None,
        "headers": {},
        "body": None,
    }

    i = 0
    while i < len(parts):
        part = parts[i]

        if part in ("-X", "--request"):
            i += 1
            if i < len(parts):
                result["method"] = parts[i].upper()

        elif part in ("-H", "--header"):
            i += 1
            if i < len(parts):
                header = parts[i]
                if ":" in header:
                    key, val = header.split(":", 1)
                    result["headers"][key.strip()] = val.strip()

        elif part in ("-d", "--data", "--data-raw"):
            i += 1
            if i < len(parts):
                result["body"] = parts[i]
                if result["method"] == "GET":
                    result["method"] = "POST"

        elif not part.startswith("-"):
            result["url"] = part

        i += 1

    return result


# ---------------------------------------------------------------------------
# Response truncation
# ---------------------------------------------------------------------------

TRUNCATE_LIST_AT = 2
LARGE_ARRAY_THRESHOLD = 3
NONJSON_MAX_CHARS = 1000


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (ValueError, TypeError):
        return False


def truncate_response_body(body: str, status_code: int) -> str:
    """Apply smart truncation rules to response body."""
    # Rule 3: never truncate errors
    if status_code >= 400:
        return body

    # Rule 1: non-JSON
    if not _is_json(body):
        if len(body) > NONJSON_MAX_CHARS:
            return body[:NONJSON_MAX_CHARS] + " [truncated - non-JSON response]"
        return body

    parsed = json.loads(body)

    # Rule 2: primitive
    if not isinstance(parsed, (dict, list)):
        return body

    # Handle top-level array
    if isinstance(parsed, list):
        if (len(parsed) >= LARGE_ARRAY_THRESHOLD
                and len(parsed) > 0 and isinstance(parsed[0], dict)):
            result = parsed[:TRUNCATE_LIST_AT]
            note = {"_list_truncated": {
                "shown": TRUNCATE_LIST_AT,
                "total": len(parsed),
                "note": f"Showing {TRUNCATE_LIST_AT} of {len(parsed)} items. "
                        "Use search_episode_data() to find a specific item from this response."
            }}
            return json.dumps(result + [note])
        return body

    # Handle dict — check each value for large arrays
    needs_truncation = {
        k for k, v in parsed.items()
        if isinstance(v, list) and len(v) >= LARGE_ARRAY_THRESHOLD
           and len(v) > 0 and isinstance(v[0], dict)
    }
    if not needs_truncation:
        return body

    result = {}
    total_truncated = {}
    for k, v in parsed.items():
        if k in needs_truncation:
            result[k] = v[:TRUNCATE_LIST_AT]
            total_truncated[k] = len(v)
        else:
            result[k] = v

    result["_list_truncated"] = {
        "fields": total_truncated,
        "shown_per_field": TRUNCATE_LIST_AT,
        "note": (
            "List fields truncated: "
            + ", ".join(f"{k} showing {TRUNCATE_LIST_AT}/{n}"
                        for k, n in total_truncated.items())
            + ". Use search_episode_data() to find a specific item from this response."
        )
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Episode index document construction
# ---------------------------------------------------------------------------

def build_index_documents(step: int, method: str, path: str,
                           request_body: Any, response_body: Any,
                           status_code: int) -> list[str]:
    """
    Build BM25-indexable documents from a curl_exec result.
    Called BEFORE truncation so all items are indexed.
    """
    docs = []

    # Index request body
    if request_body is not None:
        docs.append(
            f"step:{step} source:request endpoint:{method} {path} "
            f"body:{json.dumps(request_body, ensure_ascii=False) if isinstance(request_body, (dict, list)) else str(request_body)}"
        )

    # Index response body
    if response_body is None:
        return docs

    if isinstance(response_body, str) and not _is_json(response_body):
        docs.append(
            f"step:{step} source:response endpoint:{method} {path} "
            f"status:{status_code} body:{response_body[:500]}"
        )
        return docs

    parsed = json.loads(response_body) if isinstance(response_body, str) else response_body

    # Primitive value
    if isinstance(parsed, (str, int, float, bool)) or parsed is None:
        docs.append(
            f"step:{step} source:response endpoint:{method} {path} "
            f"status:{status_code} value:{parsed}"
        )
        return docs

    # Top-level array
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                docs.append(
                    f"step:{step} source:response endpoint:{method} {path} "
                    f"status:{status_code} item:{json.dumps(item, ensure_ascii=False)}"
                )
            else:
                docs.append(
                    f"step:{step} source:response endpoint:{method} {path} "
                    f"status:{status_code} value:{item}"
                )
        return docs

    # Dict — find array fields
    array_fields = {k: v for k, v in parsed.items()
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)}
    scalar_fields = {k: v for k, v in parsed.items() if k not in array_fields}

    if not array_fields:
        docs.append(
            f"step:{step} source:response endpoint:{method} {path} "
            f"status:{status_code} data:{json.dumps(parsed, ensure_ascii=False)}"
        )
        return docs

    # Array fields — one doc per item with parent context
    parent_context = (
        f"step:{step} source:response endpoint:{method} {path} status:{status_code} "
        + " ".join(f"{k}:{v}" for k, v in scalar_fields.items()
                   if not isinstance(v, (dict, list)))
    )
    for field_name, items in array_fields.items():
        for item in items:
            flat_item = {}
            for k, v in item.items():
                flat_item[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            docs.append(
                f"{parent_context} list_field:{field_name} "
                f"item:{json.dumps(flat_item, ensure_ascii=False)}"
            )

    return docs


# ---------------------------------------------------------------------------
# Mock execution (for testing)
# ---------------------------------------------------------------------------

# Mock responses keyed by (method, path_pattern)
MOCK_RESPONSES = {
    ("GET", "/rest/V1/categories"): {
        "status_code": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "id": 1, "name": "Root",
            "children_data": [
                {"id": 2, "name": "Default Category"},
                {"id": 3, "name": "Beauty & Personal Care"}
            ]
        })
    },
    ("GET", "/rest/V1/products"): {
        "status_code": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "items": [
                {"sku": "MH01", "name": "Radiant Tee", "price": 22.0, "type_id": "simple"},
                {"sku": "MH02", "name": "Breathe-Easy Tank", "price": 34.0, "type_id": "simple"},
                {"sku": "MH03", "name": "Stellar Solar Jacket", "price": 75.0, "type_id": "configurable"},
                {"sku": "MH04", "name": "Argus All-Weather Tank", "price": 22.0, "type_id": "simple"},
            ],
            "total_count": 4
        })
    },
    ("POST", "/rest/V1/guest-carts"): {
        "status_code": 200,
        "headers": {"Content-Type": "application/json"},
        "body": '"cart-mock-abc123"'
    },
    ("POST", "/rest/V1/guest-carts/{id}/items"): {
        "status_code": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "item_id": 5, "sku": "MH01", "qty": 1,
            "name": "Radiant Tee", "price": 22.0,
            "product_type": "simple", "quote_id": "cart-mock-abc123"
        })
    },
}


def mock_curl_exec(command: str, step: int, episode_index: list) -> dict:
    """
    Mock curl_exec for testing. Matches against MOCK_RESPONSES.
    Also builds index documents and adds to episode_index.
    """
    parsed = parse_curl_command(command)
    if "error" in parsed:
        return {"status_code": 0, "error": parsed["error"]}

    method = parsed["method"]
    url = parsed["url"]
    from urllib.parse import urlparse
    path = urlparse(url).path

    # Try exact match first, then pattern match
    response = None
    for (m, p), resp in MOCK_RESPONSES.items():
        if m != method:
            continue
        # Replace {id} with regex for matching
        pattern = re.sub(r'\{[^}]+\}', r'[^/]+', p)
        if re.fullmatch(pattern, path):
            response = resp
            break

    if response is None:
        response = {
            "status_code": 404,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": f"No mock for {method} {path}"})
        }

    # Build index documents BEFORE truncation
    req_body = None
    if parsed["body"]:
        try:
            req_body = json.loads(parsed["body"])
        except (json.JSONDecodeError, TypeError):
            req_body = parsed["body"]

    index_docs = build_index_documents(
        step=step,
        method=method,
        path=path,
        request_body=req_body,
        response_body=response["body"],
        status_code=response["status_code"]
    )
    episode_index.extend(index_docs)

    # Truncate body for context
    truncated_body = truncate_response_body(response["body"], response["status_code"])

    return {
        "status_code": response["status_code"],
        "headers": response["headers"],
        "body": json.loads(truncated_body) if _is_json(truncated_body) else truncated_body,
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: curl_exec with mock responses")
    print("=" * 70)

    # Test curl parsing
    print("\n--- Curl Command Parsing ---")
    commands = [
        'curl http://localhost:7770/rest/V1/categories',
        'curl -X POST http://localhost:7770/rest/V1/guest-carts -H "Content-Type: application/json"',
        "curl -X POST 'http://localhost:7770/rest/V1/guest-carts/cart-abc/items' -H 'Content-Type: application/json' -d '{\"cartItem\":{\"sku\":\"MH01\",\"qty\":1}}'",
    ]
    for cmd in commands:
        parsed = parse_curl_command(cmd)
        print(f"  {cmd[:70]}...")
        print(f"    method={parsed['method']} url={parsed['url']} body={'yes' if parsed['body'] else 'no'}")

    # Test truncation
    print("\n--- Response Truncation ---")

    # Primitive (never truncated)
    assert truncate_response_body('"cart-abc123"', 200) == '"cart-abc123"'
    print("  [OK] Primitive string not truncated")

    # Error (never truncated)
    long_error = json.dumps({"message": "x" * 2000})
    assert truncate_response_body(long_error, 400) == long_error
    print("  [OK] Error response not truncated")

    # Small object (not truncated)
    small = json.dumps({"id": 1, "name": "test"})
    assert truncate_response_body(small, 200) == small
    print("  [OK] Small object not truncated")

    # Large array in dict (truncated to 2 items)
    large = json.dumps({
        "items": [{"sku": f"P{i}", "name": f"Product {i}"} for i in range(20)],
        "total_count": 20
    })
    result = json.loads(truncate_response_body(large, 200))
    assert len(result["items"]) == 2
    assert "_list_truncated" in result
    assert result["_list_truncated"]["fields"]["items"] == 20
    print(f"  [OK] Large array truncated: 20 items → {len(result['items'])} shown")
    print(f"       Note: {result['_list_truncated']['note'][:80]}...")

    # Top-level array (truncated)
    top_array = json.dumps([{"id": i, "name": f"Item {i}"} for i in range(10)])
    result = json.loads(truncate_response_body(top_array, 200))
    assert len(result) == 3  # 2 items + truncation note
    print(f"  [OK] Top-level array truncated: 10 items → 2 shown + note")

    # Test mock execution with indexing
    print("\n--- Mock Execution + Indexing ---")
    episode_index = []

    # Step 1: Get categories
    r = mock_curl_exec("curl http://localhost:7770/rest/V1/categories", 1, episode_index)
    print(f"  Step 1: GET /categories → {r['status_code']}, body keys: {list(r['body'].keys()) if isinstance(r['body'], dict) else 'primitive'}")

    # Step 2: Search products
    r = mock_curl_exec(
        "curl 'http://localhost:7770/rest/V1/products?searchCriteria[filter]=name'",
        2, episode_index
    )
    print(f"  Step 2: GET /products → {r['status_code']}, items shown: {len(r['body'].get('items', []))}, total: {r['body'].get('total_count', '?')}")
    if "_list_truncated" in r["body"]:
        print(f"           Truncated: {r['body']['_list_truncated']['note'][:60]}...")

    # Step 3: Create cart
    r = mock_curl_exec(
        "curl -X POST http://localhost:7770/rest/V1/guest-carts -H 'Content-Type: application/json'",
        3, episode_index
    )
    print(f"  Step 3: POST /guest-carts → {r['status_code']}, cart_id: {r['body']}")

    # Step 4: Add item
    r = mock_curl_exec(
        'curl -X POST http://localhost:7770/rest/V1/guest-carts/cart-mock-abc123/items -H "Content-Type: application/json" -d \'{"cartItem":{"sku":"MH01","qty":1,"quote_id":"cart-mock-abc123"}}\'',
        4, episode_index
    )
    print(f"  Step 4: POST /guest-carts/.../items → {r['status_code']}, item_id: {r['body'].get('item_id')}")

    # Show episode index
    print(f"\n--- Episode Index ({len(episode_index)} documents) ---")
    for i, doc in enumerate(episode_index):
        print(f"  [{i}] {doc[:120]}...")

    print("\n[PASS] curl_exec tool tests completed successfully")
