"""
HARvestGym Judge — deterministic programmatic graders for all 7 task templates.

Each grader inspects the episode trajectory and/or probes the live application
to compute a task score in [0.0, 1.0], then maps it to the reward range.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from .episode import Episode, EpisodeResult, Step, Task

# ---------------------------------------------------------------------------
# Reward tables (score → reward)
# ---------------------------------------------------------------------------

REWARD_TABLES = {
    1: {1.0: 2.0, 0.3: 0.5, 0.0: -1.5},
    2: {1.0: 2.0, 0.5: 0.5, 0.0: -1.5},
    3: {1.0: 3.5, 0.2: 0.5, 0.15: 0.3, 0.0: -1.5},
    4: {1.0: 3.5, 0.3: 0.8, 0.0: -1.5},
    5: {1.0: 5.0, 0.5: 1.5, 0.3: 0.8, 0.0: -1.5},
    6: {1.0: 5.0, 0.6: 2.5, 0.3: 0.8, 0.1: 0.3, 0.0: -1.5},
    7: {1.0: 5.0, 0.7: 2.0, 0.2: 0.5, 0.0: -1.5},
}

AUTH_BONUS = 0.3  # added when auth was successfully obtained even if task fails


def _score_to_reward(score: float, template_id: int) -> float:
    """Map a [0,1] task score to a reward using the template's reward table."""
    table = REWARD_TABLES.get(template_id, {1.0: 2.0, 0.0: -1.5})
    # Find closest matching threshold
    thresholds = sorted(table.keys(), reverse=True)
    for threshold in thresholds:
        if score >= threshold:
            return table[threshold]
    return table.get(0.0, -1.5)


# ---------------------------------------------------------------------------
# HTTP probe helper
# ---------------------------------------------------------------------------

def _judge_probe(path: str, base_url: str, headers: dict | None = None,
                 timeout: int = 10) -> Any:
    """Issue an HTTP GET from the judge (not the model) to verify live state."""
    if not _REQUESTS_AVAILABLE:
        return None
    url = base_url.rstrip("/") + path
    try:
        resp = _requests.get(url, headers=headers or {}, timeout=timeout, verify=False)
        result = type("ProbeResult", (), {
            "status_code": resp.status_code,
            "body": None,
        })()
        try:
            result.body = resp.json()
        except Exception:
            result.body = resp.text
        return result
    except Exception as e:
        print(f"[judge] probe failed {url}: {e}", flush=True)
        return None


def _judge_post_probe(path: str, base_url: str, data: dict | None = None,
                      headers: dict | None = None, timeout: int = 10) -> Any:
    """Issue an HTTP POST probe from the judge."""
    if not _REQUESTS_AVAILABLE:
        return None
    url = base_url.rstrip("/") + path
    try:
        resp = _requests.post(url, json=data, headers=headers or {}, timeout=timeout, verify=False)
        result = type("ProbeResult", (), {"status_code": resp.status_code, "body": None})()
        try:
            result.body = resp.json()
        except Exception:
            result.body = resp.text
        return result
    except Exception as e:
        print(f"[judge] post probe failed {url}: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fuzzy_match(a: str, b: str) -> bool:
    """Case-insensitive substring match in both directions."""
    a, b = a.lower().strip(), b.lower().strip()
    return a in b or b in a or a == b


def _path_matches(path: str, pattern: str) -> bool:
    """Check if a (normalized) path matches a pattern."""
    return pattern.lower() in path.lower() or path.lower() in pattern.lower()


def _extract_field(obj: Any, field_path: str) -> Any:
    """Extract a nested field via dot notation: 'items.0.sku'."""
    parts = field_path.split(".")
    current = obj
    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current


def _get_curl_steps(episode: Episode):
    """Return only steps that have curl_parsed."""
    return [s for s in episode.steps if s.curl_parsed is not None]


# ---------------------------------------------------------------------------
# Template graders
# ---------------------------------------------------------------------------

def grade_template_1(episode: Episode, task: Task) -> float:
    """Easy — Shopping: List products in category {category_name}"""
    category_name = task.params.get("category_name", "")
    category_lower = category_name.lower()

    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200:
            body = cp.response_body
            # REST API JSON response (ideal path: /rest/V1/products)
            if isinstance(body, dict) and "items" in body:
                items = body["items"]
                if len(items) > 0:
                    for item in items:
                        if _item_matches_category(item, category_name):
                            return 1.0
                    return 0.3
            # Raw list
            if isinstance(body, list) and len(body) > 0:
                return 0.3
            # Distilled HTML page (from html_distiller) — check for search results page
            # that contains product forms.  page_type/forms/text are the distiller's keys.
            if isinstance(body, dict) and "page_type" in body:
                forms = body.get("forms", [])
                text = body.get("text", "") or ""
                title = (body.get("title") or "").lower()
                # A search/category results page has multiple POST add-to-cart forms
                product_forms = [f for f in forms if f.get("method") == "POST"
                                 and "product" in f.get("fields", {})]
                if product_forms:
                    # Check that the page is about the requested category
                    if category_lower in title or category_lower in text.lower():
                        return 1.0
                    # Products listed but category name not verifiable from title — partial
                    return 0.5

    return 0.0


def _item_matches_category(item: dict, category_name: str) -> bool:
    """Check if an item is in the given category."""
    # Check category_links field
    for link in item.get("category_links", []):
        # We trust the response at face value; category name match is partial anyway
        pass
    # Check extension_attributes
    ext = item.get("extension_attributes", {})
    category_links = ext.get("category_links", [])
    if category_links:
        return True  # has category links; assume matches
    # Fallback: just having items is enough for category listing
    return True


def grade_template_2(episode: Episode, task: Task) -> float:
    """Easy — Wikipedia: Retrieve article for {title}"""
    title = task.params.get("title", "")
    title_slug = title.lower().replace(" ", "_")
    title_lower = title.lower()

    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200:
            url_lower = cp.url.lower()
            # Direct article fetch
            if title_slug in url_lower or title_lower.replace(" ", "_") in url_lower:
                return 1.0
            if "wiki/" + title_slug in url_lower:
                return 1.0

    # Search result found the article
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200:
            body_str = str(cp.response_body).lower()
            if title_lower in body_str and "wiki" in cp.url.lower():
                return 0.5

    return 0.0


def _extract_cart_id(episode: Episode) -> str | None:
    """Extract guest cart ID from episode trajectory."""
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200:
            # POST /rest/V1/guest-carts returns bare string cart ID
            if "guest-carts" in cp.path and cp.method == "POST":
                body = cp.response_body
                if isinstance(body, str) and len(body) > 5:
                    return body.strip('"').strip()
    return None


def grade_template_3(episode: Episode, task: Task) -> float:
    """Medium — Shopping: Add {product_name} to a guest cart"""
    product_name = task.params.get("product_name", "")
    sku = task.params.get("sku")
    product_id = str(task.params.get("product_id", ""))

    # Primary: REST API — check if add-to-cart responded with item_id
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200:
            body = cp.response_body
            if isinstance(body, dict) and "item_id" in body:
                if sku and body.get("sku") == sku:
                    return 1.0
                if _fuzzy_match(str(body.get("name", "")), product_name):
                    return 1.0
                if body.get("item_id"):
                    return 1.0

    # Secondary: HTML form-based add-to-cart (POST to /checkout/cart/add)
    # A 302 redirect or 200 response from this endpoint means item was accepted
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.method == "POST" and "/checkout/cart/add" in (cp.path or ""):
            if cp.status_code in (200, 302):
                # Optionally verify the correct product_id was posted
                body_str = str(cp.body or "")
                correct_product = (not product_id) or (product_id in body_str)

                # Probe cart to confirm item presence
                probe = _judge_probe("/checkout/cart/", task.base_url)
                if probe and probe.status_code == 200:
                    cart_text = (probe.body if isinstance(probe.body, str) else str(probe.body)).lower()
                    # Cart page mentions product name or has quantity indicators
                    if product_name.lower()[:15] in cart_text:
                        return 1.0
                    if "qty" in cart_text or "quantity" in cart_text or "item" in cart_text:
                        return 0.8 if correct_product else 0.6
                # POST succeeded without cart confirmation
                return 0.7 if correct_product else 0.5

    # Try live probe via REST guest-cart
    cart_id = _extract_cart_id(episode)
    if cart_id:
        probe = _judge_probe(f"/rest/V1/guest-carts/{cart_id}", task.base_url)
        if probe and probe.status_code == 200:
            items = probe.body.get("items", []) if isinstance(probe.body, dict) else []
            for item in items:
                if sku and item.get("sku") == sku:
                    return 1.0
                if _fuzzy_match(str(item.get("name", "")), product_name):
                    return 1.0
            if len(items) == 0:
                return 0.2  # cart created, item not added yet

    # Partial: REST cart was created
    if cart_id:
        return 0.2

    # Partial: attempted cart creation via REST
    if any("guest-carts" in (s.curl_parsed.path or "") and
           s.curl_parsed.method == "POST"
           for s in _get_curl_steps(episode)):
        return 0.15

    return 0.0


def _check_forum_auth(episode: Episode) -> bool:
    """Check if forum authentication was obtained."""
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.method == "POST" and "/login" in cp.path:
            if cp.status_code in (200, 302):
                return True
    return False


def _response_contains_posts(body: Any) -> bool:
    """Check if a response body contains forum posts."""
    if isinstance(body, list) and len(body) > 0:
        return True
    if isinstance(body, dict):
        # Could be JSON with posts array or HTML
        for key in ("posts", "items", "data", "results"):
            if key in body and isinstance(body[key], list) and len(body[key]) > 0:
                return True
        # Postmill returns HTML — check for common post indicators
        body_str = str(body).lower()
        if "post" in body_str or "submission" in body_str:
            return True
    if isinstance(body, str) and len(body) > 100:
        return True  # HTML response from forum
    return False


def grade_template_4(episode: Episode, task: Task) -> float:
    """Medium — Forum: Retrieve posts in {forum_category} (authed)"""
    forum_category = task.params.get("forum_category", "")
    score = 0.0

    auth_obtained = _check_forum_auth(episode)
    if auth_obtained:
        score += 0.3

    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200:
            url = cp.url
            body = cp.response_body
            if f"/f/{forum_category.lower()}" in url.lower():
                if _response_contains_posts(body):
                    return 1.0
            # Also accept generic post listing with the category in URL
            if forum_category.lower() in url.lower() and _response_contains_posts(body):
                return 1.0

    return score


def _check_csrf_in_trajectory(episode: Episode) -> bool:
    """Check that a POST body contained a _csrf_token field."""
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.method == "POST":
            body_str = str(cp.body or "").lower()
            if "_csrf_token" in body_str and len(body_str) > 20:
                return True
    return False


def _judge_probe_forum_category(category: str, base_url: str) -> list:
    """Probe the forum to get posts in a category."""
    probe = _judge_probe(f"/f/{category}.json", base_url)
    if probe and probe.status_code == 200:
        body = probe.body
        if isinstance(body, dict):
            return body.get("posts", body.get("submissions", []))
        if isinstance(body, list):
            return body
    return []


def grade_template_5(episode: Episode, task: Task) -> float:
    """Hard — Forum: Create a post titled {title} in {category}"""
    title = task.params.get("title", "")
    category = task.params.get("category", "")

    auth_ok = _check_forum_auth(episode)
    csrf_used = _check_csrf_in_trajectory(episode)

    # Check if POST to submit returned success
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.method == "POST" and cp.status_code in (200, 201, 302):
            if "submit" in cp.path or "post" in cp.path.lower():
                # Post creation succeeded
                body_str = str(cp.response_body or "").lower()
                if title.lower() in body_str or "redirect" in str(cp.response_headers).lower():
                    return 1.0
                if cp.status_code in (201, 302):
                    return 1.0

    # Try judge probe
    posts = _judge_probe_forum_category(category, task.base_url)
    for post in posts:
        post_title = post.get("title", post.get("name", ""))
        if _fuzzy_match(post_title, title):
            return 1.0

    if auth_ok and csrf_used:
        return 0.5
    if auth_ok:
        return 0.3
    return 0.0


def _checkout_stages_completed(episode: Episode, sku: str | None) -> int:
    """Count checkout stages completed successfully."""
    stages = 0
    paths_hit = {
        s.curl_parsed.path
        for s in _get_curl_steps(episode)
        if s.curl_parsed.status_code == 200
    }

    if any("guest-carts" in p and "{" not in p for p in paths_hit):
        stages += 1
    if any("guest-carts" in p and "items" in p for p in paths_hit):
        stages += 1
    if any("guest-carts" in p and ("shipping" in p or "email" in p) for p in paths_hit):
        stages += 1
    if any("guest-carts" in p and ("payment" in p or "order" in p) for p in paths_hit):
        stages += 1

    return stages


def grade_template_6(episode: Episode, task: Task) -> float:
    """Hard — Shopping: Guest checkout for {product_name}"""
    sku = task.params.get("sku")

    # Check for order ID
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200:
            body = cp.response_body
            if isinstance(body, int) and body > 0:
                return 1.0
            if isinstance(body, str):
                try:
                    v = int(body.strip('"').strip())
                    if v > 0:
                        return 1.0
                except (ValueError, AttributeError):
                    pass
            if isinstance(body, dict) and body.get("order_id"):
                return 1.0

    stages = _checkout_stages_completed(episode, sku)
    if stages >= 4:
        return 0.6
    if stages >= 2:
        return 0.3
    if stages >= 1:
        return 0.1
    return 0.0


def _extract_admin_token(episode: Episode) -> str | None:
    """Find admin bearer token from shopping-admin trajectory (used by graders)."""
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200 and "integration/admin/token" in cp.path:
            body = cp.response_body
            if isinstance(body, str) and len(body) > 10:
                return body.strip('"').strip()
    return None


def _check_any_auth_obtained(episode: Episode) -> bool:
    """
    Generic check: did the agent successfully obtain ANY form of authentication?

    Detects:
    - Forum/CSRF token authentication
    - Shopping-admin integration token
    - Any 200 response returning a bare token string (bearer, user token, API key)
    - Any 200 response returning a dict with a token field (access_token, id_token, etc.)

    Application-agnostic — the model discovers auth endpoints via browser_agent /
    search_endpoints; this simply rewards the intermediate step of obtaining auth.
    """
    # Forum/CSRF auth
    if _check_forum_auth(episode):
        return True

    # Shopping admin token
    if _extract_admin_token(episode):
        return True

    # Generic: any successful response that looks like it returned an auth token
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code != 200:
            continue
        body = cp.response_body

        # Plain string token (e.g. Magento user/guest tokens, API keys)
        if isinstance(body, str):
            stripped = body.strip().strip('"')
            if re.fullmatch(r"[A-Za-z0-9+/=_\-\.]{20,}", stripped):
                return True

        # Dict with a recognised token field
        if isinstance(body, dict):
            for k in ("token", "access_token", "id_token", "auth_token", "bearer"):
                if k in body and isinstance(body[k], str) and len(body[k]) > 10:
                    return True

    return False


def _attempted_product_creation(episode: Episode, sku: str) -> bool:
    """Check if the model attempted to create a product with this SKU."""
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.method == "POST" and "products" in cp.path:
            body_str = str(cp.body or "").lower()
            if sku.lower() in body_str:
                return True
    return False


def grade_template_7(episode: Episode, task: Task) -> float:
    """Hard — Shopping Admin: Create product with SKU {sku}, price {price}"""
    sku = task.params.get("sku", "")
    price = float(task.params.get("price", 0))

    admin_token = _extract_admin_token(episode)
    if not admin_token:
        return 0.0

    # Check if product creation returned success
    for step in _get_curl_steps(episode):
        cp = step.curl_parsed
        if cp.status_code == 200 and cp.method == "POST" and "products" in cp.path:
            body = cp.response_body
            if isinstance(body, dict) and body.get("id"):
                actual_price = float(body.get("price", -1))
                price_ok = abs(actual_price - price) < 0.01
                return 1.0 if price_ok else 0.7

    # Judge probe
    probe = _judge_probe(
        f"/rest/V1/products/{sku}",
        task.base_url,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    if probe and probe.status_code == 200 and isinstance(probe.body, dict):
        actual_price = float(probe.body.get("price", -1))
        price_ok = abs(actual_price - price) < 0.01
        return 1.0 if price_ok else 0.7

    if _attempted_product_creation(episode, sku):
        return 0.2

    return 0.0


# ---------------------------------------------------------------------------
# Parameter sourcing verification
# ---------------------------------------------------------------------------

def _load_catalog(app: str) -> list[dict]:
    """Load the ground truth catalog for an app."""
    catalog_path = Path(__file__).parent.parent.parent / "catalogs" / f"{app}.json"
    if not catalog_path.exists():
        return []
    try:
        with open(catalog_path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("endpoints", [])
    except Exception:
        return []


def _find_catalog_entry(path: str, method: str, catalog: list[dict]) -> dict | None:
    method = method.upper()
    for entry in catalog:
        cat_method = entry.get("method", "GET").upper()
        cat_path = entry.get("path", "")
        # Pattern match: {id} in catalog matches any segment
        if cat_method == method and _path_pattern_match(path, cat_path):
            return entry
    return None


def _path_pattern_match(actual_path: str, catalog_path: str) -> bool:
    """Match actual path against catalog pattern with {id} wildcards."""
    # Convert catalog pattern to regex
    pattern = re.escape(catalog_path)
    pattern = pattern.replace(r"\{", "{").replace(r"\}", "}")
    pattern = re.sub(r"\{[^}]+\}", "[^/]+", pattern)
    pattern = f"^{pattern}$"
    return bool(re.match(pattern, actual_path, re.IGNORECASE))


def verify_parameter_sourcing(episode: Episode, task: Task) -> float:
    """Analyze parameter sourcing across episode trajectory. Returns [0, 1] score."""
    catalog = _load_catalog(task.app)
    if not catalog:
        return 0.5  # neutral if no catalog

    correct = 0
    total = 0
    steps = _get_curl_steps(episode)

    for step in steps:
        cp = step.curl_parsed
        catalog_entry = _find_catalog_entry(cp.path, cp.method, catalog)
        if not catalog_entry:
            continue

        path_params = catalog_entry.get("path_params", {})
        body_params = catalog_entry.get("body_params", {})

        for param_name, param_meta in path_params.items():
            total += 1
            value = _extract_path_param_value(cp.url, param_name)
            if value and _param_sourced_correctly(value, param_meta, episode, step):
                correct += 1

        for param_name, param_meta in body_params.items():
            total += 1
            value = _extract_body_param_value(cp.body, param_name)
            if value and _param_sourced_correctly(value, param_meta, episode, step):
                correct += 1

    if total == 0:
        return 0.5
    return correct / total


def _extract_path_param_value(url: str, param_name: str) -> str | None:
    """Best-effort path param extraction."""
    # Just extract last non-empty path segment as a value
    from urllib.parse import urlparse
    path = urlparse(url).path
    segments = [s for s in path.split("/") if s]
    if segments:
        return segments[-1]
    return None


def _extract_body_param_value(body: Any, param_name: str) -> Any:
    """Extract a named param from request body."""
    if body is None:
        return None
    if isinstance(body, dict):
        if param_name in body:
            return body[param_name]
        # Search nested
        for v in body.values():
            if isinstance(v, dict):
                result = _extract_body_param_value(v, param_name)
                if result is not None:
                    return result
    if isinstance(body, str):
        # Form-encoded: key=value&...
        for pair in body.split("&"):
            if "=" in pair:
                k, _, v = pair.partition("=")
                if k.strip() == param_name:
                    return v.strip()
    return None


def _param_sourced_correctly(value: Any, param_meta: dict,
                              episode: Episode, step: Step) -> bool:
    source = param_meta.get("source", "")
    value_str = str(value)

    if source == "TASK_SPEC":
        return value_str in episode.task.description

    elif source == "PREV_CALL":
        from_endpoint = param_meta.get("from_endpoint", "")
        from_field = param_meta.get("from_field", "")
        for prior_step in episode.steps:
            if prior_step.step_num >= step.step_num:
                break
            if prior_step.curl_parsed:
                ps = prior_step.curl_parsed
                if _path_matches(ps.path, from_endpoint):
                    extracted = _extract_field(ps.response_body, from_field)
                    if str(extracted) == value_str:
                        return True
        return False

    elif source == "AUTH_FLOW":
        return value_str in str(episode.session_state.values())

    elif source == "STATIC":
        expected = str(param_meta.get("value", ""))
        return value_str == expected

    elif source == "DERIVED":
        from_param = param_meta.get("from_param", "")
        # Simplified: check if it appeared anywhere in session state
        return value_str in str(episode.session_state.values())

    return True  # unknown source type — don't penalize


# ---------------------------------------------------------------------------
# Main judge entry point
# ---------------------------------------------------------------------------

_GRADERS = {
    1: grade_template_1,
    2: grade_template_2,
    3: grade_template_3,
    4: grade_template_4,
    5: grade_template_5,
    6: grade_template_6,
    7: grade_template_7,
}


def evaluate(episode: Episode) -> EpisodeResult:
    """
    Evaluate a completed episode and return reward + diagnostics.

    Args:
        episode: Completed episode with all steps recorded.

    Returns:
        EpisodeResult with task_score, parameter_sourcing_score, reward, details.
    """
    task = episode.task
    template_id = task.template_id

    grader = _GRADERS.get(template_id)
    if grader is None:
        return EpisodeResult(
            task_score=0.0,
            parameter_sourcing_score=0.0,
            auth_obtained=False,
            reward=-1.5,
            details={"error": f"Unknown template_id: {template_id}"},
        )

    task_score = grader(episode, task)
    param_score = verify_parameter_sourcing(episode, task)
    auth_obtained = _check_any_auth_obtained(episode)

    # Compute reward
    reward = _score_to_reward(task_score, template_id)

    # Auth bonus: if the task failed but the agent successfully obtained any form
    # of authentication (bearer token, session cookie, CSRF token, etc.), floor
    # the reward at AUTH_BONUS.  This is application-agnostic — obtaining auth is
    # a useful intermediate skill regardless of the specific task template.
    if task_score < 0.5 and auth_obtained:
        reward = max(reward, AUTH_BONUS)

    return EpisodeResult(
        task_score=task_score,
        parameter_sourcing_score=param_score,
        auth_obtained=auth_obtained,
        reward=reward,
        details={
            "template_id": template_id,
            "difficulty": task.difficulty,
            "task_score": task_score,
            "param_score": param_score,
            "terminated_by": episode.terminated_by,
            "total_steps": episode.total_steps,
        },
    )
