# HARvestGym Judge Architecture

This document specifies the full judge architecture — how task completion is verified and how rewards are computed after each episode ends.

The judge is a deterministic, programmatic component. It does **not** use an LLM to score episodes. Every grader produces a score in `[0.0, 1.0]` that is then scaled to the reward range defined in `README.md`.

---

## Overview

```
Episode ends (model calls done() or max_steps=20 reached)
    │
    ▼
Judge.evaluate(episode: Episode, task: Task) → EpisodeResult
    │
    ├─► 1. Identify task template from task.template_id
    │
    ├─► 2. Run programmatic grader for this template
    │       │
    │       ├─► Probe live application state (HTTP calls from judge, not model)
    │       ├─► Inspect episode trajectory (call sequence, parameter sources)
    │       └─► Compute score in [0.0, 1.0]
    │
    ├─► 3. Verify parameter sourcing (for partial credit)
    │       │
    │       └─► Cross-reference each curl call against ground truth catalog
    │
    └─► 4. Compute final reward
            │
            └─► Combine task score + parameter sourcing + step-level signals
```

---

## Data Structures

```python
@dataclass
class Episode:
    task: Task
    steps: list[Step]                # all tool calls and results
    session_state: dict              # final session state
    total_steps: int
    terminated_by: str               # "done_call" | "max_steps"

@dataclass
class Step:
    step_num: int
    tool: str                        # browser_agent | search_endpoints | curl_exec | search_episode_data | done
    action: str                      # raw tool call string
    result: Any                      # tool return value
    curl_parsed: CurlCall | None     # None for non-curl steps

@dataclass
class CurlCall:
    method: str
    url: str
    path: str                        # normalized (IDs replaced with {id})
    headers: dict
    body: dict | str | None
    status_code: int
    response_body: Any

@dataclass
class Task:
    template_id: int                 # 1–7
    description: str                 # instantiated task string (with actual values)
    params: dict                     # e.g. {"product_name": "Radiant Tee", "sku": "MH01"}
    app: str                         # shopping | forum | wikipedia | shopping_admin
    base_url: str
    difficulty: str                  # easy | medium | hard

@dataclass
class EpisodeResult:
    task_score: float                # 0.0–1.0 from grader
    parameter_sourcing_score: float  # 0.0–1.0 from trajectory analysis
    auth_obtained: bool              # did the model successfully authenticate?
    reward: float                    # final composite reward
    details: dict                    # per-grader diagnostic info for logging
```

---

## Graders: Per-Template Verification

Each template has its own grader. All graders make real HTTP calls to the live EC2 application to verify state — they do not rely solely on the episode trajectory.

### Template 1 — Easy | Shopping: List products in category `{category_name}`

**Success condition:** The model's curl call returned a 200 response containing at least one product in the correct category.

```python
def grade_template_1(episode: Episode, task: Task) -> float:
    category_name = task.params["category_name"]

    # Find the curl call that returned products
    for step in episode.steps:
        if step.curl_parsed and step.curl_parsed.status_code == 200:
            body = step.curl_parsed.response_body
            if isinstance(body, dict) and "items" in body:
                items = body["items"]
                # Verify at least one item belongs to the target category
                for item in items:
                    # Check category_links or category name in item
                    if _item_matches_category(item, category_name):
                        return 1.0
                # Items returned but wrong category — partial credit
                if len(items) > 0:
                    return 0.3
    return 0.0

def _item_matches_category(item: dict, category_name: str) -> bool:
    """Check category_links or custom_attributes for category match."""
    # Magento items carry category_links: [{"category_id": N}]
    # Judge verifies by calling GET /rest/V1/categories?searchCriteria[filter...]=name
    # and comparing category IDs. This is a judge-side probe, not relying on model output.
    ...
```

**Reward mapping:**

| Score | Meaning | Reward |
|-------|---------|--------|
| 1.0   | Products listed from correct category | +2.0 |
| 0.3   | Products returned but wrong/unknown category | +0.5 |
| 0.0   | No valid product list response | −1.5 |

---

### Template 2 — Easy | Wikipedia: Retrieve article for `{title}`

**Success condition:** The model made a successful HTTP GET that returned a 200 response for a URL containing the article title (or a redirect to it). Content parsing is explicitly not required.

```python
def grade_template_2(episode: Episode, task: Task) -> float:
    title = task.params["title"]
    title_slug = title.lower().replace(" ", "_")

    for step in episode.steps:
        if step.curl_parsed and step.curl_parsed.status_code == 200:
            url = step.curl_parsed.url.lower()
            if title_slug in url or title.lower() in url:
                return 1.0

    # Check for search result that found the article (indirect)
    for step in episode.steps:
        if step.curl_parsed and step.curl_parsed.status_code == 200:
            body_str = str(step.curl_parsed.response_body).lower()
            if title.lower() in body_str and "wiki" in step.curl_parsed.url.lower():
                return 0.5  # found reference but didn't fetch the article directly

    return 0.0
```

**Reward mapping:**

| Score | Reward |
|-------|--------|
| 1.0   | Correct article URL fetched with 200 | +2.0 |
| 0.5   | Article title found in search results but not fetched | +0.5 |
| 0.0   | No Wikipedia response | −1.5 |

---

### Template 3 — Medium | Shopping: Add `{product_name}` to a guest cart

**Success condition:** Judge probes the cart after the episode to verify the item is present.

```python
def grade_template_3(episode: Episode, task: Task) -> float:
    product_name = task.params["product_name"]
    sku = task.params.get("sku")  # known from parameter pool

    # Extract cart_id from episode trajectory
    cart_id = _extract_cart_id(episode)
    if not cart_id:
        return _partial_score_no_cart(episode)

    # Judge probes the live application
    cart_response = _judge_probe(
        f"GET /rest/V1/guest-carts/{cart_id}",
        task.base_url
    )
    if not cart_response or cart_response.status_code != 200:
        return 0.1  # cart was created but can't be verified

    items = cart_response.body.get("items", [])
    for item in items:
        if item.get("sku") == sku or _fuzzy_match(item.get("name", ""), product_name):
            return 1.0

    # Cart exists but item not in it
    if len(items) == 0 and cart_id:
        return 0.2  # cart created, item not added

    return 0.0

def _partial_score_no_cart(episode: Episode) -> float:
    """Partial credit: did the model attempt the right sequence?"""
    attempted_cart_create = any(
        s.curl_parsed and "guest-carts" in s.curl_parsed.path
        and s.curl_parsed.method == "POST"
        for s in episode.steps if s.curl_parsed
    )
    return 0.15 if attempted_cart_create else 0.0
```

**Reward mapping:**

| Score | Reward |
|-------|--------|
| 1.0   | Item confirmed in cart via judge probe | +3.5 |
| 0.2   | Cart created, item not added | +0.5 |
| 0.15  | Correct call attempted, cart not created | +0.3 |
| 0.0   | No valid attempt | −1.5 |

---

### Template 4 — Medium | Forum: Retrieve all posts in `{forum_category}` (authed)

**Success condition:** The model authenticated and fetched a post listing that includes posts from the target category.

```python
def grade_template_4(episode: Episode, task: Task) -> float:
    forum_category = task.params["forum_category"]
    score = 0.0

    # Check authentication was obtained
    auth_obtained = _check_forum_auth(episode)
    if auth_obtained:
        score += 0.3  # auth is partial credit on its own (see reward table)

    # Find a curl call that returned a post listing for the correct category
    for step in episode.steps:
        if step.curl_parsed and step.curl_parsed.status_code == 200:
            url = step.curl_parsed.url
            body = step.curl_parsed.response_body

            # Postmill returns post listings at /f/{category}
            if f"/f/{forum_category.lower()}" in url.lower():
                if _response_contains_posts(body):
                    return 1.0

    return score  # 0.3 if only auth, 0.0 if nothing

def _check_forum_auth(episode: Episode) -> bool:
    """Authentication: a POST to /login returned a redirect (302) or 200 with session cookie."""
    for step in episode.steps:
        if step.curl_parsed:
            if step.curl_parsed.method == "POST" and "/login" in step.curl_parsed.path:
                if step.curl_parsed.status_code in (200, 302):
                    return True
    return False
```

**Reward mapping:**

| Score | Reward |
|-------|--------|
| 1.0   | Authenticated + posts fetched from correct category | +3.5 |
| 0.3   | Authentication only, no post fetch | +0.8 |
| 0.0   | No valid attempt | −1.5 |

---

### Template 5 — Hard | Forum: Create a post titled `{title}` in `{category}`

**Success condition:** Judge probes the forum category page after the episode to verify the post exists.

```python
def grade_template_5(episode: Episode, task: Task) -> float:
    title = task.params["title"]
    category = task.params["category"]

    # Phase 1: check authentication
    auth_ok = _check_forum_auth(episode)

    # Phase 2: check CSRF token was extracted and used
    csrf_used = _check_csrf_in_trajectory(episode)

    # Phase 3: judge probes the forum to verify post exists
    posts = _judge_probe_forum_category(category, task.base_url)
    for post in posts:
        if _fuzzy_match(post.get("title", ""), title):
            return 1.0

    # Partial credit breakdown
    if auth_ok and csrf_used:
        return 0.5  # got auth and CSRF right, but post didn't land
    if auth_ok:
        return 0.3
    return 0.0

def _check_csrf_in_trajectory(episode: Episode) -> bool:
    """Check that a POST body contained a _csrf_token field."""
    for step in episode.steps:
        if step.curl_parsed and step.curl_parsed.method == "POST":
            body_str = str(step.curl_parsed.body or "")
            if "_csrf_token" in body_str and len(body_str) > 20:
                return True
    return False
```

**Reward mapping:**

| Score | Reward |
|-------|--------|
| 1.0   | Post confirmed in forum via judge probe | +5.0 |
| 0.5   | Auth + CSRF correct, post not created | +1.5 |
| 0.3   | Auth only | +0.8 |
| 0.0   | No valid attempt | −1.5 |

---

### Template 6 — Hard | Shopping: Guest checkout for `{product_name}`

**Success condition:** A complete order was created. Judge checks for an order ID in the trajectory and optionally probes the admin API.

```python
def grade_template_6(episode: Episode, task: Task) -> float:
    sku = task.params.get("sku")

    # Check for order ID in trajectory (checkout success returns an integer order ID)
    for step in episode.steps:
        if step.curl_parsed and step.curl_parsed.status_code == 200:
            body = step.curl_parsed.response_body
            # Magento checkout success: POST /rest/V1/guest-carts/{id}/order returns integer
            if isinstance(body, int) and body > 0:
                return 1.0
            # Magento checkout success: body could also be JSON with "order_id"
            if isinstance(body, dict) and body.get("order_id"):
                return 1.0

    # Partial credit: did the model get through cart + item + shipping?
    stages = _checkout_stages_completed(episode, sku)
    if stages >= 4:  # cart + item + email + shipping estimate
        return 0.6
    if stages >= 2:  # cart + item
        return 0.3
    if stages >= 1:  # cart only
        return 0.1
    return 0.0

def _checkout_stages_completed(episode: Episode, sku: str) -> int:
    """Count how many checkout stages the model completed successfully."""
    stages = 0
    paths_hit = {s.curl_parsed.path for s in episode.steps if s.curl_parsed and s.curl_parsed.status_code == 200}

    if any("guest-carts" in p and "{" not in p for p in paths_hit): stages += 1  # cart created
    if any("guest-carts" in p and "items" in p for p in paths_hit): stages += 1  # item added
    if any("guest-carts" in p and "shipping" in p for p in paths_hit): stages += 1  # shipping
    if any("guest-carts" in p and "payment" in p for p in paths_hit): stages += 1  # payment/order
    return stages
```

**Reward mapping:**

| Score | Reward |
|-------|--------|
| 1.0   | Order created (order_id in response) | +5.0 |
| 0.6   | 4+ stages completed | +2.5 |
| 0.3   | Cart + item only | +0.8 |
| 0.1   | Cart only | +0.3 |
| 0.0   | No valid attempt | −1.5 |

---

### Template 7 — Hard | Shopping Admin: Create product with SKU `{sku}`, price `{price}`

**Success condition:** Judge probes the admin API to confirm the product exists with the correct SKU and price.

```python
def grade_template_7(episode: Episode, task: Task) -> float:
    sku = task.params["sku"]
    price = float(task.params["price"])

    # Phase 1: check admin authentication
    admin_token = _extract_admin_token(episode)
    if not admin_token:
        return 0.0

    # Phase 2: judge probes the Magento REST API to confirm product exists
    product = _judge_probe(
        f"GET /rest/V1/products/{sku}",
        task.base_url,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    if not product or product.status_code != 200:
        # Product might exist under a different auth context — try admin token from env
        product = _judge_probe_with_env_admin_token(f"GET /rest/V1/products/{sku}", task.base_url)

    if product and product.status_code == 200:
        actual_price = float(product.body.get("price", -1))
        price_ok = abs(actual_price - price) < 0.01
        return 1.0 if price_ok else 0.7  # product exists but wrong price

    # Partial credit: correct API called with correct schema
    if _attempted_product_creation(episode, sku):
        return 0.2

    return 0.0

def _extract_admin_token(episode: Episode) -> str | None:
    """Find admin bearer token from a POST /rest/V1/integration/admin/token response."""
    for step in episode.steps:
        if step.curl_parsed and step.curl_parsed.status_code == 200:
            if "integration/admin/token" in step.curl_parsed.path:
                body = step.curl_parsed.response_body
                if isinstance(body, str) and len(body) > 10:
                    return body.strip('"')
    return None
```

**Reward mapping:**

| Score | Reward |
|-------|--------|
| 1.0   | Product confirmed in Magento with correct price | +5.0 |
| 0.7   | Product exists but wrong price | +2.0 |
| 0.2   | Admin auth + correct endpoint called | +0.5 |
| 0.0   | No admin auth | −1.5 |

---

## Parameter Sourcing Verification

In addition to the task-specific grader, the judge runs a parameter sourcing analysis over the full episode trajectory. This cross-references each curl call against the ground truth catalog to verify that parameter values were obtained from the correct sources.

```python
def verify_parameter_sourcing(episode: Episode, task: Task, catalog: list[dict]) -> float:
    """
    Returns a score in [0.0, 1.0] representing how correctly the model
    sourced parameter values across all curl calls in the episode.

    Checks each curl call against the ground truth catalog entry for that endpoint.
    """
    correct = 0
    total = 0

    for step in episode.steps:
        if not step.curl_parsed:
            continue

        catalog_entry = _find_catalog_entry(step.curl_parsed.path, step.curl_parsed.method, catalog)
        if not catalog_entry:
            continue

        # Check each parameter in the curl call
        for param_name, param_meta in catalog_entry.get("path_params", {}).items():
            total += 1
            value_used = _extract_path_param(step.curl_parsed.url, param_name, catalog_entry)
            if value_used and _param_sourced_correctly(value_used, param_meta, episode, step):
                correct += 1

        for param_name, param_meta in catalog_entry.get("body_params", {}).items():
            total += 1
            value_used = _extract_body_param(step.curl_parsed.body, param_name)
            if value_used and _param_sourced_correctly(value_used, param_meta, episode, step):
                correct += 1

    if total == 0:
        return 0.0
    return correct / total

def _param_sourced_correctly(value: Any, param_meta: dict, episode: Episode, step: Step) -> bool:
    """
    Verify that a parameter value came from the expected source.

    Source types:
      TASK_SPEC  — value must appear in the task description string
      PREV_CALL  — value must appear in a prior step's response body
      AUTH_FLOW  — value must come from an auth response (token, session)
      STATIC     — value must match a known constant (e.g., store_id = 1)
      DERIVED    — value must be derivable from another parameter in this call
    """
    source = param_meta.get("source")

    if source == "TASK_SPEC":
        return str(value) in episode.task.description

    elif source == "PREV_CALL":
        from_endpoint = param_meta.get("from_endpoint")
        from_field = param_meta.get("from_field")
        # Check prior steps for a response from from_endpoint with value at from_field
        for prior_step in episode.steps:
            if prior_step.step_num >= step.step_num:
                break
            if prior_step.curl_parsed:
                if _path_matches(prior_step.curl_parsed.path, from_endpoint):
                    extracted = _extract_field(prior_step.curl_parsed.response_body, from_field)
                    if str(extracted) == str(value):
                        return True
        return False

    elif source == "AUTH_FLOW":
        # Value must appear in a session_state field or auth response
        return str(value) in str(episode.session_state.values())

    elif source == "STATIC":
        expected = param_meta.get("value")
        return str(value) == str(expected)

    elif source == "DERIVED":
        same_as = param_meta.get("same_as")
        # Value must equal another param in the same call
        # (e.g., quote_id must equal cart_id which is in the path)
        if same_as and step.curl_parsed:
            other_value = _extract_param_from_call(step.curl_parsed, same_as)
            return str(value) == str(other_value)
        return False

    return False
```

---

## Final Reward Computation

```python
def compute_reward(
    task_score: float,
    parameter_sourcing_score: float,
    step_rewards: float,   # accumulated per-step rewards from README reward table
    auth_obtained: bool,
    task: Task,
    terminated_by: str
) -> float:
    """
    Combines task grader score, parameter sourcing, and step-level signals
    into the final episode reward.
    """
    # Map task score to outcome reward (scales with difficulty tier)
    tier_multipliers = {"easy": 1.0, "medium": 1.75, "hard": 2.5}
    tier = task.difficulty
    m = tier_multipliers.get(tier, 1.0)

    if task_score == 1.0:
        outcome_reward = 2.0 * m       # +2.0 (easy), +3.5 (medium), +5.0 (hard)
    elif task_score >= 0.5:
        outcome_reward = 0.5 * m       # partial
    elif task_score > 0.0:
        outcome_reward = 0.15 * m      # minimal attempt credit
    else:
        outcome_reward = -1.5          # complete failure (same across tiers)

    # Auth bonus: applies even on task failure — model learned authentication
    auth_bonus = 0.3 if auth_obtained and task_score < 1.0 else 0.0

    # Parameter sourcing bonus (weighted into outcome, not additive)
    # Only applied when task succeeds partially — avoids rewarding "busy" episodes
    param_bonus = 0.0
    if 0.0 < task_score < 1.0:
        param_bonus = parameter_sourcing_score * 0.5 * m

    total = outcome_reward + auth_bonus + param_bonus + step_rewards
    return round(total, 4)
```

**Reward separation guarantee:**

| Episode type | Approximate total reward |
|---|---|
| Easy task success (perfect param sourcing) | +2.0 to +3.2 |
| Easy task failure (busy with steps) | −1.5 + max_step_rewards ≈ −0.2 |
| Hard task success | +5.0 to +7.5 |
| Hard task failure (some progress) | −1.5 + partial ≈ −0.5 to +1.5 |

The terminal outcome reward dominates for complete successes and complete failures. Partial episodes sit in the middle — GRPO can distinguish all three signal zones.

---

## Judge Utilities

```python
def _judge_probe(path: str, base_url: str, headers: dict = None) -> ProbeResult | None:
    """
    The judge makes its own HTTP calls to verify application state.
    These calls are NOT part of the episode trajectory and do NOT affect rewards.
    Judge probes use a dedicated admin token from environment variables.
    """
    url = base_url.rstrip("/") + path
    admin_headers = {"Authorization": f"Bearer {os.environ['JUDGE_ADMIN_TOKEN']}"}
    if headers:
        admin_headers.update(headers)
    try:
        resp = requests.get(url, headers=admin_headers, timeout=10)
        return ProbeResult(status_code=resp.status_code, body=resp.json() if resp.text else None)
    except Exception:
        return None

def _fuzzy_match(s1: str, s2: str, threshold: float = 0.85) -> bool:
    """Case-insensitive substring or similarity match."""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2 or s1 in s2 or s2 in s1:
        return True
    # Jaccard similarity as fallback
    tokens1, tokens2 = set(s1.split()), set(s2.split())
    if not tokens1 or not tokens2:
        return False
    return len(tokens1 & tokens2) / len(tokens1 | tokens2) >= threshold
```

---

## Parameter Pool Alignment

The judge is aware that parameter pools are pre-built snapshots of the live application state. For graders that verify values (e.g., SKU, price), the comparison is:

- **SKU matching:** exact string match (SKUs are immutable in Magento)
- **Price matching:** float comparison with ±0.01 tolerance
- **Product name matching:** fuzzy match with 85% threshold (handles whitespace/casing)
- **Category name matching:** fuzzy match, verified against live category tree

The judge does **not** penalize the model if the live application has drifted from the parameter pool (e.g., a product was deleted). In this case, the episode is flagged as `invalid_episode` in the logs and excluded from the training batch. The `build_parameter_pools.py` script should be re-run to refresh the pool if too many episodes are flagged.

---

## Concurrent Episode Isolation

All judge probes use read-only endpoints (GETs, admin token reads) to avoid interfering with other concurrent training episodes. The judge never issues write calls to the live application — it only reads state to verify what the model did.

Write isolation (preventing two concurrent episodes from interfering with each other) is handled at the training harness level, not the judge level:

- For **Easy** tasks (read-only): no isolation needed
- For **Medium** tasks (cart operations): each episode uses a fresh guest cart; carts are session-scoped and do not conflict
- For **Hard** tasks (post creation, product creation): episode IDs are embedded in the task params (e.g., SKU is prefixed with episode ID: `{sku}_{episode_id}`) to prevent naming collisions

---

## Logging and Diagnostics

Every episode produces a structured log entry:

```json
{
  "episode_id": "ep_1234",
  "template_id": 3,
  "task_description": "Add 'Radiant Tee' to a guest cart",
  "task_score": 1.0,
  "parameter_sourcing_score": 0.8,
  "auth_obtained": false,
  "reward": 3.9,
  "step_rewards": 0.85,
  "terminated_by": "done_call",
  "total_steps": 7,
  "grader_details": {
    "cart_id_found": "cart-abc123",
    "item_confirmed_in_cart": true,
    "item_sku": "MH01"
  },
  "parameter_sourcing_details": [
    {"step": 5, "param": "cartId", "source": "PREV_CALL", "correct": true},
    {"step": 7, "param": "cartItem.sku", "source": "PREV_CALL", "correct": true},
    {"step": 7, "param": "cartItem.quote_id", "source": "DERIVED", "correct": true}
  ]
}
```

These logs drive the training analytics and help identify which parameter sourcing patterns the model is still learning.
