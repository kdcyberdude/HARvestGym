# Browser Agent Component

This document describes the browser agent tool used by the HARvestGym RL agent — how it works, how to build it, and how it integrates with the environment.

---

## What It Is

The browser agent is a multi-stage tool the RL agent calls at the start of every episode. Given a natural language task and a URL, it:

1. **Checks if a pre-recorded HAR file exists** for this application
2. If HAR exists → loads it directly (no browser launched)
3. If no HAR → **launches a real browser** (Chromium via Playwright), connects an LLM, performs the task, and records all network traffic as a HAR file
4. **Processes the HAR** (from either source) to extract an OpenAPI-like spec
5. **Builds GEMMA embeddings** over the extracted spec so `search_endpoints()` can do semantic search
6. **Returns a summary** — the list of API endpoint names and HTTP methods only

The browser agent is a script that orchestrates multiple processing stages. The RL agent sees only the final summary output — a list of endpoints like `GET /products`, `POST /guest-carts`. No headers, no body schemas, no parameter details. To get full details about any endpoint, the agent calls `search_endpoints()` with a natural language query — this searches the GEMMA embeddings built during the browser agent's processing stage.

---

## Library: browser-use

**Repository:** [browser-use/browser-use](https://github.com/browser-use/browser-use)  
**Stars:** 86k+ (April 2026)  
**License:** MIT  
**Language:** Python 3.11+

`browser-use` connects any LLM to a Playwright-controlled browser. The LLM receives the page state (DOM, screenshot, or both), decides on an action (click, type, navigate, extract), and `browser-use` executes it. It uses a sense-plan-act loop with built-in error handling.

Install:

```bash
pip install browser-use
playwright install chromium
```

---

## How It Works: Full Pipeline

### Stage 1 — Obtain HAR Data

The browser agent first checks whether a pre-recorded HAR file exists for the target application. If it does, the browser is never launched — this saves 30–120 seconds per episode.

```python
import json, os
from urllib.parse import urlparse

HAR_MAP = {
    ":7770": "hars/shopping.har",
    ":7780": "hars/shopping_admin.har",
    ":9999": "hars/forum.har",
    ":3000": "hars/osm.har",
    ":8888": "hars/wikipedia.har",
}

def resolve_har_path(url: str) -> str | None:
    """Check if a pre-recorded HAR exists for this app."""
    for port_key, path in HAR_MAP.items():
        if port_key in url:
            if os.path.exists(path):
                return path
    return None


async def get_har_data(task: str, url: str, llm_model: str) -> dict:
    """
    Stage 1: Get HAR data — from file if available, from live browser otherwise.
    Returns the parsed HAR JSON.
    """
    har_path = resolve_har_path(url)

    if har_path:
        # HAR exists — load directly, no browser needed
        with open(har_path) as f:
            return json.load(f)
    else:
        # No HAR — run live browser session and capture traffic
        raw_log = await run_browser_agent_live(task, url, llm_model)
        return convert_raw_log_to_har(raw_log)
```

### Stage 2 — Live Browser Session (only if no HAR exists)

When no pre-recorded HAR is available, the browser agent launches a real Chromium browser, connects the LLM, and performs the task while intercepting all network traffic:

```python
from playwright.async_api import async_playwright
from browser_use import Agent
from langchain_openai import ChatOpenAI

async def run_browser_agent_live(task: str, url: str, llm_model: str) -> list[dict]:
    """
    Runs browser-use agent on the given task, intercepts all network traffic,
    returns raw request/response log.
    """
    requests_log = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # Attach network interceptor
        async def on_request(request):
            requests_log.append({
                "type": "request",
                "url": request.url,
                "method": request.method,
                "headers": dict(request.headers),
                "post_data": request.post_data,
            })

        async def on_response(response):
            try:
                body = await response.text()
            except Exception:
                body = None
            requests_log.append({
                "type": "response",
                "url": response.url,
                "status": response.status,
                "headers": dict(response.headers),
                "body": body,
            })

        page.on("request", on_request)
        page.on("response", on_response)

        # Navigate to app first
        await page.goto(url)

        # Run browser agent
        llm = ChatOpenAI(model=llm_model, base_url="https://router.huggingface.co/v1")
        agent = Agent(task=task, llm=llm, page=page)
        await agent.run()

        await browser.close()

    return requests_log
```

### Stage 3 — Filter and Extract OpenAPI-like Spec

The HAR data (from either source) contains everything: fonts, analytics, CDN requests, JS bundles, CSS. The browser agent filters this down and extracts a structured OpenAPI-like specification:

```python
import re
from urllib.parse import urlparse

SKIP_EXTENSIONS = {".css", ".png", ".jpg", ".svg", ".ico", ".woff", ".woff2", ".ttf", ".gif"}
SKIP_DOMAINS = {"google-analytics.com", "doubleclick.net", "cloudflare.com", "cdn.", "fonts.googleapis.com"}
SKIP_PATH_PREFIXES = ["/static/", "/media/", "/_next/", "/assets/", "/__webpack"]

def is_application_api_call(url: str, app_base_url: str) -> bool:
    parsed = urlparse(url)
    app_host = urlparse(app_base_url).netloc

    if parsed.netloc != app_host:
        return False

    path = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if path.endswith(ext):
            return False
    for prefix in SKIP_PATH_PREFIXES:
        if path.startswith(prefix):
            return False

    return True


def extract_openapi_spec(har_data: dict, app_base_url: str) -> list[dict]:
    """
    Stage 3: Process HAR entries into an OpenAPI-like spec.
    Each entry becomes a structured endpoint document with method, path,
    query params, request body schema, response body schema, status codes, auth info.
    """
    entries = har_data["log"]["entries"]
    seen = set()
    spec_entries = []

    for entry in entries:
        req = entry["request"]
        resp = entry["response"]
        raw_url = req["url"]
        method = req["method"]

        # Filter non-API traffic
        if not is_application_api_call(raw_url, app_base_url):
            continue

        # Skip HTML page navigations
        content_type = _get_response_content_type(resp)
        if "text/html" in content_type and method == "GET":
            continue

        # Normalise path: replace IDs with {id}
        parsed = urlparse(raw_url)
        path = _normalise_path(parsed.path)

        # Deduplicate by (method, normalised_path)
        key = f"{method} {path}"
        if key in seen:
            continue
        seen.add(key)

        # Extract auth info
        has_auth = any(
            h["name"].lower() in ("authorization", "x-api-key", "cookie")
            for h in req["headers"]
        )

        # Build endpoint spec document
        spec_entry = {
            "method": method,
            "path": path,
            "query_params": parsed.query or None,
            "request_headers": {h["name"]: h["value"] for h in req["headers"]
                                if h["name"].lower() in ("content-type", "authorization", "x-requested-with")},
            "request_body": _extract_body(req),
            "status_code": resp["status"],
            "response_content_type": content_type,
            "response_body_sample": _truncate_body(resp),
            "auth_observed": has_auth,
        }
        spec_entries.append(spec_entry)

    return spec_entries
```

### Stage 4 — Build GEMMA Embeddings for Search

The extracted spec entries are converted to text documents and embedded using GEMMA embeddings. These embeddings power the `search_endpoints()` tool — when the RL agent queries "how to add item to cart", the semantic search finds the matching endpoint spec.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def build_endpoint_embeddings(spec_entries: list[dict], app_name: str) -> tuple[np.ndarray, list[str]]:
    """
    Stage 4: Convert spec entries to text chunks and build GEMMA embeddings.
    These embeddings are stored in memory for the duration of the episode,
    enabling search_endpoints() to do semantic search.
    """
    model = SentenceTransformer("google/embeddinggemma-300m")

    chunks = []
    for entry in spec_entries:
        chunk = spec_entry_to_text(entry, app_name)
        chunks.append(chunk)

    # GEMMA encode_document: "title: {endpoint} | text: {rest of chunk}"
    embeddings = model.encode_document(chunks, batch_size=32)
    # Use similarity metric from google/embeddinggemma-300m model card

    return embeddings, chunks


def spec_entry_to_text(entry: dict, app_name: str) -> str:
    """Convert a single spec entry to a searchable text document."""
    parts = [
        f"app: {app_name}",
        f"endpoint: {entry['method']} {entry['path']}",
        f"status: {entry['status_code']}",
        f"auth: {'required' if entry['auth_observed'] else 'none'}",
    ]
    if entry.get("query_params"):
        parts.append(f"query: {entry['query_params']}")
    if entry.get("request_body"):
        parts.append(f"body: {entry['request_body']}")
    if entry.get("response_body_sample"):
        parts.append(f"response_sample: {entry['response_body_sample']}")
    return " | ".join(parts)
```

### Stage 5 — Return Summary to RL Agent

The browser agent returns **only a summary** — endpoint names and HTTP methods. No headers, no body schemas, no parameter details. The agent must call `search_endpoints()` to get the full details.

```python
def build_browser_agent_output(spec_entries: list[dict], app_name: str) -> dict:
    """
    Stage 5: Build the summary output returned to the RL agent.
    This is intentionally sparse — just endpoint names and methods.
    """
    summary_endpoints = []
    for entry in spec_entries:
        summary_endpoints.append({
            "method": entry["method"],
            "path": entry["path"],
        })

    return {
        "app": app_name,
        "endpoints": summary_endpoints,
        "total_endpoints": len(summary_endpoints),
        "note": (
            "These endpoints were observed for this application. "
            "Use search_endpoints() with a natural language query to get "
            "the full schema, parameters, and auth details for any endpoint."
        )
    }
```

### Full Orchestration

```python
async def browser_agent(task: str, url: str) -> dict:
    """
    Complete browser agent pipeline:
    1. Get HAR data (from file or live browser)
    2. Filter and extract OpenAPI-like spec
    3. Build GEMMA embeddings for search_endpoints()
    4. Return summary endpoint list to RL agent
    """
    app_name = resolve_app_name(url)
    llm_model = "browser-use/bu-30b-a3b-preview"

    # Stage 1-2: Get HAR data
    har_data = await get_har_data(task, url, llm_model)

    # Stage 3: Extract OpenAPI-like spec
    spec_entries = extract_openapi_spec(har_data, url)

    # Stage 4: Build GEMMA embeddings (stored in environment for search_endpoints)
    embeddings, chunks = build_endpoint_embeddings(spec_entries, app_name)
    store_episode_embeddings(app_name, embeddings, chunks)  # makes search_endpoints() work

    # Stage 5: Return summary to RL agent
    return build_browser_agent_output(spec_entries, app_name)
```

---

## Output Example

What the RL agent sees (summary only — no schemas, no headers, no body details):

```json
{
  "app": "shopping",
  "endpoints": [
    {"method": "POST", "path": "/rest/V1/integration/customer/token"},
    {"method": "GET",  "path": "/rest/V1/products"},
    {"method": "GET",  "path": "/rest/V1/products/{id}"},
    {"method": "POST", "path": "/rest/V1/guest-carts"},
    {"method": "POST", "path": "/rest/V1/guest-carts/{id}/items"},
    {"method": "GET",  "path": "/rest/V1/guest-carts/{id}/totals"},
    {"method": "POST", "path": "/rest/V1/guest-carts/{id}/order"},
    {"method": "GET",  "path": "/rest/V1/categories"}
  ],
  "total_endpoints": 8,
  "note": "These endpoints were observed for this application. Use search_endpoints() with a natural language query to get the full schema, parameters, and auth details for any endpoint."
}
```

To get full details, the agent calls:
```
search_endpoints("add item to guest cart")
→ returns full schema: POST /rest/V1/guest-carts/{cartId}/items, body params, auth, response fields
```

---

## How search_endpoints() Uses the Embeddings

The GEMMA embeddings built in Stage 4 are what power `search_endpoints()`. When the RL agent calls `search_endpoints("create guest cart")`:

1. The query is encoded using GEMMA `encode_query`
2. Dot product similarity against all endpoint embeddings
3. Top-3 matching endpoint spec documents are returned with full details

```python
def search_endpoints(query: str, embeddings, texts, model, top_k=3) -> list[str]:
    q_emb = model.encode_query(query)          # shape: (D,)
    # Use similarity metric specified by google/embeddinggemma-300m model card
    scores = model.similarity(q_emb, embeddings).squeeze(0)  # shape: (N,)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [texts[i] for i in top_idx]
```

The endpoint documents returned by search contain the full extracted spec — method, path, query params, request body structure, response samples, auth requirements. This is the detailed view that complements the summary list from `browser_agent`.

---

## LLM Choice for Browser Agent

We use **[`browser-use/bu-30b-a3b-preview`](https://huggingface.co/browser-use/bu-30b-a3b-preview)** — a model purpose-built and fine-tuned specifically for browser-use tasks.

| Property | Value |
|----------|-------|
| **Base model** | Qwen3-VL-30B-A3B-Instruct |
| **Architecture** | Vision-Language MoE (Mixture of Experts) |
| **Total parameters** | 30B |
| **Active parameters** | 3B (MoE — only 3B fire per forward pass) |
| **Context length** | 65,536 tokens |
| **Specialization** | Superior DOM understanding + visual reasoning for web tasks |

This model is designed to be served with vLLM and integrates directly with the `browser-use` library via its `ChatOpenAI`-compatible interface:

```python
from browser_use import Agent, ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",    # vLLM server
    model="browser-use/bu-30b-a3b-preview",
    temperature=0.6,
    top_p=0.95,
    dont_force_structured_output=True,      # speeds up inference
)

agent = Agent(task=task, llm=llm)
agent.run_sync()
```

Serve with vLLM:

```bash
vllm serve browser-use/bu-30b-a3b-preview \
  --max-model-len 65536 \
  --host 0.0.0.0 \
  --port 8000
```

Because 3B parameters are active per forward pass (MoE), this model is fast enough for deployment without requiring a full large-model GPU allocation.

---

## Training vs. Inference: What Changes

```
                  Training                          Inference
                     │                                  │
    browser_agent    │  HAR file exists → loads from    │  HAR file may not exist
    Stage 1          │  disk, no browser launched       │  → launches live browser session
                     │                                  │  → records traffic as HAR
                     │                                  │
    browser_agent    │  Processes HAR → extracts spec   │  Same processing pipeline
    Stages 2-4       │  → builds GEMMA embeddings       │  on the live-captured traffic
                     │  → returns summary               │
                     │                                  │
    curl_exec        │  hits REAL live server           │  hits REAL live server
    calls            │  (WebArena EC2)                  │  (WebArena EC2)
                     │                                  │
    judge            │  probes REAL live server         │  probes REAL live server
    verification     │  to verify task completion       │  to verify task completion
```

**What changes between training and inference:** only Stage 1 — where the HAR data comes from. During training, pre-recorded HAR files exist for all tasks, so the browser is never launched. At inference, the HAR may not exist for novel tasks, so the browser runs live.

**What never changes:** Stages 3-5 (spec extraction, embedding, summary output) run identically regardless of the HAR source. And `curl_exec` always hits the real live server — no responses are ever mocked.

---

## Integration with the Environment

```
RL Environment (FastAPI server)
    │
    ├── receives Action: {tool: "browser_agent", input: {task, url}}
    │
    ├── Stage 1: HAR file exists?
    │     ├── YES → load HAR from disk (~0ms)
    │     └── NO  → spawn live browser session (30-120s)
    │               ├── Playwright + bu-30b-a3b-preview
    │               ├── intercept all HTTP traffic
    │               └── produce HAR data
    │
    ├── Stage 3: Extract OpenAPI-like spec from HAR
    │
    ├── Stage 4: Build GEMMA embeddings → stored in env for search_endpoints()
    │
    └── Stage 5: Return summary endpoint list as Observation.last_tool_result
          │
          │  (agent now knows WHAT endpoints exist, but not HOW to call them)
          │
          ▼
    search_endpoints("natural language query")
          │  → semantic search over GEMMA embeddings
          │  → returns full endpoint schema with params, auth, response fields
          ▼
    curl_exec("curl -X POST ...")
          │  → executes against real live WebArena server (EC2)
          │  → indexes full response into episode BM25 store
          ▼
    search_episode_data("keyword query")
          │  → BM25 search over indexed responses from this episode
          ▼
    done() → judge evaluates against ground truth
```

---

## Reference Tools

- [browser-use GitHub](https://github.com/browser-use/browser-use) — the core library
- [browser-use docs](https://docs.browser-use.com) — configuration, custom actions, LLM setup
- [Playwright network events](https://playwright.dev/python/docs/network) — request/response interception API
- [har-to-openapi](https://github.com/jonluca/har-to-openapi) — alternative: convert HAR files to OpenAPI spec format
- [jsluice](https://github.com/BishopFox/jsluice) — extract API routes from JavaScript bundles (useful supplement to network interception) - Future Scope
