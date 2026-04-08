---
title: HARvestGym
emoji: 🕸️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - api-agent
  - web-tasks
---

# HARvestGym

*Core idea: Trains LLMs to reverse-engineer and complete web tasks through raw HTTP APIs. No browser. No docs. Just a URL and a task.*

### Can a small model learn to explore the API surface of any web application — and complete real tasks through those APIs, without ever opening a browser?

Web applications are full of APIs. Every click in a browser triggers an HTTP call with a precise schema, a specific authentication header, an exact sequence of prerequisites. **HARvestGym trains a small model to do all of that directly** — given a task and a URL, it discovers the relevant endpoints, understands what each one needs, chains the calls in the right order, and completes the task without any browser.

The model starts with nothing: no schema, no documentation, no endpoint list. It uses tools to explore — issuing requests, inspecting responses, building up its own understanding of how the application works. This is what a developer does when they reverse-engineer an API. The model learns to do the same.

Given a URL and a task string, the agent must discover which endpoints exist, figure out schemas and parameter dependencies, and execute the right sequence. Zero prior knowledge.

## What the Model (Policy) Is  Learning

Given: a natural language task + a live web application URL. No prior knowledge of the application.

The model calls `browser_agent` first — this returns the list of API endpoints the browser used to complete the task. The model now has a map: it knows what endpoints exist. What it does not know:

- which of those endpoints are actually needed for this specific task
- in what order they must be called (you cannot add to a cart before the cart exists)
- where each required parameter value comes from
- how to re-authenticate if a session expires mid-episode

The model must learn to:

1. **Discover endpoints** — by using a browser agent tool that completes the same task in a real browser while recording all network traffic, then filtering that traffic to extract only the meaningful application API calls (stripping out CDN requests, analytics, static assets). The browser agent runs once and generates the raw discovery data; the model uses this as its starting context.
2. **Select the right endpoints** — from the browser agent's list, identify the subset relevant to the current task (not every observed endpoint is needed)
3. **Sequence calls correctly** — determine the prerequisite order (create cart → find product → add item), including calls that must happen before others even though the task description doesn't say so
4. **Thread parameters** — this is the hardest part. APIs form a dependency graph:
  - Some values come from a previous response (`cart_id` from step 1 → path param in step 3)
  - Some values come from the authentication flow (`form_key`, `Bearer token` → header in every subsequent call)
  - Some values come from the task description (`product name` → search query → `sku` → body of add-item call)
  - The ground truth catalog defines these relationships precisely; the model learns to navigate them
5. **Handle auth and errors** — detect 401 / session-expired responses, re-authenticate, and continue; interpret 4xx errors and adjust the next call accordingly

---

## Architecture

```
                          TRAINING LOOP
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Task + App URL                                                         │
│       │                                                                 │
│       ▼                                                                 │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                  Policy Model (RL Agent)                       │     │
│  │         small model — no prior knowledge of the app           │     │
│  │                                                                │     │
│  │  Observation: task + history + session_state + last_result    │     │
│  │                                                                │     │
│  │  Step 1   ──► browser_agent(task, url)                        │     │
│  │  Step 2+  ──► search_endpoints(query)                         │     │
│  │           ──► curl_exec(command)                              │     │
│  │           ──► search_episode_data(query)                      │     │
│  │           ──► done(result)                                    │     │
│  └────────┬───────────────────────────────────────────────────────┘     │
│           │                                                             │
│    ┌──────┴──────────────────────────────┐                             │
│    │                                     │                             │
│    ▼                                     ▼                             │
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐    │
│  │   Browser Agent     │    │         Environment                 │    │
│  │  (step 1 only)      │    │                                     │    │
│  │                     │    │  • Executes curl_exec via subprocess│    │
│  │ Training:           │    │  • Auto-injects session cookies     │    │
│  │  Load pre-recorded  │    │  • Smart-truncates response bodies  │    │
│  │  cached HAR from    │    │  • Indexes full responses into      │    │
│  │   disk or launch    │    │    per-episode BM25 + GEMMA store   │    │
│  │   on real browser   │    │  • Manages session_state: cookies,  │    │
│  │                     │    │    CSRF tokens, auth headers        │    │
│  │ Inference:          │    └──────────────┬──────────────────────┘    │
│  │  Launch real browser│                   │                           │
│  │  via Playwright +   │                   │ HTTP calls (always live)  │
│  │  bu-30b-a3b-preview │                   ▼                           │
│  │                     │    ┌─────────────────────────────────────┐    │
│  │ Both paths produce: │    │     WebArena EC2 (live apps)        │    │
│  │  • Filtered HAR     │    │                                     │    │
│  │  • OpenAPI-like spec│    │  :7770  Shopping (Magento 2)        │    │
│  │  • GEMMA embeddings │    │  :7780  Shopping Admin              │    │
│  │    for search_      │    │  :9999  Forum (Postmill)            │    │
│  │    endpoints()      │    │  :8888  Wikipedia (Kiwix)          │    │
│  └─────────────────────┘    │  :3000  Map (OpenStreetMap)        │    │
│                             └──────────────┬──────────────────────┘    │
│                                            │                           │
│                                            │ episode trajectory        │
│                                            ▼                           │
│                             ┌─────────────────────────────────────┐    │
│                             │    Deterministic Judge              │    │
│                             │                                     │    │
│                             │  Per-template programmatic grader:  │    │
│                             │  • Inspects episode trajectory      │    │
│                             │  • Optionally probes live app state │    │
│                             │  • Verifies parameter sourcing      │    │
│                             │    (TASK_SPEC / PREV_CALL /         │    │
│                             │     AUTH_FLOW / STATIC / DERIVED)  │    │
│                             │  • Scores [0.0 → 1.0]              │    │
│                             └──────────────┬──────────────────────┘    │
│                                            │                           │
│                                            ▼                           │
│                             ┌─────────────────────────────────────┐    │
│                             │         Reward Signal               │    │
│                             │                                     │    │
│                             │  Per-step:                          │    │
│                             │   +0.2  valid API call (2xx)        │    │
│                             │   +0.1  new path explored           │    │
│                             │   +0.25 correct param sourcing      │    │
│                             │   −0.15 repeated identical call     │    │
│                             │   −0.3  browser_agent called again  │    │
│                             │                                     │    │
│                             │  Episode end:                       │    │
│                             │   +2.0–+5.0 task complete (easy→hard│    │
│                             │   −1.5      task failed             │    │
│                             └──────────────┬──────────────────────┘    │
│                                            │                           │
│                                            ▼                           │
│                             ┌─────────────────────────────────────┐    │
│                             │    GRPO (via HF TRL)                │    │
│                             │                                     │    │
│                             │  8 parallel rollouts per prompt     │    │
│                             │  Computes advantages without        │    │
│                             │  a value function                   │    │
│                             │  Updates policy weights             │    │
│                             └─────────────────────────────────────┘    │
│                                            │                           │
│                                            └──► updated Policy Model   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Browser Agent → Search Index → Execution

```
HAR File (cached using Browser Agent) ──► filter_har_entries()
                                │
                                ▼
                     drop: CDN, analytics, static assets
                     keep: {method, path, request_body,
                             response_body, status_code}
                                │
                                ▼
                     extract_openapi_spec()
                       → structured endpoint catalog
                          {path, method, params, auth, response_fields}
                                │
                         ┌──────┴──────┐
                         │             │
                         ▼             ▼
               build_GEMMA_embeddings  return summary list
               (search_endpoints       to RL agent:
                index — full schemas)    [GET /products,
                                          POST /guest-carts, ...]
                         │
                         ▼
               search_endpoints("create guest cart")
               → top-3 endpoint schemas with:
                  • path params + sources
                  • body params + sources
                  • auth requirements
                  • response field names
```

### Episode Response Indexing

```
curl_exec(command)
     │
     ├──► subprocess: execute against live EC2
     │
     ├──► index_full_response()
     │       BM25 index  ── keyword match (IDs, SKUs, tokens)
     │       GEMMA embed ── semantic match (paraphrases)
     │       (indexes BEFORE truncation — all items stored)
     │
     └──► smart_truncate()
              non-JSON HTML    → 3,000 chars
              JSON primitive   → never truncated
              error (4xx/5xx)  → never truncated
              small JSON       → returned as-is
              large array      → first 2 items shown
                                 + _list_truncated annotation
                                 + hint to call search_episode_data()
```

### Parameter Dependency Graph (what the judge tracks)

```
Task: "Add 'Radiant Tee' to a guest cart"

┌─────────────────────────────────────────────────────────┐
│  TASK_SPEC ──────────────────────────────────────────┐  │
│    "Radiant Tee" (product name)                      │  │
│         │                                            │  │
│         ▼                                            │  │
│  GET /rest/V1/products?name=Radiant+Tee              │  │
│    → items[0].sku = "MH01"          (PREV_CALL) ──┐  │  │
│                                                   │  │  │
│  POST /rest/V1/guest-carts                        │  │  │
│    → body = "cart-abc123"           (PREV_CALL) ──┼──┼─►│
│                                                   │  │  │
│  POST /rest/V1/guest-carts/{cartId}/items         │  │  │
│    path: cartId      ◄────── "cart-abc123" ───────┘  │  │
│    body: sku         ◄────── "MH01"         ─────────┘  │
│    body: qty         ◄────── TASK_SPEC (quantity)       │
│    body: quote_id    ◄────── DERIVED (= cartId)         │
└─────────────────────────────────────────────────────────┘

Source types tracked by the judge:
  TASK_SPEC  — value stated in the task string
  PREV_CALL  — value from a prior curl response in this episode
  AUTH_FLOW  — value from a session/token auth step
  STATIC     — fixed application constant (e.g. store_id = 1)
  DERIVED    — computed from another param (e.g. quote_id = cart_id)
```

### Curriculum: Complexity Tiers

```
  Easy  ──────────────────────── graduate when P(success) > 0.7
  │  Single call, no auth                                    │
  │  Templates 1, 2                                          │
  │  1 API call required                                     │
  │                                                          ▼
  Medium ──────────────────────── graduate when P(success) > 0.7
  │  Auth + 1–2 dependent calls                              │
  │  Templates 3, 4                                          │
  │  2–3 API calls required                                  │
  │                                                          ▼
  Hard ────────────────────────── final tier
     Multi-step chain, full auth, ID threading
     Templates 5, 6, 7
     4–8+ API calls required
     Reward scaling: ×2.5 vs Easy
```

### The RL Agent's Tool: Browser Agent

The RL agent has access to a **browser agent tool** powered by `[browser-use/bu-30b-a3b-preview](https://huggingface.co/browser-use/bu-30b-a3b-preview)` — a 30B MoE vision-language model (3B active parameters) purpose-built for web task completion, served via the [browser-use](https://github.com/browser-use/browser-use) library on Playwright. When the RL agent calls this tool with a natural language task, the browser agent:

1. Opens the target application in a real browser
2. Completes the task by clicking, typing, and navigating — exactly as a human would
3. All HTTP traffic is intercepted via Playwright network events
4. Returns the intercepted traffic, filtered down to only the application's own API calls

The filtering step strips analytics pings, CDN requests, font loads, JS/CSS bundles and returns only `{method, path, request_body, response_body, status_code}` tuples for the app's actual API endpoints.

**Training vs. inference — what gets cached:**

- The browser agent output (filtered endpoint list) is pre-computed once per task and cached. During training, the RL model receives this cached result instantly — no live browser session runs.
- The RL agent's own `curl_exec` calls **always hit the real live WebArena server** — during both training and inference. No API response is mocked or cached.
- At inference, the browser agent runs live to handle novel tasks or changed application state.

Full architecture and code: `[BROWSER_AGENT.md](BROWSER_AGENT.md)`

### Ground Truth: From the Codebase, Not the Browser

The browser agent shows *what* API calls happen. It does not explain *why* — specifically, it does not document where each parameter comes from or what field constraints exist. That comes from the application codebase.

For each WebArena application, we perform a one-time static analysis (using a large model against the Docker image source) to produce a **ground truth API catalog** — a precise, hard-coded document specifying:

```
endpoint:    POST /rest/V1/guest-carts/{cartId}/items
method:      POST
auth:        None (guest cart)
path_params:
  cartId:    [string] obtained from: POST /rest/V1/guest-carts → response body
body:
  cartItem.sku:       [string] the product's SKU, from: GET /rest/V1/products → items[].sku
  cartItem.qty:       [number] quantity, from: task specification
  cartItem.quote_id:  [string] same as cartId
```

This is what the judge compares against. The ground truth defines the complete parameter relationship graph for each application.

Full extraction process: `[GROUND_TRUTH_EXTRACTION.md](GROUND_TRUTH_EXTRACTION.md)`

### The Training Loop

```
Task (natural language) + App URL
          │
          ▼
Policy Model (sees: task + history of all prior actions/results + session_state + findings)
    │  calls tools to explore and execute
    ├─► browser_agent(task, url)   → filtered API call list (cached during training)
    ├─► search_endpoints(query)   → full schema for a specific endpoint
    ├─► curl_exec(command)        → execute HTTP call, get {status, headers, body}
    ├─► search_episode_data(q)    → search prior response bodies in this episode
    └─► done(result)              → declare task complete
          │
          ▼
Live WebArena App (EC2)  ←─── real HTTP responses (always live, never mocked)
          │
          ▼
Judge (compares against ground truth API catalog)
          │
          ▼
Reward Signal  ──►  GRPO  ──►  updated policy
```

---

## Target Applications

All running on a single AWS EC2 instance. Real production software, no simulation.


| App            | Port | URL                                                                                                                        | Software                                                   |
| -------------- | ---- | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Shopping       | 7770 | [http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/](http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/) | Magento 2 — open-source e-commerce platform                |
| Shopping Admin | 7780 | [http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7780/](http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7780/) | Magento 2 Admin — backend panel for the same store         |
| Forum          | 9999 | [http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:9999/](http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:9999/) | Postmill — open-source Reddit-like link aggregation forum  |
| Wikipedia      | 8888 | [http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:8888/](http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:8888/) | Kiwix — read-only offline mirror of English Wikipedia      |
| Map            | 3000 | [http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:3000/](http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:3000/) | OpenStreetMap — open-source collaborative mapping platform |


Source: [WebArena environment_docker](https://github.com/web-arena-x/webarena/tree/main/environment_docker)

---

## Spaces

### Observation Space

What the model sees at each step:

```python
class Observation(BaseModel):
    task: str                  # Natural language task
    app_base_url: str          # Root URL of the target application
    last_tool_result: Any      # Result of last tool call:
                               #   search_endpoints → list of endpoint schema strings
                               #   curl_exec → {status_code, headers, body (smart-truncated)}
                               #   search_episode_data → list of matching JSON object strings
    history: list[dict]        # Full episode trajectory: list of {action, tool_result} pairs
                               # from all prior steps. The model sees what it already tried,
                               # enabling value threading (read a cart_id from step 2's response
                               # and use it in step 5's curl call) and loop avoidance.
    session_state: dict        # Auto-managed by environment: cookies, tokens, CSRF values
                               # extracted from all prior HTTP Set-Cookie and response bodies
                               # e.g. {"PHPSESSID": "abc", "form_key": "xyz", "cart_id": "123"}
    step_count: int
    max_steps: int             # 20 
```

`session_state` is maintained by the environment. The model never parses `Set-Cookie` headers — the environment extracts tokens automatically and makes them available. The model decides *when* to authenticate and *which* session values to use; the environment handles *extraction*.

**curl execution:** The agent outputs a curl command string. The environment parses it and executes it via subprocess against the live EC2 server — the agent machine never has a direct network connection to WebArena. The environment also injects cookies from `session_state` automatically before each call.

**Response truncation — smart array truncation, not byte cutoff:** HTTP response bodies are processed by a pure Python function before being returned to the model. Rules applied in order:

1. **Non-JSON body** (HTML, CSS, JS, plain text): truncate to 3,000 characters. HTML from form-serving pages (login, post creation) is kept longer than pure prose because CSRF tokens and `<input>` fields are embedded inside the markup and the model needs to locate them. See the [HTML / Form-Submission Handling](#html--form-submission-handling) section below for how the model is expected to work with HTML responses.
2. **JSON primitive** (string, number, boolean): never truncated — these are tokens, IDs, confirmations.
3. **Error response (4xx / 5xx)**: never truncated — the model needs every word to self-correct.
4. **JSON object or array with no large arrays** (< 3 dict items per array): returned as-is.
5. **JSON with a large array field** (≥ 3 dict items): keep first 2 items, drop the rest, and add a `_list_truncated` annotation:

```json
{
  "items": [
    {"sku": "MH01", "name": "Radiant Tee", "price": 22.0},
    {"sku": "MH02", "name": "Breathe-Easy Tank", "price": 34.0}
  ],
  "_list_truncated": {
    "field": "items",
    "shown": 2,
    "total": 50,
    "note": "Showing 2 of 50 items. Use search_episode_data() to find a specific item from this response."
  }
}
```

**Episode response indexing:** Every `curl_exec` call indexes the full request and response bodies into a per-episode hybrid index (BM25 for keyword matching + GEMMA semantic embeddings for paraphrase handling). When a list is truncated, all items (not just the 2 shown) are indexed. The model can retrieve any specific object using `search_episode_data("keyword or natural language query")` without needing a filtered API endpoint to exist. See `TOOLS.md` for the full indexing algorithm.

### Action Space

The model outputs a single tool call per step. Full technical specifications for all tools (document construction, truncation implementation, index architecture, caveats) are in `[TOOLS.md](./TOOLS.md)`.


| Tool                         | Input                             | What It Does                                                                                                                                                                              | Output                                                                                                                         |
| ---------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `browser_agent(task, url)`   | Task string + app base URL        | Checks for pre-recorded HAR; if found, processes it — otherwise launches live browser to perform task and record traffic. Extracts OpenAPI-like spec, builds GEMMA embeddings for search. | Summary list of API endpoint names + methods (e.g. `GET /products`). No schemas/headers. Use `search_endpoints()` for details. |
| `search_endpoints(query)`    | Natural language query            | Semantic search over GEMMA-embedded endpoint spec built by `browser_agent`. Returns full parameter details for matching endpoints.                                                        | Top-3 endpoint schemas (method, path, auth, params with sources, response fields)                                              |
| `curl_exec(command)`         | Full curl command string          | Executes HTTP call against live EC2 server, indexes full response into episode BM25 store, returns truncated observation.                                                                 | `{status_code, headers, body}` — body smart-truncated; full body indexed to episode store                                      |
| `search_episode_data(query)` | Keyword or natural language query | Hybrid BM25 + GEMMA semantic search over all request/response bodies from prior `curl_exec` calls in this episode.                                                                        | Top-5 JSON objects from this episode's request/response history                                                                |
| `done(result?)`              | Optional result string            | Signals task complete, triggers judge evaluation.                                                                                                                                         | Ends episode                                                                                                                   |


`browser_agent` is called **exactly once per episode at step 1**. During training, it loads a cached pre-recorded HAR file(if available); at inference, it will launch a live browser session. It returns the deduplicated list of API endpoint patterns observed in the network traffic. **If called again after step 1, the call executes normally but a −0.3 penalty is applied to the reward.** `search_endpoints` then provides the full schema for any specific endpoint the model wants to call — searching the GEMMA embeddings built by `browser_agent` from the HAR data.

`curl_exec` is the primary HTTP action — one string that encodes method, URL, headers, and body together, exactly as API documentation is written. This lets the model leverage its pretrained knowledge of `curl` syntax while producing calls that are self-documenting.

```bash
# Step 1 — Discover which endpoint creates a guest cart
# (model calls search_endpoints first, sees: POST /rest/V1/guest-carts)

# Step 2 — Create guest cart
curl -X POST 'http://ec2-.../rest/V1/guest-carts' -H 'Content-Type: application/json'
# → body: "cart-abc123"  (plain string — never truncated)

# Step 3 — Find the product SKU (list response, truncated to 2 items + note)
curl 'http://ec2-.../rest/V1/products?searchCriteria[filter_groups][0][filters][0][field]=name&searchCriteria[filter_groups][0][filters][0][value]=Radiant+Tee'
# → body: {"items":[{"sku":"MH01","name":"Radiant Tee","price":22.0}],"total_count":1}
# (1 item — not truncated; if 200 items, all 200 indexed, 2 shown in context)

# Step 4 — Add item (model reads cart-abc123 from step 2, MH01 from step 3 — all in history)
curl -X POST 'http://ec2-.../rest/V1/guest-carts/cart-abc123/items' \
  -H 'Content-Type: application/json' \
  -d '{"cartItem":{"sku":"MH01","qty":1,"quote_id":"cart-abc123"}}'
```

Values from prior responses (cart IDs, SKUs, tokens) are threaded directly from the growing episode history. `session_state` tokens (cookies, CSRF values) are auto-injected by the environment. If a list response was truncated and the model needs a specific item not shown in the 2-item sample, it calls `search_episode_data("Radiant Tee sku")` — all 200 items are indexed, even though only 2 were shown in context.

### Prompt Structure:

```
SYSTEM: You are an API agent. Complete the task using only the tools available:
        browser_agent, search_endpoints, curl_exec, search_episode_data, done.
        When a response is HTML, look for JSON data embedded in <script> tags or
        extract values from <input> fields. CSRF tokens appear as hidden inputs:
        <input type="hidden" name="_csrf_token" value="XYZ">

TASK: Add "Radiant Tee" to a guest cart at http://ec2-16-59-2-56.../

[session_state: {}]

STEP 1 ACTION: browser_agent("Add Radiant Tee to a guest cart", "http://ec2-...:7770/")
STEP 1 RESULT: {"app": "shopping", "endpoints": [
  "POST /rest/V1/guest-carts",
  "GET  /rest/V1/products",
  "POST /rest/V1/guest-carts/{id}/items",
  ...
], "note": "Use search_endpoints() to get full schema for any of these."}

STEP 2 ACTION: search_endpoints("create guest cart")
STEP 2 RESULT: ["endpoint: POST /rest/V1/guest-carts | auth: none | returns: string (cartId)", ...]

STEP 3 ACTION: curl_exec("curl -X POST 'http://ec2-.../rest/V1/guest-carts' -H 'Content-Type: application/json'")
STEP 3 RESULT: {status_code: 200, body: "cart-abc123"}

STEP 4 ACTION: search_endpoints("find product by name get sku")
STEP 4 RESULT: ["endpoint: GET /rest/V1/products | query: searchCriteria filters | returns: .items[].sku .items[].name", ...]

STEP 5 ACTION: curl_exec("curl 'http://ec2-.../rest/V1/products?searchCriteria[filter_groups][0][filters][0][field]=name&searchCriteria[filter_groups][0][filters][0][value]=Radiant+Tee'")
STEP 5 RESULT: {status_code: 200, body: {"items":[{"sku":"MH01","name":"Radiant Tee","price":22.0}],"total_count":1}}

STEP 6 ACTION: search_endpoints("add item to guest cart cartId")
STEP 6 RESULT: ["endpoint: POST /rest/V1/guest-carts/{cartId}/items | path: cartId from POST /rest/V1/guest-carts | body: cartItem.sku, cartItem.qty, cartItem.quote_id (same as cartId)", ...]

STEP 7 ACTION: curl_exec("curl -X POST 'http://ec2-.../rest/V1/guest-carts/cart-abc123/items' -H 'Content-Type: application/json' -d '{\"cartItem\":{\"sku\":\"MH01\",\"qty\":1,\"quote_id\":\"cart-abc123\"}}'")
STEP 7 RESULT: {status_code: 200, body: {"item_id": 5, "sku": "MH01", "qty": 1}}

→ generate STEP 8: done("Radiant Tee added to cart")
```

`browser_agent` at step 1 gives the model the full endpoint landscape upfront — it can see `/rest/V1/guest-carts` and `/rest/V1/products` immediately and plan the call sequence before making any HTTP calls. `search_endpoints` fills in the exact parameter schemas. Value threading (`"MH01"`, `"cart-abc123"`) happens through the growing history — if step 5 had returned 200 products truncated to 2, the model would call `search_episode_data("Radiant Tee sku")` to retrieve `MH01` from the episode index.

### Parameter Relationship Graph (What the Judge Knows)

The judge holds a complete dependency map for each task:

```
Parameter Source Types:
  TASK_SPEC    — value given directly in the task (e.g., "product #42")
  PREV_CALL    — value from a prior API response in this episode
  AUTH_FLOW    — value obtained during authentication (session token, CSRF key)
  STATIC       — fixed value known from the application (e.g., store_id = 1)
  DERIVED      — computed from another value (e.g., cart_id = quote_id)
```

For each task, the judge knows which parameters fall into which category, and whether the model correctly sourced each value. This is how partial credit works — the model gets reward for correctly threading a `cart_id` even if the final call had a wrong field elsewhere.

### Reward Space

**Per-step:**


| Signal                       | Value | Trigger                                                                                             |
| ---------------------------- | ----- | --------------------------------------------------------------------------------------------------- |
| Valid API call (2xx)         | +0.2  | `curl_exec` returns 2xx status                                                                      |
| New path called this episode | +0.1  | `curl_exec` normalized path not called before in this episode — discourages looping on one endpoint |
| Correct parameter sourcing   | +0.25 | judge: value in curl call came from the correct source type                                         |
| Session value correctly used | +0.1  | auth token/cookie present and correct in curl call                                                  |
| Repeated identical call      | −0.15 | exact duplicate curl command issued twice                                                           |
| browser_agent called again   | −0.3  | `browser_agent` called after step 1 — call executes normally, penalty applied to reward             |
| Malformed curl command       | −0.1  | curl cannot be parsed or executed by the environment                                                |
| 4xx response (recoverable)   | −0.05 | call failed but episode continues                                                                   |


Note: `search_endpoints`, `search_episode_data`, and `done` carry no direct per-step reward. Using `search_endpoints` to find the correct schema is indirectly rewarded by enabling correct parameter sourcing (+0.25) in the curl call that follows. `search_episode_data` is indirectly rewarded by allowing the model to retrieve the correct value to place in the next curl command.

**Episode end:**


| Outcome                                                     | Reward                                     |
| ----------------------------------------------------------- | ------------------------------------------ |
| Task completed correctly                                    | +2.0 to +5.0 (scales with difficulty tier) |
| Partial completion (right endpoints, wrong param threading) | +0.5 to +1.5                               |
| Authentication correctly obtained (even if task fails)      | +0.3                                       |
| Timeout / task failed entirely                              | −1.5                                       |


Target signal separation: successful episodes `+3` to `+7`, failed episodes `−2` to `−1`. Required for GRPO.

> **Reward design insight:** Pure step-level rewards can teach a model to "look busy" — accumulating +0.2 (valid call) and +0.1 (new path) rewards while never converging to task completion. To prevent this, the terminal outcome reward must dominate the sum of all per-step rewards. Two mechanisms enforce this:
>
> 1. **Hard ceiling on step rewards per episode.** Maximum achievable per-step reward over 20 steps is bounded: `20 × (0.2 + 0.1 + 0.25 + 0.1) = 13`. But a failed episode still ends at `−1.5`, so any correct episode completion still produces a substantially better total.
> 2. **Curriculum learning as the primary defense.** Easy tasks (Template 1: single GET, no auth) have a trivially short optimal path (2 steps). There is no room to accumulate "fake" exploration reward when the optimal episode only needs 2 calls. The model learns that the terminal reward is the only thing that matters before it encounters tasks long enough to be gamed. Medium and Hard tiers are introduced only after the model reliably solves Easy — by then the behavior pattern is already anchored. This mirrors how SWE-gym-style environments scale difficulty: start simple enough that the reward signal is unambiguous, then broaden.
>
> **Premature `done()` penalty:** If the judge scores the final state as incorrect (task not completed), the episode ends at `−1.5`. There is no bonus for calling `done()` early — it is strictly worse than continuing to make correct API calls. The model only benefits from calling `done()` when the task is actually complete.

**Reset behavior:** `reset()` clears session state, episode history, episode BM25 index, step counter. It does not reset the remote application database. The judge evaluates relative state (did the cart contain the item?), not absolute state (is the DB row count exactly N?).

---

## HTML / Form-Submission Handling

Not every endpoint in the target applications returns JSON. The Forum (Postmill) and Wikipedia (Kiwix) applications rely on HTML form submissions and HTML responses respectively. The agent is designed to handle both transparently.

### Why This Matters

A generalizable API agent must work with the full spectrum of web interfaces — not just REST JSON endpoints. Form-based POST submissions (with CSRF tokens, multipart bodies, URL-encoded fields) are ubiquitous in real web applications. Training on them is intentional: the model learns to identify the correct request format from context rather than assuming JSON everywhere.

### CSRF Token Extraction

Postmill protects state-changing routes (login, post creation) with a per-session CSRF token. This token is embedded as a hidden `<input>` field in the HTML form:

```html
<input type="hidden" name="_csrf_token" value="abc123XYZ">
```

**How the model handles this — no dedicated CSRF tool needed:**

1. The model issues a GET to the form page (e.g., `GET /login`).
2. The environment returns the HTML body, truncated to 3,000 characters (raised from 1,000 specifically to ensure hidden input fields near the end of small forms are included).
3. The model reads the `value` attribute of `input[name="_csrf_token"]` directly from the returned HTML string. HTML parsing is not required — the token appears as a predictable plain-text pattern in the markup.
4. The model places the extracted token into the subsequent POST body or form field.
5. The environment auto-extracts any `Set-Cookie` header from the login response into `session_state`, so subsequent requests are automatically authenticated.

If the CSRF token is positioned after the 3,000-character cutoff (possible in very large rendered pages), the model can call `search_episode_data("_csrf_token")` — the full HTML body is indexed into the episode store before truncation, making the token retrievable by keyword search.

```bash
# Forum login flow
curl -X POST 'http://ec2-.../login' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d '_csrf_token=abc123XYZ&_username=user&_password=pass'
# → 302 redirect + Set-Cookie: PHPSESSID=... (auto-injected into session_state)

# Forum post creation
curl -X POST 'http://ec2-.../f/general/submit' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d '_csrf_token=abc123XYZ&title=My+Post&body=Hello+World'
```

### Wikipedia / HTML-Only Responses

Kiwix serves static HTML pages — there is no JSON API. The agent treats Wikipedia responses as structured text: search results appear in `<a href>` anchor tags; article content is in `<p>` tags.

The environment wraps the truncated HTML response in a lightweight JSON envelope before returning it to the model, so the observation format is always `{status_code, headers, body}` regardless of content type:

```json
{
  "status_code": 200,
  "headers": {"Content-Type": "text/html"},
  "body": "<html>...<ul class='mw-search-results'><li><a href='/wiki/Mars'>Mars</a>...</ul>..."
}
```

For Template 2 ("Retrieve article summary for `{title}`"), task completion is verified by confirming the correct article URL was fetched and returned HTTP 200 — not by parsing article content. This makes the grader robust to HTML structure changes.

### Form vs. JSON Detection

`curl_exec` detects whether a request is form-encoded or JSON by inspecting the `Content-Type` header in the curl command string:

- `Content-Type: application/json` → body is JSON, response indexed as JSON
- `Content-Type: application/x-www-form-urlencoded` or `multipart/form-data` → body is form data, response indexed as text
- No `Content-Type` (GET requests) → response indexed based on `Content-Type` of the response

The model is responsible for setting the correct `Content-Type` in its curl command. The system prompt includes explicit guidance on when to use each.

---

## Tasks

HARvestGym trains on **7 task templates** rather than a larger flat task list. Each template is a parameterized scenario: one reward function, one ground truth catalog entry, one grader — but potentially hundreds of distinct episode variations produced by substituting different values for the template slots (`{product_name}`, `{category_name}`, etc.).

If the training went smoothly, then we can scale it to automatically task creation to create all possible aspects of a task.

**How template parameters are populated:** Before training, a one-time data prep step calls the application's own listing APIs and builds a static **parameter pool** for each template (see `[parameter_pools.json](parameter_pools.json)`, refreshed via `[scripts/build_parameter_pools.py](scripts/build_parameter_pools.py)`):


| Template slot                 | Source                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| `{category_name}`             | `GET /rest/V1/categories` — all leaf category names             |
| `{product_name}`              | `GET /rest/V1/products?pageSize=200` — all product names + SKUs |
| `{forum_category}`            | Forum's category listing API                                    |
| `{title}`, `{sku}`, `{price}` | Generated or sampled from existing product names                |


Each episode samples randomly from its pool. The model never sees the pool directly — it gets the task string (e.g., `"Add 'Radiant Tee' to a guest cart"`) and must discover the correct endpoint + SKU through its own API calls.

### Complexity Tiers

Templates are organized into **complexity tiers** for curriculum training — the model only graduates to harder templates once it reliably solves easier ones:


| Tier   | Characteristic                                | API calls required |
| ------ | --------------------------------------------- | ------------------ |
| Easy   | Single call, no auth                          | 1                  |
| Medium | Auth + 1–2 dependent calls                    | 2–3                |
| Hard   | Multi-step chain with ID threading, full auth | 4–8+               |


### Task Templates


| #   | Tier   | App            | Template                                               | Key Challenge                                           |
| --- | ------ | -------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| 1   | Easy   | Shopping       | List products in category `{category_name}`            | Single GET with query params                            |
| 2   | Easy   | Wikipedia      | Retrieve article summary for `{title}`                 | Single GET, path parameter resolution                   |
| 3   | Medium | Shopping       | Add `{product_name}` to a guest cart                   | 2 calls: create cart → add item; ID threading           |
| 4   | Medium | Forum          | Retrieve all posts in `{forum_category}` (authed)      | Login → extract session → GET                           |
| 5   | Hard   | Forum          | Create a post titled `{title}` in `{category}`         | Login → extract CSRF `form_key` → POST with full schema |
| 6   | Hard   | Shopping       | Guest checkout for `{product_name}`                    | 5+ chained calls; cart → item → shipping → payment      |
| 7   | Hard   | Shopping Admin | Create a new product with SKU `{sku}`, price `{price}` | Admin bearer token → full Magento product schema        |


Each task has a deterministic programmatic grader (score in `[0.0, 1.0]`):

- **Easy graders**: check HTTP response body for expected values
- **Medium graders**: probe application state after episode (e.g., fetch the cart, verify item is present)
- **Hard graders**: verify multi-step state change in the application (e.g., post exists, checkout created)

**On optional request parameters:** API responses and real network traffic often contain extra headers and parameters (`X-Requested-With`, `Cache-Control`, correlation IDs, etc.) that are not functionally required. The judge scores only on *required* parameters. Extra or missing optional headers or body params do not affect the reward signal.

---

