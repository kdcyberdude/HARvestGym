---
title: HARvestGym
emoji: 🕸️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - api-agent
  - web-tasks
---

# HARvestGym

### Can a small model learn to reverse-engineer any web application's API — and complete real tasks through those APIs, without ever opening a browser?

Web applications are full of APIs. Every click in a browser triggers an HTTP call with a precise schema, a specific authentication header, an exact sequence of prerequisites. **HARvestGym trains a small model to do all of that directly** — given a task and a URL, it discovers the relevant endpoints, figures out what each one needs, chains the calls in the right order, and completes the task without any browser.

The model starts with nothing: no schema, no documentation, no endpoint list. It uses tools to explore — issuing requests, inspecting responses, building up its own understanding of how the application works. This is what a developer does when they reverse-engineer an API. The model learns to do the same.

---

## Why It Matters

The environment HARvestGym trains — discover an API, understand its dependencies, complete a task through it — is one of the most broadly useful things an AI agent can do.

| Application | What it unlocks |
| --- | --- |
| 🔍 **API reverse engineering** | Point an agent at any web app and get a working API map — no docs, no SDK, no source code required |
| 📄 **Automatic API documentation** | Capture real call sequences and parameter provenance to produce accurate, living documentation from observed traffic |
| 🤖 **Browser-free automation** | Automate form submissions, cart flows, content management, and data extraction at the HTTP layer — immune to UI redesigns |
| 🔧 **Any website as MCP tools** | Turn any web application into a set of callable agent tools on the fly, with no official integration needed |
| 🛡️ **Security & API auditing** | Autonomously probe endpoints, trace auth flows, and map attack surfaces for penetration testing and compliance review |

---

## How It Works

```
Task + App URL
      │
      ▼
Policy Model (RL Agent)
  small model — no prior knowledge of the app

  Step 1     ──► browser_agent(task, url)     → filtered API endpoint list
  Step 2+    ──► search_endpoints(query)      → full schema for a specific endpoint
             ──► curl_exec(command)           → execute HTTP call, get response
             ──► search_episode_data(query)   → search prior response bodies
             ──► done(result)                 → declare task complete
      │
      ▼
Live WebArena Apps (EC2)  ←── real HTTP responses (always live, never mocked)
      │
      ▼
Deterministic Judge (compares against ground truth API catalog)
      │
      ▼
Reward Signal  ──►  GRPO  ──►  updated policy
```

The agent calls `browser_agent` once at the start — this runs a real browser to complete the same task while recording all network traffic, then returns the filtered list of API endpoints observed. The agent now has a map of what endpoints exist. What it does *not* know:

- which of those endpoints are actually needed for this specific task
- in what order they must be called (you cannot add to a cart before the cart exists)
- where each required parameter value comes from
- how to re-authenticate if a session expires mid-episode

The model must learn to discover all of this on its own.

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
│  │         small model — no prior knowledge of the app            │     │
│  │                                                                │     │
│  │  Observation: task + history + session_state + last_result     │     │
│  │                                                                │     │
│  │  Step 1   ──► browser_agent(task, url)                         │     │
│  │  Step 2+  ──► search_endpoints(query)                          │     │
│  │           ──► curl_exec(command)                               │     │
│  │           ──► search_episode_data(query)                       │     │
│  │           ──► done(result)                                     │     │
│  └────────┬───────────────────────────────────────────────────────┘     │
│           │                                                             │
│    ┌──────┴──────────────────────────────┐                              │
│    │                                     │                              │
│    ▼                                     ▼                              │
│  ┌─────────────────────┐    ┌─────────────────────────────────────┐     │
│  │   Browser Agent     │    │         Environment                 │     │
│  │  (step 1 only)      │    │                                     │     │
│  │                     │    │  • Executes curl_exec via subprocess│     │
│  │ Training:           │    │  • Auto-injects session cookies     │     │
│  │  Load pre-recorded  │    │  • Smart-truncates response bodies  │     │
│  │  cached HAR from    │    │  • Indexes full responses into      │     │
│  │   disk or launch    │    │    per-episode BM25 + GEMMA store   │     │
│  │   on real browser   │    │  • Manages session_state: cookies,  │     │
│  │                     │    │    CSRF tokens, auth headers        │     │
│  │ Inference:          │    └──────────────┬──────────────────────┘     │
│  │  Launch real browser│                   │                            │
│  │  via Playwright +   │                   │ HTTP calls (always live)   │
│  │  bu-30b-a3b-preview │                   ▼                            │
│  │                     │    ┌─────────────────────────────────────┐     │
│  │ Both paths produce: │    │     WebArena EC2 (live apps)        │     │
│  │  • Filtered HAR     │    │                                     │     │
│  │  • OpenAPI-like spec│    │  :7770  Shopping (Magento 2)        │     │
│  │  • GEMMA embeddings │    │  :7780  Shopping Admin              │     │
│  │    for search_      │    │  :9999  Forum (Postmill)            │     │
│  │    endpoints()      │    │  :8888  Wikipedia (Kiwix)           │     │
│  └─────────────────────┘    │  :3000  Map (OpenStreetMap)         │     │
│                             └──────────────┬──────────────────────┘     │
│                                            │                            │
│                                            │ episode trajectory         │
│                                            ▼                            │
│                             ┌─────────────────────────────────────┐     │
│                             │    Deterministic Judge              │     │
│                             │                                     │     │
│                             │  Per-template programmatic grader:  │     │
│                             │  • Inspects episode trajectory      │     │
│                             │  • Optionally probes live app state │     │
│                             │  • Verifies parameter sourcing      │     │
│                             │    (TASK_SPEC / PREV_CALL /         │     │
│                             │     AUTH_FLOW / STATIC / DERIVED)   │     │
│                             │  • Scores [0.0 → 1.0]               │     │
│                             └──────────────┬──────────────────────┘     │
│                                            │                            │
│                                            ▼                            │
│                             ┌─────────────────────────────────────┐     │
│                             │         Reward Signal               │     │
│                             │                                     │     │
│                             │  Per-step:                          │     │
│                             │   +0.2  valid API call (2xx)        │     │
│                             │   +0.1  new path explored           │     │
│                             │   +0.25 correct param sourcing      │     │
│                             │   −0.15 repeated identical call     │     │
│                             │   −0.3  browser_agent called again  │     │
│                             │                                     │     │
│                             │  Episode end:                       │     │
│                             │   +2.0–+5.0 task complete (easy→hard│     │
│                             │   −1.5      task failed             │     │
│                             └──────────────┬──────────────────────┘     │
│                                            │                            │
│                                            ▼                            │
│                             ┌─────────────────────────────────────┐     │
│                             │    GRPO (via HF TRL)                │     │
│                             │                                     │     │
│                             │  8 parallel rollouts per prompt     │     │
│                             │  Computes advantages without        │     │
│                             │  a value function                   │     │
│                             │  Updates policy weights             │     │
│                             └─────────────────────────────────────┘     │
│                                            │                            │
│                                            └──► updated Policy Model    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Target Applications

All running on a single AWS EC2 instance — real production software, no simulation.

| App            | Port | Software                                          |
| -------------- | ---- | ------------------------------------------------- |
| Shopping       | 7770 | Magento 2 — open-source e-commerce platform       |
| Shopping Admin | 7780 | Magento 2 Admin — backend panel for the same store|
| Forum          | 9999 | Postmill — open-source Reddit-like forum          |
| Wikipedia      | 8888 | Kiwix — read-only offline mirror of Wikipedia     |
| Map            | 3000 | OpenStreetMap — collaborative mapping platform    |

Source: [WebArena environment_docker](https://github.com/web-arena-x/webarena/tree/main/environment_docker)

---

## Tasks

HARvestGym trains on **7 task templates** across three complexity tiers. Each template is a parameterized scenario: one reward function, one ground truth catalog entry, one grader — but potentially hundreds of distinct episode variations produced by substituting different values for the template slots (`{product_name}`, `{category_name}`, etc.).

### Complexity Tiers

| Tier   | Characteristic                                | API calls required |
| ------ | --------------------------------------------- | ------------------ |
| Easy   | Single call, no auth                          | 1                  |
| Medium | Auth + 1–2 dependent calls                    | 2–3                |
| Hard   | Multi-step chain with ID threading, full auth | 4–8+               |

The model only graduates to harder templates once it reliably solves easier ones.

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

**Template parameters** are populated from a static parameter pool built by querying the live applications before training (see `parameter_pools.json`, refreshed via `scripts/build_parameter_pools.py`). Each episode samples randomly from its pool — the model never sees the pool directly, it must discover the correct values through its own API calls.

Each task has a deterministic programmatic grader (score in `[0.0, 1.0]`):
- **Easy graders**: check HTTP response body for expected values
- **Medium graders**: probe application state after episode (e.g., fetch the cart, verify item is present)
- **Hard graders**: verify multi-step state change in the application (e.g., post exists, checkout created)

---

## Spaces

### Observation Space

What the model sees at each step:

```python
class Observation(BaseModel):
    task: str                  # Natural language task
    app_base_url: str          # Root URL of the target application
    last_tool_result: Any      # Result of last tool call
    history: list[dict]        # Full episode trajectory: [{action, tool_result}, ...]
    session_state: dict        # Auto-managed: cookies, tokens, CSRF values
    step_count: int
    max_steps: int             # 20
```

`session_state` is maintained by the environment — the model decides *when* to authenticate and *which* session values to use; the environment handles *extraction* from `Set-Cookie` headers and response bodies.

**Response truncation** rules applied in order:
1. Non-JSON body (HTML, CSS): truncated to 3,000 characters
2. JSON primitive (string, number): never truncated — these are tokens, IDs
3. Error response (4xx/5xx): never truncated — the model needs every word to self-correct
4. Small JSON (no large arrays): returned as-is
5. Large JSON array (≥ 3 items): first 2 items shown + `_list_truncated` annotation + hint to call `search_episode_data()`

Every `curl_exec` call indexes the *full* response into a per-episode hybrid index (BM25 + GEMMA embeddings) *before* truncation — so all items are always retrievable even when only 2 were shown.

### Action Space

The model outputs a single tool call per step.

| Tool                         | Input                             | Output                                                                          |
| ---------------------------- | --------------------------------- | ------------------------------------------------------------------------------- |
| `browser_agent(task, url)`   | Task string + app base URL        | Summary list of API endpoint names + methods (e.g. `GET /products`)             |
| `search_endpoints(query)`    | Natural language query            | Top-3 endpoint schemas (method, path, auth, params with sources, response fields)|
| `curl_exec(command)`         | Full curl command string          | `{status_code, headers, body}` — body smart-truncated; full body indexed        |
| `search_episode_data(query)` | Keyword or natural language query | Top-5 JSON objects from this episode's request/response history                 |
| `done(result?)`              | Optional result string            | Ends episode, triggers judge evaluation                                         |

`browser_agent` is called **exactly once per episode at step 1**. Calling it again applies a −0.3 penalty. During training, it loads a cached HAR file; at inference, it launches a live browser session.

Full technical specifications for all tools: [`TOOLS.md`](./TOOLS.md)

### Reward Space

**Per-step:**

| Signal                       | Value  | Trigger                                                              |
| ---------------------------- | ------ | -------------------------------------------------------------------- |
| Valid API call (2xx)         | +0.2   | `curl_exec` returns 2xx status                                       |
| New path called this episode | +0.1   | Normalized path not called before — discourages looping              |
| Correct parameter sourcing   | +0.25  | Judge: value came from the correct source type                       |
| Session value correctly used | +0.1   | Auth token/cookie present and correct in curl call                   |
| Repeated identical call      | −0.15  | Exact duplicate curl command issued twice                            |
| browser_agent called again   | −0.3   | `browser_agent` called after step 1                                  |
| Malformed curl command       | −0.1   | curl cannot be parsed or executed                                    |
| 4xx response (recoverable)   | −0.05  | Call failed but episode continues                                    |

**Episode end:**

| Outcome                                                     | Reward                                     |
| ----------------------------------------------------------- | ------------------------------------------ |
| Task completed correctly                                    | +2.0 to +5.0 (scales with difficulty tier) |
| Partial completion (right endpoints, wrong param threading) | +0.5 to +1.5                               |
| Authentication correctly obtained (even if task fails)      | +0.3                                       |
| Timeout / task failed entirely                              | −1.5                                       |

Target signal separation: successful episodes `+3` to `+7`, failed episodes `−2` to `−1`. Required for GRPO.

> **Reward design note:** Pure step-level rewards can teach a model to "look busy" — accumulating exploration rewards while never completing the task. The terminal outcome reward is designed to dominate the sum of all per-step rewards. The curriculum is the primary defense: Easy tasks have a trivially short optimal path (2 steps), so there's no room to accumulate fake exploration reward before the model learns that the terminal reward is what matters.

---

## Key Design Decisions

### Browser Agent as a Discovery Tool

The RL agent has access to a **browser agent tool** powered by [`bu-30b-a3b-preview`](https://huggingface.co/browser-use/bu-30b-a3b-preview) — a 30B MoE vision-language model (3B active parameters) served via the [browser-use](https://github.com/browser-use/browser-use) library on Playwright. When called, it completes the task in a real browser while intercepting all network traffic, then returns the filtered API call list.

**Training vs. inference:** The browser agent output is pre-computed and cached per task during training — the RL model receives it instantly, no live browser session runs. At inference, the browser agent runs live to handle novel tasks.

Full details: [`BROWSER_AGENT.md`](BROWSER_AGENT.md)

### Ground Truth from the Codebase, Not the Browser

The browser agent shows *what* API calls happen. It does not explain *why* — where each parameter comes from or what field constraints exist. That comes from a one-time static analysis of each WebArena application's Docker image source, producing a **ground truth API catalog**:

```
endpoint:    POST /rest/V1/guest-carts/{cartId}/items
path_params:
  cartId:    obtained from: POST /rest/V1/guest-carts → response body
body:
  cartItem.sku:       the product's SKU, from: GET /rest/V1/products → items[].sku
  cartItem.qty:       quantity, from: task specification
  cartItem.quote_id:  same as cartId
```

The judge uses this to verify not just *what* the model called, but *where each parameter value came from*. Source types: `TASK_SPEC`, `PREV_CALL`, `AUTH_FLOW`, `STATIC`, `DERIVED`. This is how partial credit works — the model gets reward for correctly threading a `cart_id` even if the final call had a wrong field elsewhere.

Full extraction process: [`GROUND_TRUTH_EXTRACTION.md`](GROUND_TRUTH_EXTRACTION.md)

### HTML and Form-Based Applications

Not every endpoint returns JSON. The Forum (Postmill) relies on HTML form submissions with CSRF tokens; Wikipedia (Kiwix) serves static HTML pages. The agent handles both:

- **CSRF tokens**: The model GETs the form page, reads the `value` attribute of `input[name="_csrf_token"]` from the returned HTML, and places it in the subsequent POST. If the token is beyond the 3,000-character truncation point, it calls `search_episode_data("_csrf_token")` — the full HTML is indexed before truncation.
- **HTML-only responses**: Wikipedia responses are returned in the standard `{status_code, headers, body}` envelope. Search results appear in `<a href>` tags; article content in `<p>` tags.

---

## Example Episode

```
TASK: Add "Radiant Tee" to a guest cart at http://ec2-16-59-2-56.../

STEP 1: browser_agent("Add Radiant Tee to a guest cart", "http://ec2-...:7770/")
→ {"endpoints": ["POST /rest/V1/guest-carts", "GET /rest/V1/products",
                  "POST /rest/V1/guest-carts/{id}/items", ...]}

STEP 2: search_endpoints("create guest cart")
→ ["endpoint: POST /rest/V1/guest-carts | auth: none | returns: string (cartId)"]

STEP 3: curl_exec("curl -X POST 'http://ec2-.../rest/V1/guest-carts' -H 'Content-Type: application/json'")
→ {status_code: 200, body: "cart-abc123"}

STEP 4: search_endpoints("find product by name get sku")
→ ["endpoint: GET /rest/V1/products | query: searchCriteria filters | returns: .items[].sku"]

STEP 5: curl_exec("curl 'http://ec2-.../rest/V1/products?searchCriteria[filter_groups][0][filters][0][field]=name&...[value]=Radiant+Tee'")
→ {status_code: 200, body: {"items":[{"sku":"MH01","name":"Radiant Tee","price":22.0}]}}

STEP 6: search_endpoints("add item to guest cart cartId")
→ ["endpoint: POST /rest/V1/guest-carts/{cartId}/items | path: cartId from POST /rest/V1/guest-carts | body: cartItem.sku, cartItem.qty, cartItem.quote_id"]

STEP 7: curl_exec("curl -X POST 'http://ec2-.../rest/V1/guest-carts/cart-abc123/items' -H 'Content-Type: application/json' -d '{\"cartItem\":{\"sku\":\"MH01\",\"qty\":1,\"quote_id\":\"cart-abc123\"}}'")
→ {status_code: 200, body: {"item_id": 5, "sku": "MH01", "qty": 1}}

STEP 8: done("Radiant Tee added to cart")
```

Values from prior responses (`cart-abc123`, `MH01`) are threaded directly from the growing episode history. If step 5 had returned 200 products truncated to 2, the model would call `search_episode_data("Radiant Tee sku")` to retrieve `MH01` from the episode index.

---

## Setup

### Prerequisites

- Docker installed and running
- Python 3.11+ with [`uv`](https://github.com/astral-sh/uv)
- A Hugging Face token with read access

### Local Development

```bash
# Clone and enter the project
git clone <your-hf-space-url>
cd HARvestGym

# Install dependencies
uv sync

# Validate the OpenEnv spec
openenv validate

# Build and run the Docker image
docker build -t harvgym .
docker run -p 8000:8000 harvgym

# Run the inference script
HF_TOKEN=hf_xxx uv run inference.py
```

### Environment Variables

| Variable       | Default                              | Required | Purpose                                   |
| -------------- | ------------------------------------ | -------- | ----------------------------------------- |
| `HF_TOKEN`     | —                                    | **Yes**  | HuggingFace auth token                    |
| `API_BASE_URL` | `https://router.huggingface.co/v1`   | No       | LLM API endpoint                          |
| `MODEL_NAME`   | `google/gemma-4-31B-it`              | No       | Model for inference                       |
| `HARVGYM_TASK` | `har_classify_easy`                  | No       | Override which task to run                |

### API Endpoints

```bash
# Reset episode
curl -X POST http://localhost:8000/reset

# Execute a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "browser_agent", "args": {"task": "...", "url": "..."}}'

# Get current state
curl http://localhost:8000/state
```

---

## Baseline Performance

Scores generated by running `uv run inference.py` with `google/gemma-4-31B-it` via the HuggingFace Router.

| Task | Difficulty | Score | Steps | Notes |
| ---- | ---------- | ----- | ----- | ----- |
| `easy_list_pants` | Easy | **0.74** | 7 | List products in 'Pants' category |
| `medium_cart_camera_backpack` | Medium | **0.46** | 20 | Add Camera Backpack to guest cart |
| `medium_cart_flannel_jacket` | Medium | **0.52** | 20 | Add Flannel Jacket to guest cart |
| `hard_checkout_ripstop_pants` | Hard | **0.14** | 20 | Full guest checkout (hit step limit) |
| **Overall** | — | **0.47** | — | |

> **To regenerate:** `HF_TOKEN=hf_xxx uv run inference.py`
