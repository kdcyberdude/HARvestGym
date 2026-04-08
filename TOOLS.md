# HARvestGym Tool Specification

Technical specification for all tools available to the RL agent. Each tool is a Python function called by the environment on behalf of the model. The model outputs a single tool call per step; the environment executes it and returns the result.

---

## Tool Set Summary


| Tool                         | Input                             | What It Does                                                                                                                                                                                                                                                                                                                                                                                                 | Output                                                                                                                                                                                                                                      |
| ---------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `browser_agent(task, url)`   | Task string + app base URL        | Checks if a pre-recorded HAR file exists for this app; if so, processes it(only for training; at inference time it will always use browser-agent) — otherwise launches a live browser agent to perform the task and record network traffic. Either way, extracts an OpenAPI-like spec from the captured traffic, builds GEMMA embeddings for the search endpoint index, and returns a summary endpoint list. | Deduplicated list of API endpoint names with HTTP methods (e.g. `GET /products`, `POST /guest-carts`) — summary only, no headers/body/schemas. Use `search_endpoints()` with a natural-language query to get full details for any endpoint. |
| `search_endpoints(query)`    | Natural language query            | Semantic search over the endpoint embeddings built by `browser_agent`. Matches the query against the GEMMA-embedded OpenAPI-like spec and returns the top-3 endpoint schemas with full parameter details.                                                                                                                                                                                                    | Top-3 endpoint schemas (method, path, auth, params with sources, response fields)                                                                                                                                                           |
| `curl_exec(command)`         | Full curl command string          | Parses the curl command, executes it via subprocess against the live EC2 server, indexes the full response body into the episode's hybrid BM25 + GEMMA store (before truncation), then returns a truncated observation.                                                                                                                                                                                      | `{status_code, headers, body}` — body smart-truncated; full body indexed into episode store for `search_episode_data()`                                                                                                                     |
| `search_episode_data(query)` | Natural language or keyword query | Hybrid BM25 + GEMMA semantic search across all request/response bodies accumulated during this episode from prior `curl_exec` calls. BM25 handles exact keyword matches (IDs, SKUs); GEMMA handles semantic paraphrases. Finds specific values from truncated or prior responses.                                                                                                                            | Top-5 JSON objects from this episode's request/response history, each annotated with step number and source endpoint                                                                                                                        |
| `done(result?)`              | Optional result string            | Signals the model believes the task is complete. Triggers the judge to evaluate the episode against the ground truth catalog.                                                                                                                                                                                                                                                                                | Ends the episode                                                                                                                                                                                                                            |


`browser_agent` is always called **once at the start of an episode** (step 1). It gives the model an API landscape map for the target application, so the model knows which endpoints exist before it begins probing. All subsequent discovery and execution uses the other tools. **If the model calls `browser_agent` again after step 1, it receives a −0.3 penalty reward** — the call still executes normally (loads HAR if it exists, or runs live browser), the penalty is just applied to the reward signal.

---

## Tool 0: `browser_agent(task, url)`

### Purpose

Give the model an initial map of the API surface for the target application at the start of every episode. The browser agent is a multi-stage pipeline that:

1. Obtains HAR data (from pre-recorded file if available, or by launching a live browser)
2. Processes it to extract an OpenAPI-like spec
3. Builds GEMMA embeddings so `search_endpoints()` can work
4. Returns a **summary-only** endpoint list to the RL agent — just names and methods, no schemas

The output is intentionally sparse. Because there could be many endpoints that will waste context window. The agent sees *what* endpoints exist but not *how* to call them. It must call `search_endpoints()` to get full parameter details for any endpoint.

### Interface

```python
def browser_agent(task: str, url: str) -> dict:
    """
    Multi-stage pipeline:
    1. Check for pre-recorded HAR file → load if exists, else launch live browser
    2. Filter HAR → extract OpenAPI-like spec (methods, paths, params, bodies)
    3. Build GEMMA embeddings over the spec → stored for search_endpoints()
    4. Return summary endpoint list (names + methods only)

    Returns: {
        "app": str,                      # resolved app name (shopping, forum, osm, wikipedia)
        "endpoints": list[dict],         # summary: [{method, path}] — no schemas, no headers
        "total_endpoints": int,          # count of deduplicated endpoints
        "note": str                      # directs agent to use search_endpoints() for details
    }
    """
```

### Stage 1 — HAR Data Source

The browser agent first checks if a pre-recorded HAR file exists. If it does, the browser is never launched — saving 30–120s per episode.

```
hars/
  shopping.har       # all shopping tasks, all API calls recorded for all task templates
  shopping_admin.har # all admin tasks
  forum.har          # all forum tasks
  osm.har            # all OSM tasks
  wikipedia.har      # all Wikipedia tasks
```

```python
HAR_MAP = {
    ":7770": "hars/shopping.har",
    ":7780": "hars/shopping_admin.har",
    ":9999": "hars/forum.har",
    ":3000": "hars/osm.har",
    ":8888": "hars/wikipedia.har",
}

def get_har_data(task: str, url: str) -> dict:
    har_path = resolve_har_path(url)       # port-based lookup from HAR_MAP
    if har_path and os.path.exists(har_path):
        # HAR exists — load from disk, no browser needed
        with open(har_path) as f:
            return json.load(f)
    else:
        # No HAR — launch live browser, perform task, capture traffic
        raw_log = await run_browser_agent_live(task, url, "browser-use/bu-30b-a3b-preview")
        return convert_raw_log_to_har(raw_log)
```

If no HAR exists, the browser agent launches Chromium via Playwright, connects the `bu-30b-a3b-preview` LLM, performs the task while intercepting all network traffic, and produces a HAR-format output. See `BROWSER_AGENT.md` for the live browser implementation.

### Stage 2 — Filter and Extract OpenAPI-like Spec

The HAR data (from either source) is processed to extract a structured spec:

```python
def extract_openapi_spec(har_data: dict, app_base_url: str) -> list[dict]:
    entries = har_data["log"]["entries"]
    seen = set()
    spec_entries = []

    for entry in entries:
        req = entry["request"]
        resp = entry["response"]
        raw_url = req["url"]
        method = req["method"]

        # 1. Skip static assets (images, fonts, CSS, JS bundles, favicon)
        if _is_static_asset(raw_url):
            continue

        # 2. Skip page navigation (HTML document loads)
        content_type = _get_response_content_type(resp)
        if "text/html" in content_type and method == "GET":
            continue

        # 3. Normalise path: replace concrete IDs with {id} placeholders
        path = _normalise_path(urlparse(raw_url).path)

        # 4. Deduplicate by (method, normalised_path)
        key = f"{method} {path}"
        if key in seen:
            continue
        seen.add(key)

        # 5. Extract auth, body, query params for the spec document
        has_auth = any(
            h["name"].lower() in ("authorization", "x-api-key", "cookie")
            for h in req["headers"]
        )

        spec_entries.append({
            "method": method,
            "path": path,
            "query_params": urlparse(raw_url).query or None,
            "request_body": _extract_body(req),
            "status_code": resp["status"],
            "response_content_type": content_type,
            "response_body_sample": _truncate_body(resp),
            "auth_observed": has_auth,
        })

    return spec_entries
```

### Stage 3 — Build GEMMA Embeddings

The spec entries are embedded using `google/embeddinggemma-300m` (GEMMA). These embeddings are stored in the environment and power `search_endpoints()`.

```python
def build_endpoint_embeddings(spec_entries: list[dict], app_name: str):
    model = SentenceTransformer("google/embeddinggemma-300m", token=os.environ.get("HF_TOKEN"))
    chunks = [spec_entry_to_text(e, app_name) for e in spec_entries]
    embeddings = model.encode_document(chunks, batch_size=32)
    return embeddings, chunks  # stored in env for search_endpoints()
```

### Stage 4 — Return Summary

The RL agent receives **only endpoint names and methods** — no schemas, no headers, no body details:

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

### Path Normalisation

The `_normalise_path` function replaces concrete dynamic segments with `{id}` placeholders so that duplicates collapse:

- Numeric IDs: `/products/42` → `/products/{id}`
- UUIDs: `/carts/3fa85f64-5717-4562-b3fc` → `/carts/{id}`
- Magento cart IDs (mixed alphanumeric, 32+ chars): detected by length and character set
- OSM node/way/relation IDs: `/api/0.6/node/12345678` → `/api/0.6/node/{id}`
- Forum post slugs: `/f/general/1-hello-world` → `/f/{slug}/{id}-{slug}`

Normalisation is pattern-based (regex), not AI-generated. No external calls.

### When to Call

`browser_agent` is called **exactly once per episode, at step 1**, before any other tool. It serves as the API landscape orientation AND builds the search index. **If called again after step 1, the call executes normally but the model receives a −0.3 penalty reward.** The model should not need to call it again mid-episode.

### Relationship to Other Tools

```
browser_agent  →  "what endpoints exist?" (summary only)
    │                    + builds GEMMA embeddings internally
    │
    ▼
search_endpoints  →  "give me full schema for endpoint X"
    │                    (searches the GEMMA embeddings built above)
    ▼
curl_exec         →  "call endpoint X, get live response"
    │                    (indexes full response into BM25 episode store)
    ▼
search_episode_data  →  "find specific value from a prior response"
                         (BM25 search over indexed episode data)
```

`browser_agent` provides breadth (what exists) and builds the search index. `search_endpoints` provides depth (how to call it). `curl_exec` provides live data and feeds the episode index. `search_episode_data` retrieves specific values from that index.

---

## Tool 1: `search_endpoints(query)`

### Purpose

Find which API endpoint to call for a given subtask. The model calls this when it does not yet know the correct URL, method, or parameter schema for the next HTTP call it needs to make.

### Interface

```python
def search_endpoints(query: str) -> list[str]:
    """
    Semantic search over the endpoint embeddings built by browser_agent.
    Returns the top-3 matching endpoint schemas as formatted text strings.
    """
```

### Underlying Index

- **Source:** The GEMMA embeddings built by `browser_agent` during Stage 4. These embeddings are created from the OpenAPI-like spec extracted from HAR data — the actual network traffic observed when the browser agent performed tasks on the application.
- **Embedding model:** `google/embeddinggemma-300m` via `sentence-transformers`
- **Built:** By `browser_agent` at the start of each episode (Stage 4). The browser agent processes the HAR data, extracts the OpenAPI-like spec, converts each endpoint to a text chunk, and embeds them using GEMMA.
- **At runtime:** Stored in environment memory after `browser_agent` completes. Available for the rest of the episode. Discarded at episode end (rebuilt from HAR at next episode start).
- **Query embedding:** Uses the `encode_query` method with prompt `task: search result | query: {query}`.
- **Document embedding:** Uses the `encode_document` method with prompt `title: {endpoint} | text: {full_schema_text}`.
- **Similarity:** Use the similarity function specified by the `google/embeddinggemma-300m` model card. The model's `sentence_transformers_config.json` specifies the correct metric (typically cosine similarity for normalized embeddings). Pure numpy, no FAISS needed at this scale.

### Document Structure (one per extracted endpoint)

Each endpoint from the browser agent's OpenAPI-like spec is converted to a searchable text chunk by `spec_entry_to_text()`:

```
app: shopping | endpoint: POST /rest/V1/guest-carts/{id}/items | status: 200 | auth: none | body: {"cartItem":{"sku":"MH01","qty":1,"quote_id":"cart-abc123"}} | response_sample: {"item_id":5,"sku":"MH01","qty":1}
```

The text chunks include method, path, status code, auth observation, query params, request body sample, and response body sample — all extracted from the actual HAR traffic. This is richer than just endpoint names (which is what the RL agent sees from `browser_agent`'s summary output) but less structured than a hand-written catalog.

### Output Format

Returns a list of 3 strings, each being the full text of one matching endpoint schema. The model reads these directly and extracts the method, URL pattern, observed parameters, and response structure.

### When to Call

- At the start of a task subtask: "I need to authenticate — what endpoint handles login?"
- When discovering a prerequisite: "I need a cart ID first — what creates a cart?"
- When unsure of the exact URL pattern: "Is it `/products/{id}` or `/products?id=`?"

### Caveats

- Returns observed traffic patterns, not formal API documentation. The schemas reflect what was seen in the HAR, not what the API formally supports. Some optional parameters may be missing if the browser agent's session didn't exercise them.
- Returns schemas, not live values. The model still needs `curl_exec` to get actual data (product SKUs, cart IDs, etc.).
- If no relevant endpoint exists in the index, returns the closest matches by cosine similarity. The model should treat low-confidence results skeptically and try `curl_exec` to probe.
- The index covers only the current application (determined by `browser_agent`'s URL). Each episode's index is specific to one app.

---

## Tool 2: `curl_exec(command)`

### Purpose

Execute an HTTP request against the live EC2 application and return the response. This is the primary action tool — it is how the model actually interacts with the application.

### Interface

```python
def curl_exec(command: str) -> dict:
    """
    Parses a curl command string, executes it via subprocess against the live EC2 server,
    indexes the full response into the episode store, then returns a truncated observation.

    Returns: {
        "status_code": int,
        "headers": dict,          # response headers
        "body": str | dict        # truncated; see truncation rules below
    }
    """
```

### Execution Pipeline

The environment performs these steps in order on every `curl_exec` call:

```
1. Parse the curl command string
      Extract: method, URL, headers, body
      Validate: URL host must match app_base_url (reject requests to external hosts)
      Inject: session cookies from session_state into headers automatically

2. Execute via subprocess
      subprocess.run(["curl", ...parsed args...], timeout=10)
      Capture: status_code, response headers, response body (full, untruncated)

3. Index into episode store (BEFORE truncation)
      Index the request body (if any)
      Index the response body
      See: Episode Store section below

4. Truncate the response body for context
      Apply truncation rules (see below)
      Add truncation note if any array was trimmed

5. Return to model
      {status_code, headers, truncated_body} or error
```

### Truncation Rules

Applied in order. First matching rule wins.

**Rule 1 — Non-JSON body:**

HTML from form-serving pages (login, post creation, etc.) is kept longer than a byte cutoff would allow because CSRF tokens and `<input>` fields are embedded inside the markup. The model locates them by reading the raw HTML string — no HTML parser required since tokens appear as predictable plain-text patterns (`<input type="hidden" name="_csrf_token" value="…">`). Even with 3,000 characters, if the CSRF token appears after the cutoff (possible in large pages), the full body is indexed in the episode store and can be retrieved with `search_episode_data("_csrf_token")`.

```python
NONJSON_MAX_CHARS = 3000

if not is_valid_json(body):
    return body[:NONJSON_MAX_CHARS] + (" [truncated — non-JSON response]" if len(body) > NONJSON_MAX_CHARS else "")
```

**Rule 2 — JSON primitive (string, number, boolean, null):**

```python
if isinstance(parsed, (str, int, float, bool)) or parsed is None:
    return body  # never truncate; these are tokens, IDs, simple confirmations
```

**Rule 3 — Error response (4xx or 5xx):**

```python
if status_code >= 400:
    return body  # never truncate error messages; the model needs every word to self-correct
```

**Rule 4 — JSON object or array with no large arrays:**

```python
# "large" means an array with >= 3 objects (dicts)
if no array field contains >= 3 dict items:
    return body  # small enough; return as-is
```

**Rule 5 — JSON with large array field(s):**

```python
# For each top-level field whose value is a list of >= 3 dicts:
#   Keep first 2 items, drop the rest
#   Add a _list_truncated annotation at the top level

truncated = {k: v for k, v in parsed.items() if not is_large_list(v)}
for key, val in parsed.items():
    if is_large_list(val):
        truncated[key] = val[:2]
        truncated["_list_truncated"] = {
            "field": key,
            "shown": 2,
            "total": len(val),
            "note": f"Showing 2 of {len(val)} items. Use search_episode_data() to find a specific item from this response."
        }
return json.dumps(truncated)
```

The note is a static Python format string. It is not AI-generated. It does not suggest specific query parameters or URL patterns.

### Session State Injection

Before executing the curl command, the environment reads `session_state` and injects any relevant cookies or tokens:

- If `session_state` contains `PHPSESSID`, inject as `Cookie: PHPSESSID=...`
- If `session_state` contains `form_key` (Magento CSRF), inject as a header: `X-Form-Key: ...`
- If `session_state` contains `PHPSESSID` for Forum requests (port 9999), inject as `Cookie: PHPSESSID=...`
- If `session_state` contains a bearer token, inject as `Authorization: Bearer ...` only if the model's curl command does not already include an `Authorization` header

**CSRF note for Postmill (Forum):** Postmill's `_csrf_token` is a request-body field, not a header. The environment does **not** auto-inject it — the model must extract it from the HTML form response and include it explicitly in the POST body. The `session_state` cookie (`PHPSESSID`) is auto-injected so the server associates the CSRF token with the active session. The expected workflow:

```
GET /login → HTML body contains <input type="hidden" name="_csrf_token" value="XYZ">
Model reads "XYZ" from body string
POST /login -d '_csrf_token=XYZ&_username=user&_password=pass'
Environment auto-injects Cookie: PHPSESSID from session_state
```

The model is responsible for setting the correct `Content-Type` in its curl command. The model declares intent (which headers to include); the environment fills in the actual token values from `session_state`.

### Caveats

- `curl_exec` always hits the live EC2 server. No responses are mocked.
- The timeout is 10 seconds. If the server does not respond, returns `{status_code: 0, error: "timeout"}`.
- URL must be on the same host as `app_base_url`. Cross-host requests are rejected with `{status_code: 0, error: "host_not_allowed"}`.
- The model must include the full URL including host and port. Relative paths are not supported.

---

## Tool 3: `search_episode_data(query)`

### Purpose

Search through all request and response bodies accumulated during the current episode. The model calls this when it needs a specific value (an ID, a name, a token) that was returned in a prior API response but the list was truncated.

This tool exists because **not every data type has a search or filter API endpoint**. For applications where the model cannot make a targeted filtered query (e.g., a listing endpoint that only supports pagination, not field-based filtering)

### Interface

```python
def search_episode_data(query: str) -> list[str]:
    """
    Keyword + BM25 search over all request and response bodies indexed during this episode.
    Returns the top-5 matching JSON objects as formatted text strings, each annotated
    with the step number and endpoint that produced them.
    """
```

### Hybrid Search: BM25 + GEMMA Semantic Embeddings

The episode index uses a **hybrid approach** combining BM25 keyword matching with GEMMA semantic embeddings (`google/embeddinggemma-300m`). Both indexes are maintained in parallel — BM25 for fast exact keyword recall, GEMMA for semantic understanding when the agent's query uses different terminology than what appears in the response data.

**Why hybrid, not BM25 alone:**

BM25 excels at exact keyword matches ("MH01", "cart-abc123", "Radiant Tee") but fails on paraphrases. If the agent queries "price of the tee shirt I found earlier", BM25 won't match "Radiant Tee" because the terms don't overlap. GEMMA semantic embeddings bridge this gap — "tee shirt" and "Radiant Tee" are semantically close in embedding space.

**Why hybrid, not GEMMA alone:**

GEMMA embeddings are weaker at exact string matching. Searching for a specific cart ID like "cart-abc123" benefits from BM25's precise token matching. The hybrid approach gets the best of both.

**Scoring:** Results are ranked by a weighted combination of BM25 score (normalized) and GEMMA cosine similarity:

```python
hybrid_score = alpha * bm25_score_normalized + (1 - alpha) * GEMMA_cosine_similarity
# alpha = 0.4 (tunable; favors semantic slightly over keyword)
```

**Performance:** GEMMA is 300M parameters. On GPU, embedding a batch of 200 response items takes ~1-2 seconds — acceptable overhead per `curl_exec` call. The GEMMA model is already loaded in memory for `search_endpoints`, so no additional model loading cost. BM25 remains instantaneous.

**Fallback:** If no GPU is available, the system falls back to BM25-only mode. The GEMMA model is shared with `search_endpoints` — if it's loaded, episode data search uses it too.

### Episode Index — Document Construction

Every time `curl_exec` completes, the environment constructs embedding documents from the full (pre-truncation) request and response bodies and adds them to the in-memory BM25 index for the current episode.

**Algorithm:**

```python
def build_index_documents(step: int, method: str, path: str,
                           request_body: Any, response_body: Any,
                           status_code: int) -> list[str]:
    docs = []

    # 1. Index the request body (if any)
    if request_body is not None:
        docs.append(
            f"step:{step} source:request endpoint:{method} {path} "
            f"body:{json.dumps(request_body, ensure_ascii=False)}"
        )

    # 2. Index the response body
    if response_body is None or not is_valid_json(response_body):
        docs.append(
            f"step:{step} source:response endpoint:{method} {path} "
            f"status:{status_code} body:{str(response_body)[:500]}"
        )
        return docs

    parsed = json.loads(response_body) if isinstance(response_body, str) else response_body

    # 3. JSON primitive — one document
    if isinstance(parsed, (str, int, float, bool)) or parsed is None:
        docs.append(
            f"step:{step} source:response endpoint:{method} {path} "
            f"status:{status_code} value:{parsed}"
        )
        return docs

    # 4. JSON object — find top-level array fields
    array_fields = {k: v for k, v in parsed.items()
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)}
    scalar_fields = {k: v for k, v in parsed.items() if k not in array_fields}

    if not array_fields:
        # No arrays — one document for the whole object
        docs.append(
            f"step:{step} source:response endpoint:{method} {path} "
            f"status:{status_code} data:{json.dumps(parsed, ensure_ascii=False)}"
        )
        return docs

    # 5. Has array fields — one document per array item, with parent context attached
    parent_context = (
        f"step:{step} source:response endpoint:{method} {path} status:{status_code} "
        + " ".join(f"{k}:{v}" for k, v in scalar_fields.items())
    )
    for field_name, items in array_fields.items():
        for item in items:
            # Flatten nested arrays within each item to strings (do not recurse further)
            flat_item = {}
            for k, v in item.items():
                flat_item[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
            docs.append(
                f"{parent_context} list_field:{field_name} "
                f"item:{json.dumps(flat_item, ensure_ascii=False)}"
            )

    return docs
```

**Key design principle:** The parent context (step number, endpoint, HTTP status, scalar fields like `total_count`) is prepended to every child item document. When the model searches for "Radiant Tee product SKU", the returned document contains both `name:Radiant Tee sku:MH01` and the context `endpoint:GET /rest/V1/products step:2` — the model knows where this value came from and which step it appeared in.

### Episode Index — Lifecycle

```
episode start  →  BM25 index initialized (empty)
                  GEMMA embedding store initialized (empty)
                        │
each curl_exec  →  build_index_documents() called
                   documents appended to BM25 corpus (BM25 index rebuilt, fast)
                   documents embedded via GEMMA and appended to embedding store
                        │
search_episode_data()  →  BM25 scores computed (keyword match)
                          GEMMA cosine similarity computed (semantic match)
                          hybrid ranking: alpha * BM25 + (1-alpha) * GEMMA
                          top-5 documents returned
                        │
episode end    →  both indexes discarded entirely
next episode   →  fresh indexes from scratch
```

### Output Format

Returns a list of up to 5 strings, each being one indexed document. Example:

```
[
  "step:2 source:response endpoint:GET /rest/V1/products status:200 total_count:200 list_field:items item:{\"sku\": \"MH01\", \"name\": \"Radiant Tee\", \"price\": 22.0, \"type_id\": \"simple\"}",
  "step:2 source:response endpoint:GET /rest/V1/products status:200 total_count:200 list_field:items item:{\"sku\": \"MH03\", \"name\": \"Radiant Tee Long Sleeve\", \"price\": 28.0, \"type_id\": \"simple\"}"
]
```

The model reads `sku: MH01` from the first result and uses it in the next curl call.

### When to Call

- A prior curl response was truncated (`_list_truncated` present in the response) and the model needs a specific item not shown in the 2-item sample.
- The model needs a value from a prior step but cannot easily locate it by scanning history (many steps ago, or buried in a complex response).
- There is no filter/search API for the data type (practical assumption: not all applications expose filtered listing endpoints for every resource).

### Caveats

- Only searches data from **the current episode**. Values from prior episodes are not accessible (each episode starts with an empty index).
- Only finds data that was actually returned by a `curl_exec` call in this episode. If the relevant API has not been called yet, the data is not indexed.
- The hybrid search handles both exact keywords ("MH01", "cart-abc123") and semantic paraphrases ("tee shirt price"). However, using the same terminology seen in the response still produces the best results.
- For large lists (200+ items), all items are indexed. BM25 search is fast regardless of index size. GEMMA embedding of large responses adds 1-2 seconds of overhead per `curl_exec` call.

---

## Tool 4: `done(result?)`

### Interface

```python
def done(result: str = None) -> None:
    """
    Signals that the model believes the task is complete.
    Triggers the judge to evaluate the episode against the ground truth catalog.
    Ends the episode.
    """
```

### Behavior

- Calling `done()` immediately ends the episode. No further tool calls are processed.
- The optional `result` string is logged but does not affect the reward. The judge evaluates the live application state, not the model's self-report.
- If the model calls `done()` and the task is not actually complete, the episode ends with `−1.5` reward (timeout/failure outcome). The model should only call `done()` after the final `curl_exec` has returned a 2xx confirming the required state change.

### How the Model Learns When to Call `done()`

There is **no explicit "task complete" signal** from the environment. The model learns when to call `done()` purely through the reward signal over many episodes:

- **Calling `done()` too early** (before the task is actually complete) → judge finds the expected state change is missing → `−1.5` reward. The model learns to avoid this.
- **Calling `done()` after a successful final API call** (e.g., add-to-cart returns 2xx with `item_id`) → judge confirms the state change → `+2.0` to `+5.0` reward. The model learns that a 2xx response confirming the desired action is the right signal to call `done()`.
- **Never calling `done()`** (running out of steps) → episode times out → `−1.5` reward. The model learns it must eventually commit.

The learned pattern is: after the final state-changing `curl_exec` returns a 2xx response whose body confirms the expected outcome (e.g., `item_id` present in add-to-cart, `order_id` present in checkout), call `done()`. This mirrors how a human developer knows an API call succeeded — you check the response.

**Optional verification step:** Before calling `done()`, the model can issue one more `curl_exec` to verify the state change (e.g., `GET /rest/V1/guest-carts/{id}` to confirm the item is in the cart). This costs one step but reduces the risk of premature `done()` calls. The model learns whether verification is worth the step cost through reward optimization.

---

## Episode Index — Full Example

**Task:** `"Add 'Radiant Tee' to a guest cart at http://ec2-.../"`

```
SYSTEM: ...

TASK: Add "Radiant Tee" to a guest cart at http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/

[session_state: {}]

STEP 1 ACTION: browser_agent("Add Radiant Tee to a guest cart", "http://ec2-...:7770/")
STEP 1 RESULT: {
  "app": "shopping",
  "endpoints": [
    {"method": "POST", "path": "/rest/V1/guest-carts",             "status": 200, "auth_observed": false},
    {"method": "GET",  "path": "/rest/V1/products",                "status": 200, "auth_observed": false},
    {"method": "GET",  "path": "/rest/V1/products/{id}",           "status": 200, "auth_observed": false},
    {"method": "POST", "path": "/rest/V1/guest-carts/{id}/items",  "status": 200, "auth_observed": false},
    ...
  ],
  "note": "These endpoints were observed in pre-recorded browser sessions..."
}

STEP 2 ACTION: search_endpoints("find product by name, get sku")
STEP 2 RESULT: [
  "endpoint: GET /rest/V1/products | auth: none | query: searchCriteria filters (field, value, condition_type) | returns: .items[].sku, .items[].name, .items[].price",
  ...
]

STEP 3 ACTION: curl_exec("curl 'http://ec2-.../rest/V1/products?searchCriteria[filter_groups][0][filters][0][field]=name&searchCriteria[filter_groups][0][filters][0][value]=Radiant+Tee'")
STEP 3 RESULT: {"status_code": 200, "body": {"items": [{"sku": "MH01", "name": "Radiant Tee", "price": 22.0}], "total_count": 1}}

STEP 4 ACTION: search_endpoints("create guest cart, get cart id")
STEP 4 RESULT: [
  "endpoint: POST /rest/V1/guest-carts | auth: none | no body | returns: cartId string (plain quoted string)",
  ...
]

STEP 5 ACTION: curl_exec("curl -X POST 'http://ec2-.../rest/V1/guest-carts' -H 'Content-Type: application/json'")
STEP 5 RESULT: {"status_code": 200, "body": "cart-abc123"}

STEP 6 ACTION: search_endpoints("add item to guest cart")
STEP 6 RESULT: [
  "endpoint: POST /rest/V1/guest-carts/{cartId}/items | auth: none | path: cartId (from POST /rest/V1/guest-carts) | body: cartItem.sku, cartItem.qty, cartItem.quote_id (same as cartId) | returns: item_id",
  ...
]

STEP 7 ACTION: curl_exec("curl -X POST 'http://ec2-.../rest/V1/guest-carts/cart-abc123/items' -H 'Content-Type: application/json' -d '{\"cartItem\":{\"sku\":\"MH01\",\"qty\":1,\"quote_id\":\"cart-abc123\"}}'")
STEP 7 RESULT: {"status_code": 200, "body": {"item_id": 5, "sku": "MH01", "qty": 1}}

STEP 8 ACTION: done("Radiant Tee (MH01) added to guest cart cart-abc123, item_id 5")
```

### Embedding build (by browser_agent, once per episode)

The GEMMA embeddings for `search_endpoints` are built by `browser_agent` during Stage 4, not pre-built offline. Each episode starts fresh:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# google/embeddinggemma-300m requires accepting Google's license on HuggingFace.
# Set HF_TOKEN env variable to a token that has accepted the license.
# Uses encode_query / encode_document / similarity API from sentence-transformers.
# NOTE: activations do not support float16 — use float32 or bfloat16.
HF_TOKEN = os.environ.get("HF_TOKEN")  # must have accepted the license

def build_endpoint_embeddings(spec_entries: list[dict], app_name: str):
    """Called by browser_agent Stage 4 after extracting OpenAPI-like spec from HAR."""
    model = SentenceTransformer("google/embeddinggemma-300m", token=HF_TOKEN)
    chunks = [spec_entry_to_text(e, app_name) for e in spec_entries]
    # encode_document uses: "title: {endpoint} | text: {rest of chunk}"
    embeddings = model.encode_document(chunks, batch_size=32)
    # embeddings are returned normalized; dot product = cosine similarity
    return embeddings, chunks  # stored in env memory for search_endpoints()
```

### Runtime search

```python
def search_endpoints(query: str, embeddings, texts, model, top_k=3) -> list[str]:
    q_emb = model.encode_query(query)          # shape: (D,)
    # Use similarity metric specified by google/embeddinggemma-300m model card
    scores = model.similarity(q_emb, embeddings).squeeze(0)  # shape: (N,)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [texts[i] for i in top_idx]
```

### Index size per episode

Typical endpoint counts per app (after HAR filtering and deduplication):

- Shopping (Magento REST): ~8–15 endpoints per HAR
- Shopping Admin (Magento Admin AJAX + REST): ~5–10 endpoints per HAR
- Forum (Postmill forms + REST): ~3–8 endpoints per HAR
- OSM (Rails API + web): ~5–10 endpoints per HAR
- Wikipedia (Kiwix): ~2 endpoints per HAR

**Typical: ~5–15 endpoints × 768 dims × 4 bytes = negligible memory.** Embedding time on GPU: <1 second per episode.

---

## Truncation Helper — Python Pseudocode

```python
import json

TRUNCATE_LIST_AT = 2       # keep this many items from large arrays
LARGE_ARRAY_THRESHOLD = 3  # arrays with >= this many dicts are "large"
NONJSON_MAX_CHARS = 3000   # 3 000 chars — enough to capture hidden CSRF inputs in most HTML forms

def truncate_response_body(body: str, status_code: int) -> str:
    # Rule 3: never truncate errors
    if status_code >= 400:
        return body

    # Rule 1: non-JSON
    if not _is_json(body):
        if len(body) > NONJSON_MAX_CHARS:
            return body[:NONJSON_MAX_CHARS] + " [truncated — non-JSON response]"
        return body

    parsed = json.loads(body)

    # Rule 2: primitive
    if not isinstance(parsed, (dict, list)):
        return body

    # Rule 4/5: find large array fields
    if isinstance(parsed, list):
        if len(parsed) >= LARGE_ARRAY_THRESHOLD and isinstance(parsed[0], dict):
            result = parsed[:TRUNCATE_LIST_AT]
            note = {"_list_truncated": {
                "shown": TRUNCATE_LIST_AT,
                "total": len(parsed),
                "note": f"Showing {TRUNCATE_LIST_AT} of {len(parsed)} items. "
                        "Use search_episode_data() to find a specific item from this response."
            }}
            return json.dumps(result + [note])
        return body

    # parsed is a dict — check each value
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
            f"List fields truncated: "
            + ", ".join(f"{k} showing {TRUNCATE_LIST_AT}/{n}" for k, n in total_truncated.items())
            + ". Use search_episode_data() to find a specific item from this response."
        )
    }
    return json.dumps(result)


def _is_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except (ValueError, TypeError):
        return False
```

---

## Tool Call Format in the Episode Prompt

The growing episode context uses this format for tool calls and results:

```
SYSTEM: You are an API agent. Your task is to complete the given task by calling the
available tools: browser_agent, search_endpoints, curl_exec, search_episode_data, done.
Complete the task using only HTTP calls to the application at the given URL.
When a response body is HTML, read hidden input fields directly from the markup to
extract CSRF tokens (pattern: <input type="hidden" name="_csrf_token" value="...">).
For form submissions, use Content-Type: application/x-www-form-urlencoded.

TASK: Add "Radiant Tee" to a guest cart at http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/

[session_state: {}]

STEP 1 ACTION: browser_agent("Add Radiant Tee to a guest cart", "http://ec2-...:7770/")
STEP 1 RESULT: {
  "app": "shopping",
  "endpoints": [
    "POST /rest/V1/guest-carts",
    "GET  /rest/V1/products",
    "GET  /rest/V1/products/{sku}",
    "POST /rest/V1/guest-carts/{id}/items",
    ...
  ],
  "note": "Use search_endpoints() to get full schema for any of these."
}

STEP 2 ACTION: search_endpoints("find product by name, get sku")
STEP 2 RESULT: [
  "endpoint: GET /rest/V1/products | auth: none | query: searchCriteria filters (field, value, condition_type) | returns: .items[].sku, .items[].name, .items[].price",
  ...
]

STEP 3 ACTION: curl_exec("curl 'http://ec2-.../rest/V1/products?searchCriteria[filter_groups][0][filters][0][field]=name&searchCriteria[filter_groups][0][filters][0][value]=Radiant+Tee'")
STEP 3 RESULT: {"status_code": 200, "body": {"items": [{"sku": "MH01", "name": "Radiant Tee", "price": 22.0}], "total_count": 1}}

STEP 4 ACTION: search_endpoints("create guest cart, get cart id")
STEP 4 RESULT: [
  "endpoint: POST /rest/V1/guest-carts | auth: none | no body | returns: cartId string (plain quoted string)",
  ...
]

STEP 5 ACTION: curl_exec("curl -X POST 'http://ec2-.../rest/V1/guest-carts' -H 'Content-Type: application/json'")
STEP 5 RESULT: {"status_code": 200, "body": "cart-abc123"}

STEP 6 ACTION: search_endpoints("add item to guest cart")
STEP 6 RESULT: [
  "endpoint: POST /rest/V1/guest-carts/{cartId}/items | auth: none | path: cartId (from POST /rest/V1/guest-carts) | body: cartItem.sku, cartItem.qty, cartItem.quote_id (same as cartId) | returns: item_id",
  ...
]

STEP 7 ACTION: curl_exec("curl -X POST 'http://ec2-.../rest/V1/guest-carts/cart-abc123/items' -H 'Content-Type: application/json' -d '{\"cartItem\":{\"sku\":\"MH01\",\"qty\":1,\"quote_id\":\"cart-abc123\"}}'")
STEP 7 RESULT: {"status_code": 200, "body": {"item_id": 5, "sku": "MH01", "qty": 1}}

STEP 8 ACTION: done("Radiant Tee (MH01) added to guest cart cart-abc123, item_id 5")
```

Value threading happens entirely through the multi-turn context. The model reads `"MH01"` from step 2's result and `"cart-abc123"` from step 4's result directly — no explicit store/retrieve tools needed.

---

