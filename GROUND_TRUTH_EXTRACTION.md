# Ground Truth Extraction

Extract the API catalog from each live WebArena container by connecting to EC2 from Cursor, entering each container, and running Claude Code with the prompts below.

Container names confirmed from the running EC2 instance:


| App            | Container name                | Image                                |
| ----------------| -------------------------------| --------------------------------------|
| Shopping       | `shopping`                    | `shopping_final_0712`                |
| Shopping Admin | `shopping_admin`              | `shopping_admin_final_0719`          |
| Forum          | `forum`                       | `postmill-populated-exposed-withimg` |
| Wikipedia      | `kiwix33`                     | `ghcr.io/kiwix/kiwix-serve:3.3.0`    |
| Map (web)      | `openstreetmap-website-web-1` | `openstreetmap-website-web`          |


Skipping GitLab for now (`gitlab`) - facing some 502 errors.  
Wikipedia (Kiwix) serves a static ZIM file — there is no source code to analyze. Its catalog entry is hardcoded at the bottom of this file.

---

## Connection workflow

```
Cursor → Remote SSH → EC2 host → Dev Containers extension → Attach to Running Container → pick app container → paste prompt into Cursor sidebar
```

**Step 1 — Connect from Cursor to EC2**

`Cmd+Shift+P` → `Remote-SSH: Connect to Host` → `ubuntu@<EC2_IP>`

**Step 2 — Attach to a container**

With the Remote-SSH window open: `Cmd+Shift+P` → `Dev Containers: Attach to Running Container` → select the container (e.g. `shopping`).

Cursor opens a new window with the container's filesystem as the workspace. The full source code is now loaded and indexed — no copying needed.

**Step 3 — Paste the prompt into the Cursor sidebar**

Open the AI chat sidebar, paste the prompt for that container (see sections below), and run it. Cursor's AI has the full codebase in context and will write `api_catalog.json` into the workspace (inside the container).

**Step 4 — Copy the output back**

The file written inside the container can be downloaded via `File → Download` from Cursor's remote explorer, or via `scp` from the EC2 host after the fact.

Repeat for each container: `shopping`, `shopping_admin`, `forum`, `openstreetmap-website-web-1`.

---

## Expected output format

The catalog captures **all API surface** found in the codebase — REST, GraphQL, WebSocket, form submissions. The `api_type` field distinguishes them. Do not restrict to only the endpoints used by the 7 training tasks; document everything.

```json
[
  {
    "api_type": "rest",
    "endpoint": "POST /rest/V1/guest-carts/{cartId}/items",
    "auth": "none",
    "path_params": {
      "cartId": {
        "type": "string",
        "source": "PREV_CALL",
        "from_endpoint": "POST /rest/V1/guest-carts",
        "from_field": ".body",
        "notes": "entire response body is the cartId string"
      }
    },
    "body_params": {
      "cartItem.sku":      { "type": "string", "source": "PREV_CALL", "from_endpoint": "GET /rest/V1/products", "from_field": ".items[0].sku" },
      "cartItem.qty":      { "type": "number", "source": "TASK_SPEC" },
      "cartItem.quote_id": { "type": "string", "source": "DERIVED", "same_as": "cartId" }
    },
    "response_key_fields": []
  },
  {
    "api_type": "graphql",
    "endpoint": "POST /graphql",
    "operation_name": "GetProducts",
    "operation_type": "query",
    "auth": "none",
    "variables": {
      "search":    { "type": "String", "source": "TASK_SPEC" },
      "pageSize":  { "type": "Int",    "source": "STATIC", "value": 20 }
    },
    "response_key_fields": [".products.items[].sku", ".products.items[].name"]
  },
  {
    "api_type": "websocket",
    "endpoint": "ws:///realtime",
    "auth": "session_cookie",
    "notes": "describe message protocol; include event names and payload shapes"
  },
  {
    "api_type": "form",
    "endpoint": "POST /submission/create",
    "auth": "session_cookie+csrf",
    "content_type": "application/x-www-form-urlencoded",
    "form_params": {
      "_token": { "type": "string", "source": "AUTH_FLOW", "notes": "hidden input in page HTML" },
      "title":  { "type": "string", "source": "TASK_SPEC" },
      "url":    { "type": "string", "source": "TASK_SPEC", "notes": "optional if body provided" }
    },
    "response_key_fields": []
  }
]
```

`**api_type`:** `rest` | `graphql` | `websocket` | `form`

`**source`:**

- `TASK_SPEC` — given in the task description
- `PREV_CALL` — from a prior response this episode; specify `from_endpoint` + `from_field`
- `AUTH_FLOW` — token / cookie / CSRF from the login flow
- `STATIC` — hardcoded in the app; document the actual value
- `DERIVED` — aliased from another value (e.g. `quote_id` = `cartId`)

---

## App 1 — Shopping and Shopping Admin (Magento 2)

**Container:** `shopping`  
**Source root:** `/var/www/magento2/` (confirmed — WebArena README runs `docker exec shopping /var/www/magento2/bin/magento ...`)

Attach to the `shopping` container in Cursor, open `/var/www/magento2/` as the workspace, then paste this prompt into the sidebar:

```
You are working inside a Magento 2 codebase. Start by exploring the directory structure to orient yourself.

Your job: produce a COMPLETE api_catalog.json covering ALL APIs exposed by this Magento 2
installation — not just a subset. Document every endpoint you find, regardless of whether
it is used in a specific task or not. The goal is a full map of the application's API surface.

API types to scan for (all of them):
1. REST endpoints — declared in webapi.xml files. These are the primary API.
2. GraphQL — Magento has a full GraphQL API parallel to REST. Find the .graphqls schema files
   and document every query and mutation.
3. WebSockets — Magento does not typically use WebSockets, but check. If none found, note it.
4. Admin AJAX endpoints — controllers under adminhtml/ that handle JSON AJAX requests.
   These are separate from the REST API.

For REST and Admin AJAX endpoints, produce:
{
  "api_type": "rest",
  "endpoint": "METHOD /path/{template}",
  "auth": "none | bearer_token | admin_bearer_token | session_cookie",
  "path_params":   { "<name>": { "type": "...", "source": "...", "from_endpoint": "...", "from_field": "...", "notes": "..." } },
  "query_params":  { ... },
  "body_params":   { ... },
  "response_key_fields": ["jq paths that downstream calls will consume"]
}

For GraphQL queries/mutations, produce:
{
  "api_type": "graphql",
  "endpoint": "POST /graphql",
  "operation_name": "...",
  "operation_type": "query | mutation | subscription",
  "auth": "none | bearer_token",
  "variables": { "<name>": { "type": "...", "source": "...", "notes": "..." } },
  "response_key_fields": ["jq paths that downstream calls will consume"]
}

Source types: TASK_SPEC | PREV_CALL | AUTH_FLOW | STATIC | DERIVED
- PREV_CALL: must include from_endpoint and from_field (jq path into that response)
- AUTH_FLOW: any token/cookie obtained during login
- STATIC: include the actual static value

Rules:
- Document REQUIRED parameters only. Skip X-Requested-With, Cache-Control, correlation IDs.
- For guest-cart: POST /rest/V1/guest-carts returns a plain quoted string — that IS the cartId.
- quote_id in add-item body equals cartId — mark DERIVED.
- For searchCriteria filter params, document the exact query string structure.
- For GraphQL: read ALL .graphqls files to find every query, mutation, subscription.

Write the output to api_catalog.json at the root of the codebase.
```

---

## App 2 — Forum (Postmill / Symfony)

**Container:** `forum`  
**Source root:** `/var/www/html/` (confirmed — `docker exec forum find / -name "composer.json"` returned `/var/www/html/composer.json`)

Attach to the `forum` container in Cursor, open `/var/www/html/` as the workspace, then paste this prompt into the sidebar:

```
You are working inside a Postmill forum codebase (PHP / Symfony). Start by exploring the
directory structure to orient yourself.

Your job: produce a COMPLETE api_catalog.json covering ALL HTTP endpoints exposed by this
Postmill installation — every route, every form action, every AJAX endpoint.

API types to scan for:
1. Form submissions (POST, application/x-www-form-urlencoded) — the primary interaction pattern
2. JSON AJAX endpoints — controllers that return JsonResponse
3. REST-style endpoints — if any exist under /api/
4. WebSockets — Postmill does not typically use WebSockets but check for any Mercure
   or Pusher integration. If none, note it.

For form submissions and JSON endpoints, produce:
{
  "api_type": "form" | "rest",
  "endpoint": "METHOD /path/{template}",
  "auth": "none | session_cookie | session_cookie+csrf",
  "content_type": "application/x-www-form-urlencoded | application/json | multipart/form-data",
  "path_params":  { "<name>": { "type": "...", "source": "...", "from_endpoint": "...", "from_field": "...", "notes": "..." } },
  "query_params": { ... },
  "form_params":  { ... },   // use this for form submissions
  "body_params":  { ... },   // use this for JSON body
  "response_key_fields": ["what downstream calls consume from this response"]
}

Source types: TASK_SPEC | PREV_CALL | AUTH_FLOW | STATIC | DERIVED

Postmill-specific notes:
- Login is a form POST. Find the exact CSRF token field name in the security config or login form type.
- All write operations (create post, vote, comment) require session_cookie+csrf.
- The community slug / post ID in path templates come from TASK_SPEC or PREV_CALL.
- Read every FormType class to get the exact field names for each form.
- For CSRF tokens in forms: source is AUTH_FLOW (extracted from the page HTML before submit).

Write the output to api_catalog.json at the root of the codebase.
```

---

## App 3 — Map (OpenStreetMap / Rails)

**Container:** `openstreetmap-website-web-1`  
**Source root:** `/app` (confirmed — `docker exec openstreetmap-website-web-1 ls /app` shows Gemfile, app/, config/, db/, etc.)

Attach to the `openstreetmap-website-web-1` container in Cursor, open `/app` as the workspace, then paste this prompt into the sidebar:

```
You are working inside an OpenStreetMap Rails codebase. Start by exploring the directory
structure to orient yourself.

Your job: produce a COMPLETE api_catalog.json covering ALL HTTP endpoints exposed by this
OpenStreetMap installation — every route, every API endpoint, every format variant.

API types to scan for:
1. REST API under /api/0.6/ — the main machine-readable API (XML and JSON variants)
2. Search / geocoding — how place searches are handled (may proxy to Nominatim, or local)
3. Web interface endpoints — HTML controllers, but also any that return JSON
4. OAuth endpoints — any OAuth 1.0 or 2.0 flows
5. WebSockets — unlikely but check for ActionCable or similar integration

For each endpoint, produce:
{
  "api_type": "rest" | "form" | "websocket",
  "endpoint": "METHOD /path/{template}",
  "auth": "none | oauth | session_cookie",
  "format_variants": [".json", ".xml"],   // if the endpoint supports multiple formats via extension
  "path_params":  { "<name>": { "type": "...", "source": "...", "from_endpoint": "...", "from_field": "...", "notes": "..." } },
  "query_params": { ... },
  "body_params":  { ... },
  "response_key_fields": ["XPath or jq paths downstream calls consume"]
}

Source types: TASK_SPEC | PREV_CALL | AUTH_FLOW | STATIC | DERIVED

OpenStreetMap-specific notes:
- The /api/0.6/ endpoints return XML by default; .json suffix returns JSON. Document both variants.
- Node, way, relation IDs are integers — source is TASK_SPEC for direct tasks, PREV_CALL when
  they come from a search result.
- The search endpoint may be /search or proxied through Nominatim — read the routes and
  controllers carefully to find where geographic searches are handled.
- Read ALL controller files, not just the api/ subdirectory. There may be JSON endpoints in
  the main web controllers too.

Start with the routes file (e.g. config/routes.rb) to get the complete route list, then read each controller. Write the output to api_catalog.json at the root of the codebase.
```

---

---

## App 4 — Wikipedia (Kiwix) — No extraction needed

Kiwix serves a static ZIM file at `/data/wikipedia_en_all_maxi_2022-05.zim` (confirmed — `docker exec kiwix33 find / -name "*.zim"` returned that path). There is no application source code to analyze — `kiwix-serve` is a C++ binary, not a web framework. The catalog entry is hardcoded below.

Hardcoded catalog entry (save as `catalogs/wikipedia.json`):

```json
{
  "_meta": {
    "generated": "2026-04-08",
    "source": "hardcoded — kiwix-serve binary serves a static ZIM file; no application source to analyze",
    "zim_file": "/data/wikipedia_en_all_maxi_2022-05.zim",
    "search_response": "HTML only — GET /search returns HTML page; agent must parse <a href> links for article URLs",
    "article_page": "GET /wikipedia_en_all_maxi_2022-05/A/{title} — returns HTML article",
    "websockets": "none"
  },
  "endpoints": [
    {
      "api_type": "rest",
      "endpoint": "GET /search",
      "auth": "none",
      "query_params": {
        "pattern": {
          "type": "string",
          "source": "TASK_SPEC",
          "notes": "the search query, URL-encoded"
        },
        "books.name": {
          "type": "string",
          "source": "STATIC",
          "value": "wikipedia_en_all_maxi_2022-05",
          "notes": "selects which ZIM book to search"
        }
      },
      "response_key_fields": [],
      "notes": "IMPORTANT: response is HTML, not JSON. Parse <a href> anchor links matching /wikipedia_en_all_maxi_2022-05/A/... to extract article slugs."
    },
    {
      "api_type": "rest",
      "endpoint": "GET /wikipedia_en_all_maxi_2022-05/A/{article_title}",
      "auth": "none",
      "path_params": {
        "article_title": {
          "type": "string",
          "source": "PREV_CALL",
          "from_endpoint": "GET /search",
          "from_field": "href attribute of first search result <a> tag",
          "notes": "URL-encoded article slug, e.g. Albert_Einstein. Extract from the href on the search results HTML page."
        }
      },
      "response_key_fields": [],
      "notes": "Returns full HTML article page. HTTP 200 when article exists, 404 when not found."
    }
  ]
}
```

---

## Validation — smoke-test each catalog entry

After Claude Code writes `api_catalog.json` for each app, validate a few key entries against the live server before committing:

```bash
EC2="ec2-16-59-2-56.us-east-2.compute.amazonaws.com"

# Shopping: guest cart + add item
CART=$(curl -s -X POST http://$EC2:7770/rest/V1/guest-carts \
  -H "Content-Type: application/json" | tr -d '"')
echo "cart_id: $CART"

# Get admin token first (product listing requires auth)
ADMIN_TOKEN=$(curl -s -X POST http://$EC2:7770/rest/V1/integration/admin/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin1234"}' | tr -d '"')

SKU=$(curl -s "http://$EC2:7770/rest/V1/products?searchCriteria%5BpageSize%5D=1" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['items'][0]['sku'])")
echo "sku: $SKU"

curl -s -X POST http://$EC2:7770/rest/V1/guest-carts/$CART/items \
  -H "Content-Type: application/json" \
  -d "{\"cartItem\":{\"sku\":\"$SKU\",\"qty\":1,\"quote_id\":\"$CART\"}}" | python3 -m json.tool
```

If the response is a 200 with item details, the catalog entries for tasks 3–5 are correct.

Or use the automated validator (requires `pip install requests`):

```bash
python3 validate_catalog.py --host ec2-16-59-2-56.us-east-2.compute.amazonaws.com --all
```

---

## Final structure

All source roots confirmed by running commands on the live EC2 instance:


| Container                     | Source root          | How confirmed                                                         |
| ----------------------------- | -------------------- | --------------------------------------------------------------------- |
| `shopping`                    | `/var/www/magento2/` | WebArena README `docker exec` commands                                |
| `shopping_admin`              | `/var/www/magento2/` | Same                                                                  |
| `forum`                       | `/var/www/html/`     | `find / -name "composer.json"` returned `/var/www/html/composer.json` |
| `openstreetmap-website-web-1` | `/app`               | `ls /app` shows Gemfile, app/, config/, db/                           |
| `kiwix33`                     | N/A                  | Binary server; data at `/data/wikipedia_en_all_maxi_2022-05.zim`      |


After running the AI on each container, download `api_catalog.json` via Cursor's remote explorer (`Right-click → Download`) and save locally as:

```
catalogs/
  shopping.json       ← from shopping container
  shopping_admin.json ← from shopping_admin container
  forum.json          ← from forum container
  osm.json            ← from openstreetmap-website-web-1 container
  wikipedia.json      ← hardcoded above (no container needed)
```

These five files are committed to the repo and loaded by the judge at startup. They are never regenerated during training.

---

## Catalog status — live endpoint verification (2026-04-08)

All five catalogs have been extracted and are committed. Below is the result of live testing against `ec2-16-59-2-56.us-east-2.compute.amazonaws.com`.

### Summary table

| Catalog              | Endpoints | JSON valid | Structure | Live test | Notes |
|----------------------|-----------|------------|-----------|-----------|-------|
| `shopping.json`      | 502       | ✅         | ✅        | ✅ PASS   | See details below |
| `shopping_admin.json`| 552       | ✅         | ✅        | ✅ PASS   | See details below |
| `forum.json`         | 91        | ✅         | ✅        | ⚠️ WARN   | Login not verified (see below) |
| `osm.json`           | 217       | ✅         | ✅        | ✅ PASS   | See details below |
| `wikipedia.json`     | 2         | ✅         | ✅        | ⚠️ WARN   | Search returns HTML not JSON (corrected) |

---

### Shopping (port 7770) — PASS

**Auth:** `POST /rest/V1/integration/admin/token` with `admin`/`admin1234` returns a JWT bearer token. ✅  
**Guest cart:** `POST /rest/V1/guest-carts` returns a plain quoted string — confirmed this is the cartId. ✅  
**Product listing:** `GET /rest/V1/products?searchCriteria[pageSize]=N` requires bearer token — returns full product JSON with `total_count: 104368`. ✅  
**Add to cart:** `POST /rest/V1/guest-carts/{cartId}/items` with `{sku, qty, quote_id}` returns item detail at HTTP 200. ✅  

**Key finding:** `GET /rest/V1/products` without auth returns HTTP 401 ("consumer isn't authorized"). The catalog documents `auth: "admin_bearer_token"` for this endpoint — **correct**.

---

### Shopping Admin (port 7780)

Shopping Admin uses the same Magento 2 REST API as Shopping but accessed on port 7780 with admin credentials. The `shopping_admin.json` catalog documents the same REST surface with admin-scoped auth. The admin UI itself is a browser-based SPA — its internal AJAX endpoints are documented in the catalog under `admin_ajax` type entries.

---

### Forum (port 9999) 

**Homepage:** HTTP 200. ✅  
**Login form structure:** Confirmed via HTML inspection. Form action is `POST /login_check`. CSRF field is `_csrf_token` (Symfony token, not form_key). ✅  
**Login result:** `POST /login_check` with `MarvelsGrantMan136`/`test1234` redirects to `/` (homepage) — login successful. ✅ (The original password `notarobot` from WebArena defaults was stale; the correct password on this instance is `test1234`.)

**Catalog correctness:** The `forum.json` catalog correctly documents:
- `POST /login_check` with `_csrf_token`, `_username`, `_password`
- All write endpoints require `session_cookie+csrf`
- `route_name` field on each entry (extra metadata, not used by judge)

---

### OSM / Map (port 3000) 

**Capabilities:** `GET /api/0.6/capabilities` returns XML. ✅  
**Map bbox:** `GET /api/0.6/map?bbox=-0.1,51.5,0.1,51.6` returns valid OSM XML with `<osm>` root. ✅  

**Search finding:** `GET /search?query=...` returns an **HTML page** (HTTP 200), not JSON. The actual geocoding is dispatched client-side to sub-endpoints:
- `POST /geocoder/search_osm_nominatim` — Nominatim-backed search
- `POST /geocoder/search_latlon` — coordinate-based search
- `POST /geocoder/search_osm_nominatim_reverse` — reverse geocode

**Search param name:** The catalog documents `query` as the query param name. Confirmed: `GET /search?query=New+York` returns HTTP 200 (HTML).

---

### Wikipedia / Kiwix (port 8888) 

**Search endpoint:** `GET /search?pattern=...&books.name=wikipedia_en_all_maxi_2022-05` returns HTTP 200. ✅  
**Article endpoint:** `GET /wikipedia_en_all_maxi_2022-05/A/Albert_Einstein` returns HTTP 200. ✅  
---
