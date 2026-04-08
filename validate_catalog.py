#!/usr/bin/env python3
"""
Smoke-tests an extracted API catalog against live WebArena endpoints.

Run locally pointing at your EC2 instance:
  python validate_catalog.py --host <EC2_IP> --app shopping --catalog catalogs/shopping.json
  python validate_catalog.py --host <EC2_IP> --all

Run directly on EC2:
  python validate_catalog.py --host localhost --all

Requirements: pip install requests
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

try:
    import requests
    requests.packages.urllib3.disable_warnings()
except ImportError:
    print("pip install requests")
    sys.exit(1)

# ── Port map ──────────────────────────────────────────────────────────────────
APP_PORTS = {
    "shopping":       7770,
    "shopping_admin": 7780,
    "forum":          9999,
    "wikipedia":      8888,
    "map":            3000,
}

# ── Test credentials (fill in before running) ─────────────────────────────────
# These are the known WebArena default credentials. Update if your instance differs.
CREDENTIALS = {
    "shopping": {
        "login_endpoint": "POST /rest/V1/integration/customer/token",
        "body": {"username": "emma.lopez@gmail.com", "password": "Password.1"},
        "token_path": None,         # entire response body is the token string
        "header": "Authorization",
        "header_prefix": "Bearer ",
    },
    "shopping_admin": {
        "login_endpoint": "POST /rest/V1/integration/admin/token",
        "body": {"username": "admin", "password": "admin1234"},
        "token_path": None,
        "header": "Authorization",
        "header_prefix": "Bearer ",
    },
    "forum": {
        "login_endpoint": "POST /login_check",
        "form": {"_username": "MarvelsGrantMan136", "_password": "test1234"},
        "uses_session": True,
    },
}

# ── Minimal test values for path/query params ─────────────────────────────────
# When an endpoint needs a param that comes from a prior call (PREV_CALL) or task,
# we substitute a safe dummy value just to see if the endpoint exists and responds.
DUMMY_VALUES = {
    "string":  "test",
    "integer": "1",
    "number":  "1",
    "boolean": "true",
    "int":     "1",
    "Int":     "1",
    "String":  "test",
}

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
GREY   = "\033[90m"

def col(text, colour): return f"{colour}{text}{RESET}"

# ── Validator ─────────────────────────────────────────────────────────────────
class Validator:
    def __init__(self, host: str, app: str, catalog: list[dict], timeout: int = 10):
        self.host    = host
        self.app     = app
        self.port    = APP_PORTS[app]
        self.base    = f"http://{host}:{self.port}"
        self.catalog = catalog
        self.timeout = timeout
        self.session = requests.Session()
        self.auth_headers: dict[str, str] = {}
        self.results: list[dict] = []

    # ── Auth ──────────────────────────────────────────────────────────────────
    def authenticate(self):
        creds = CREDENTIALS.get(self.app)
        if not creds:
            return
        parts   = creds["login_endpoint"].split(" ", 1)
        method  = parts[0].upper()
        path    = parts[1]
        url     = self.base + path
        try:
            if creds.get("uses_session"):
                resp = self.session.post(url, data=creds.get("form", {}),
                                         timeout=self.timeout, allow_redirects=True)
                if resp.status_code in (200, 302):
                    print(col(f"  [auth] forum session cookie set", GREEN))
                else:
                    print(col(f"  [auth] forum login returned {resp.status_code}", YELLOW))
            else:
                resp = requests.request(method, url, json=creds.get("body"),
                                         timeout=self.timeout)
                if resp.status_code == 200:
                    token = resp.json() if creds["token_path"] else resp.text.strip().strip('"')
                    prefix = creds.get("header_prefix", "Bearer ")
                    self.auth_headers[creds["header"]] = prefix + token
                    print(col(f"  [auth] got token for {self.app}", GREEN))
                else:
                    print(col(f"  [auth] login failed ({resp.status_code}) — authenticated endpoints will be skipped", YELLOW))
        except Exception as e:
            print(col(f"  [auth] {e}", RED))

    # ── Single endpoint test ──────────────────────────────────────────────────
    def test_endpoint(self, entry: dict) -> dict:
        api_type = entry.get("api_type", "rest")
        endpoint = entry.get("endpoint", "")

        # WebSocket — not testable with HTTP, skip
        if api_type == "websocket":
            return {"endpoint": endpoint, "api_type": api_type, "status": "SKIP",
                    "notes": "WebSocket — skipped (requires WS client)"}

        # GraphQL — send introspection or a minimal operation
        if api_type == "graphql":
            return self._test_graphql(entry)

        # REST or form
        parts  = endpoint.split(" ", 1)
        if len(parts) != 2:
            return {"endpoint": endpoint, "api_type": api_type, "status": "SKIP", "notes": "cannot parse endpoint"}
        method = parts[0].upper()
        path   = parts[1]

        # Substitute path params with dummies
        path = self._fill_path(path, entry.get("path_params", {}))

        # Build query string
        query  = self._build_query(entry.get("query_params", {}))
        url    = self.base + path + ("?" + urlencode(query) if query else "")

        # Auth
        auth_type = entry.get("auth", "none")
        if "bearer_token" in auth_type or "admin_bearer_token" in auth_type:
            if not self.auth_headers:
                return {"endpoint": endpoint, "api_type": api_type, "status": "SKIP",
                        "notes": "auth required but login failed/not configured"}
            headers = {"Content-Type": "application/json", **self.auth_headers}
        elif "csrf" in auth_type:
            if not self.session.cookies:
                return {"endpoint": endpoint, "api_type": api_type, "status": "SKIP",
                        "notes": "session+csrf required but login not configured"}
            headers = {}
        elif "session_cookie" in auth_type:
            headers = {}
        else:
            headers = {"Content-Type": "application/json"}

        # Body
        body = None
        body_params = entry.get("body_params") or entry.get("form_params")
        if method in ("POST", "PUT", "PATCH") and body_params:
            if entry.get("content_type", "").startswith("application/x-www"):
                body = {k: DUMMY_VALUES.get(v.get("type", "string"), "test")
                        for k, v in body_params.items() if v.get("source") == "TASK_SPEC"}
            else:
                body = {k: DUMMY_VALUES.get(v.get("type", "string"), "test")
                        for k, v in body_params.items() if v.get("source") == "TASK_SPEC"}

        # Fire request
        try:
            start = time.time()
            if "csrf" in auth_type or "session_cookie" in auth_type:
                resp = self.session.request(method, url, headers=headers,
                                             json=body if api_type == "rest" else None,
                                             data=body if api_type == "form" else None,
                                             timeout=self.timeout, allow_redirects=False)
            else:
                resp = requests.request(method, url, headers=headers,
                                         json=body if api_type == "rest" else None,
                                         data=body if api_type == "form" else None,
                                         timeout=self.timeout, allow_redirects=False)
            elapsed = int((time.time() - start) * 1000)
            status_code = resp.status_code

            # Decide pass/fail
            # 200-299 = clear pass; 400/422 = endpoint exists but our dummy params are wrong (expected)
            # 401/403 = auth issue; 404 = endpoint not found; 500 = server error
            if 200 <= status_code < 300:
                outcome = "PASS"
            elif status_code in (400, 405, 422):
                outcome = "WARN"   # endpoint reached, validation error from dummy params — expected
            elif status_code in (401, 403):
                outcome = "AUTH"   # exists but needs auth we didn't provide
            elif status_code == 404:
                outcome = "FAIL"   # not found
            elif status_code in (301, 302):
                outcome = "REDIR"  # redirect — endpoint exists
            else:
                outcome = f"HTTP{status_code}"

            snippet = resp.text[:120].replace("\n", " ") if resp.text else ""
            return {"endpoint": endpoint, "api_type": api_type, "status": outcome,
                    "http_code": status_code, "ms": elapsed, "snippet": snippet}
        except requests.exceptions.ConnectionError:
            return {"endpoint": endpoint, "api_type": api_type, "status": "CONN_ERR",
                    "notes": "could not connect — is the app running?"}
        except requests.exceptions.Timeout:
            return {"endpoint": endpoint, "api_type": api_type, "status": "TIMEOUT"}
        except Exception as e:
            return {"endpoint": endpoint, "api_type": api_type, "status": "ERROR", "notes": str(e)}

    def _test_graphql(self, entry: dict) -> dict:
        url = self.base + "/graphql"
        op  = entry.get("operation_name", "")
        op_type = entry.get("operation_type", "query")
        # Send a minimal introspection or operation
        if op_type == "query":
            payload = {"query": f"{{ __typename }}"}   # simplest valid query
        else:
            payload = {"query": f"mutation {{ __typename }}"}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            snippet = resp.text[:120].replace("\n", " ")
            outcome = "PASS" if resp.status_code == 200 else f"HTTP{resp.status_code}"
            return {"endpoint": f"POST /graphql ({op})", "api_type": "graphql",
                    "status": outcome, "http_code": resp.status_code, "snippet": snippet}
        except Exception as e:
            return {"endpoint": f"POST /graphql ({op})", "api_type": "graphql",
                    "status": "ERROR", "notes": str(e)}

    def _fill_path(self, path: str, path_params: dict) -> str:
        for name in path_params:
            param_type = path_params[name].get("type", "string")
            dummy = DUMMY_VALUES.get(param_type, "test")
            path = path.replace("{" + name + "}", dummy)
        return path

    def _build_query(self, query_params: dict) -> dict:
        out = {}
        for name, meta in query_params.items():
            if meta.get("source") in ("TASK_SPEC", "STATIC"):
                val = meta.get("value") or DUMMY_VALUES.get(meta.get("type", "string"), "test")
                out[name] = val
        return out

    # ── Run all ───────────────────────────────────────────────────────────────
    def run(self):
        print(f"\n{'─'*60}")
        print(f"  {self.app.upper()} → {self.base}  ({len(self.catalog)} endpoints)")
        print(f"{'─'*60}")

        self.authenticate()

        for entry in self.catalog:
            result = self.test_endpoint(entry)
            self.results.append(result)

        self._print_results()
        return self.results

    def _print_results(self):
        print()
        w = max(len(r["endpoint"]) for r in self.results) + 2
        header = f"  {'ENDPOINT':<{w}}  {'TYPE':<10}  {'STATUS':<10}  {'CODE':<6}  {'ms':<6}  SNIPPET"
        print(header)
        print("  " + "─" * (len(header) - 2))

        counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "SKIP": 0, "other": 0}
        for r in self.results:
            s = r["status"]
            code = str(r.get("http_code", ""))
            ms   = str(r.get("ms", ""))
            snip = r.get("snippet", r.get("notes", ""))[:60]

            if s == "PASS":
                colour, counts["PASS"] = GREEN, counts["PASS"] + 1
            elif s in ("WARN", "REDIR", "AUTH"):
                colour, counts["WARN"] = YELLOW, counts["WARN"] + 1
            elif s in ("FAIL", "CONN_ERR", "TIMEOUT", "ERROR") or s.startswith("HTTP"):
                colour, counts["FAIL"] = RED, counts["FAIL"] + 1
            else:
                colour, counts["SKIP"] = GREY, counts["SKIP"] + 1

            line = f"  {r['endpoint']:<{w}}  {r['api_type']:<10}  {col(s, colour):<20}  {code:<6}  {ms:<6}  {col(snip, GREY)}"
            print(line)

        total = len(self.results)
        print()
        print(f"  Results: {col(str(counts['PASS']), GREEN)} pass  "
              f"{col(str(counts['WARN']), YELLOW)} warn  "
              f"{col(str(counts['FAIL']), RED)} fail  "
              f"{col(str(counts['SKIP']), GREY)} skip  "
              f"/ {total} total")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────
def load_catalog(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"catalog not found: {path}")
        sys.exit(1)
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return data
    # some catalogs are wrapped in an object
    for v in data.values():
        if isinstance(v, list):
            return v
    return []


def main():
    parser = argparse.ArgumentParser(description="Smoke-test an API catalog against live WebArena apps")
    parser.add_argument("--host",    required=True, help="EC2 IP or hostname")
    parser.add_argument("--app",     help="single app: shopping|shopping_admin|forum|wikipedia|map")
    parser.add_argument("--catalog", help="path to api_catalog.json (required if --app is set)")
    parser.add_argument("--all",     action="store_true",
                        help="run all apps; looks for catalogs/shopping.json etc.")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    if args.all:
        catalog_dir = Path("catalogs")
        apps = {
            "shopping":       catalog_dir / "shopping.json",
            "shopping_admin": catalog_dir / "shopping_admin.json",
            "forum":          catalog_dir / "forum.json",
            "wikipedia":      catalog_dir / "wikipedia.json",
            "map":            catalog_dir / "osm.json",
        }
        all_results = {}
        for app, cat_path in apps.items():
            if cat_path.exists():
                catalog = load_catalog(str(cat_path))
                v = Validator(args.host, app, catalog, args.timeout)
                all_results[app] = v.run()
            else:
                print(col(f"  {app}: catalog not found at {cat_path} — skipping", YELLOW))

        # Summary
        print(f"\n{'═'*60}")
        print("  SUMMARY")
        print(f"{'═'*60}")
        for app, results in all_results.items():
            total  = len(results)
            passed = sum(1 for r in results if r["status"] == "PASS")
            warned = sum(1 for r in results if r["status"] in ("WARN", "REDIR", "AUTH"))
            failed = sum(1 for r in results if r["status"] in ("FAIL", "CONN_ERR", "TIMEOUT", "ERROR") or r["status"].startswith("HTTP"))
            print(f"  {app:<20} {col(str(passed), GREEN)}/{total} pass  "
                  f"{col(str(warned), YELLOW)} warn  {col(str(failed), RED)} fail")
        print()

    elif args.app:
        if not args.catalog:
            parser.error("--catalog is required when using --app")
        catalog = load_catalog(args.catalog)
        v = Validator(args.host, args.app, catalog, args.timeout)
        v.run()

    else:
        parser.error("use --app + --catalog, or --all")


if __name__ == "__main__":
    main()
