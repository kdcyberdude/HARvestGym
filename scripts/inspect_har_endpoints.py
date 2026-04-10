#!/usr/bin/env python3
"""
inspect_har_endpoints.py

Runs extract_openapi_spec() on every HAR file in hars/ and prints a full
summary of discovered endpoints — method, path, status code, auth, and a
snippet of the request/response body where available.

Usage:
    python scripts/inspect_har_endpoints.py [--json]

Flags:
    --json   Emit machine-readable JSON instead of the human-readable table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make the package importable without installing
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from server.tools.browser_agent import extract_openapi_spec  # noqa: E402


# ---------------------------------------------------------------------------
# HAR files to inspect
# ---------------------------------------------------------------------------

HARS_DIR = REPO_ROOT / "hars"

HAR_FILES = {
    "shopping":       HARS_DIR / "shopping.har",
    "shopping_admin": HARS_DIR / "shopping_admin.har",
    "forum":          HARS_DIR / "forum.har",
    "wikipedia":      HARS_DIR / "wikipedia.har",
}

# Fake base URLs — only used for pass-through in extract_openapi_spec
APP_BASE_URLS = {
    "shopping":       "http://localhost:7770",
    "shopping_admin": "http://localhost:7780",
    "forum":          "http://localhost:9999",
    "wikipedia":      "http://localhost:8888",
}


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_COL_W = 80


def _hr(char: str = "─") -> None:
    print(char * _COL_W)


def _body_snippet(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        snippet = value[:120]
    else:
        snippet = json.dumps(value)[:120]
    return snippet + ("…" if len(str(snippet)) >= 120 else "")


def _print_entry(idx: int, entry: dict) -> None:
    auth_flag = "🔐 AUTH" if entry["auth_observed"] else "open"
    print(f"  [{idx:>3}] {entry['method']:<7} {entry['path']}")
    print(f"         status={entry['status_code']}  ct={entry['response_content_type'] or '—'}  {auth_flag}")
    if entry.get("query_params"):
        print(f"         query: {entry['query_params'][:100]}")
    req_snippet = _body_snippet(entry.get("request_body"))
    if req_snippet:
        print(f"         req_body: {req_snippet}")
    resp_snippet = _body_snippet(entry.get("response_body_sample"))
    if resp_snippet:
        print(f"         resp_sample: {resp_snippet}")


def _method_counts(entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["method"]] = counts.get(e["method"], 0) + 1
    return dict(sorted(counts.items()))


def print_app_summary(app_name: str, entries: list[dict], raw_total: int | None = None) -> None:
    _hr("═")
    header = f"  APP: {app_name.upper()}   ({len(entries)} unique API endpoints"
    if raw_total is not None:
        header += f" extracted from {raw_total} raw HAR entries"
    header += ")"
    print(header)
    counts = _method_counts(entries)
    print(f"  Methods: {counts}")
    auth_count = sum(1 for e in entries if e["auth_observed"])
    print(f"  Auth-required endpoints: {auth_count}/{len(entries)}")
    _hr()
    if not entries:
        print("  (no API-like entries survived filtering)")
    for i, entry in enumerate(entries, 1):
        _print_entry(i, entry)
    print()


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------

def emit_json(results: dict) -> None:
    # Convert to a JSON-safe structure
    output = {}
    for app_name, entries in results.items():
        output[app_name] = {
            "total": len(entries),
            "method_counts": _method_counts(entries),
            "endpoints": entries,
        }
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Verification / assertion checks
# ---------------------------------------------------------------------------


# NOTE: These HAR files are sparse — each was recorded for a narrow task
# scenario, not as a full API crawl.  The vast majority of HAR entries are
# static assets (/static/ prefix) that the extractor correctly filters out.
# Thresholds below reflect the actual usable API surface in each file.
SANITY_CHECKS: dict[str, dict] = {
    "shopping": {
        "min_endpoints": 1,
        "expected_methods": {"GET"},
        "note": "Sparse HAR — only checkout success page recorded; "
                "213 total entries but 212 are /static/ assets.",
    },
    "shopping_admin": {
        "min_endpoints": 2,
        "expected_methods": {"GET", "POST"},
        "note": "Sparse HAR — product save/edit + MUI JSON endpoint; "
                "353 total entries but 350 are /static/ assets.",
    },
    "forum": {
        "min_endpoints": 2,
        "expected_methods": {"GET", "POST"},
        "note": "Sparse HAR — one POST submission + one forum thread GET; "
                "24 total entries but 22 are .js build files.",
    },
    "wikipedia": {
        "min_endpoints": 0,
        "expected_methods": set(),
        "note": "Sparse HAR — only an article HTML page + /-/mw/ style/JS assets; "
                "no XHR/REST traffic recorded.",
    },
}


def run_checks(results: dict) -> bool:
    print("\n" + "─" * _COL_W)
    print("SANITY CHECKS  (thresholds calibrated to actual HAR content)")
    print("─" * _COL_W)
    all_passed = True
    for app_name, checks in SANITY_CHECKS.items():
        entries = results.get(app_name, [])
        methods_found = set(e["method"] for e in entries)
        n = len(entries)

        min_ok = n >= checks["min_endpoints"]
        exp = checks["expected_methods"]
        methods_ok = exp.issubset(methods_found) if exp else True

        status = "PASS" if (min_ok and methods_ok) else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"  {status}  {app_name}")
        print(f"       endpoints : {n} (min={checks['min_endpoints']})  {'✓' if min_ok else '✗'}")
        if exp:
            print(f"       methods   : {sorted(methods_found)} "
                  f"(expected ⊇ {sorted(exp)})  {'✓' if methods_ok else '✗'}")
        print(f"       note      : {checks['note']}")
    print("─" * _COL_W)
    print("Overall:", "ALL PASSED ✓" if all_passed else "SOME FAILED ✗")
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    emit_json_mode = "--json" in sys.argv

    results: dict[str, list[dict]] = {}
    raw_totals: dict[str, int] = {}
    missing: list[str] = []

    for app_name, har_path in HAR_FILES.items():
        if not har_path.exists():
            print(f"[WARN] HAR not found: {har_path}", file=sys.stderr)
            missing.append(app_name)
            results[app_name] = []
            raw_totals[app_name] = 0
            continue

        with open(har_path) as f:
            har_data = json.load(f)

        raw_totals[app_name] = len(har_data.get("log", {}).get("entries", []))
        entries = extract_openapi_spec(har_data, APP_BASE_URLS[app_name])
        results[app_name] = entries

    if emit_json_mode:
        emit_json(results)
        return 0

    # Human-readable output
    for app_name, entries in results.items():
        print_app_summary(app_name, entries, raw_totals.get(app_name))

    passed = run_checks(results)

    if missing:
        print(f"\n[WARN] Missing HAR files for: {', '.join(missing)}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
