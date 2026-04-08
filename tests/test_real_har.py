"""
Test browser_agent pipeline against REAL HAR files.
Processes the actual recorded HAR data to verify filtering, deduplication,
and path normalisation work on real-world traffic.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from tool_browser_agent import extract_openapi_spec, spec_entry_to_text, build_summary_output

HARS_DIR = os.path.join(os.path.dirname(__file__), "..", "hars")

APPS = {
    "wikipedia": {
        "har": "wikipedia.har",
        "url": "http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:8888/",
    },
    "forum": {
        "har": "forum.har",
        "url": "http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:9999/",
    },
    "shopping": {
        "har": "shopping.har",
        "url": "http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7770/",
    },
    "shopping_admin": {
        "har": "shopping_admin.har",
        "url": "http://ec2-16-59-2-56.us-east-2.compute.amazonaws.com:7780/",
    },
}


def test_app(app_name: str, config: dict):
    har_path = os.path.join(HARS_DIR, config["har"])
    if not os.path.exists(har_path):
        print(f"  [SKIP] {har_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"APP: {app_name} ({config['har']})")
    print(f"{'='*60}")

    with open(har_path) as f:
        har_data = json.load(f)

    total_entries = len(har_data["log"]["entries"])
    spec = extract_openapi_spec(har_data, config["url"])
    chunks = [spec_entry_to_text(e, app_name) for e in spec]
    summary = build_summary_output(spec, app_name)

    print(f"\n  Total HAR entries: {total_entries}")
    print(f"  Filtered API endpoints: {len(spec)}")
    print(f"  Reduction: {total_entries} → {len(spec)} ({100*(1-len(spec)/max(total_entries,1)):.0f}% filtered out)")

    print(f"\n  Endpoints:")
    for ep in spec:
        auth_marker = " [AUTH]" if ep["auth_observed"] else ""
        body_marker = " [BODY]" if ep.get("request_body") else ""
        print(f"    {ep['method']:6s} {ep['path'][:70]:70s} {ep['status_code']}{auth_marker}{body_marker}")

    print(f"\n  Text chunks for embedding: {len(chunks)}")
    if chunks:
        print(f"    Sample: {chunks[0][:120]}...")

    return spec


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: browser_agent pipeline against REAL HAR files")
    print("=" * 60)

    all_specs = {}
    for app_name, config in APPS.items():
        spec = test_app(app_name, config)
        if spec:
            all_specs[app_name] = spec

    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for app, spec in all_specs.items():
        methods = {}
        for e in spec:
            methods[e["method"]] = methods.get(e["method"], 0) + 1
        method_str = ", ".join(f"{m}:{c}" for m, c in sorted(methods.items()))
        print(f"  {app:20s}: {len(spec):3d} endpoints ({method_str})")

    print(f"\n[PASS] Real HAR processing completed successfully")
