#!/usr/bin/env python3
"""
Build (or refresh) parameter_pools.json by calling the live EC2 application APIs.

Usage:
    python scripts/build_parameter_pools.py --host ec2-16-59-2-56.us-east-2.compute.amazonaws.com
    python scripts/build_parameter_pools.py --host localhost   # if running directly on EC2
    python scripts/build_parameter_pools.py --host <IP> --output parameter_pools.json

Requirements: pip install requests
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

try:
    import requests
    requests.packages.urllib3.disable_warnings()
except ImportError:
    print("pip install requests")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
PORTS = {
    "shopping":       7770,
    "shopping_admin": 7780,
    "forum":          9999,
    "wikipedia":      8888,
}

ADMIN_USER = "admin"
ADMIN_PASS = "admin1234"

# Wikipedia articles to verify exist in the ZIM snapshot
WIKIPEDIA_TITLES = [
    ("Python (programming language)", "Python programming language",   "Python_(programming_language)"),
    ("Albert Einstein",               "Albert Einstein",               "Albert_Einstein"),
    ("World War II",                  "World War II",                  "World_War_II"),
    ("Photosynthesis",                "Photosynthesis",                "Photosynthesis"),
    ("Marie Curie",                   "Marie Curie",                   "Marie_Curie"),
    ("Moon",                          "Moon",                          "Moon"),
    ("JavaScript",                    "JavaScript",                    "JavaScript"),
    ("Eiffel Tower",                  "Eiffel Tower",                  "Eiffel_Tower"),
    ("Black hole",                    "Black hole",                    "Black_hole"),
    ("Charles Darwin",                "Charles Darwin",                "Charles_Darwin"),
    ("Artificial intelligence",       "Artificial intelligence",       "Artificial_intelligence"),
    ("DNA",                           "DNA",                           "DNA"),
    ("Mount Everest",                 "Mount Everest",                 "Mount_Everest"),
    ("Isaac Newton",                  "Isaac Newton",                  "Isaac_Newton"),
    ("Solar System",                  "Solar System",                  "Solar_System"),
    ("Great Wall of China",           "Great Wall of China",           "Great_Wall_of_China"),
    ("William Shakespeare",           "William Shakespeare",           "William_Shakespeare"),
    ("Amazon River",                  "Amazon River",                  "Amazon_River"),
    ("Quantum mechanics",             "Quantum mechanics",             "Quantum_mechanics"),
    ("Napoleon",                      "Napoleon",                      "Napoleon"),
]

# Post titles generated for template_5 — not fetched from the live app
FORUM_POST_TITLES = [
    "Thoughts on the latest developments in AI safety",
    "Best practices for remote work in 2026",
    "How do you stay motivated when learning a new skill?",
    "What are your favourite open-source projects right now?",
    "Underrated books that changed how you think",
    "Tips for beginner photographers — what I wish I knew",
    "The most interesting science paper I read this week",
    "Ask me anything about Python performance tuning",
    "Weekly discussion: what are you building this month?",
    "Hidden gems in streaming music you should know about",
    "Travel destinations that are worth the hype",
    "How to cook a perfect risotto — my method after 10 attempts",
    "What sport have you picked up recently and why?",
    "Recommend a documentary that genuinely surprised you",
    "Discussion: is functional programming overrated?",
    "Things that made you better at managing personal finance",
    "The weirdest film you watched and actually enjoyed",
    "My experience switching from VS Code to a different editor",
    "Why I started journaling and what changed",
    "Gaming setup upgrades that actually made a difference",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def base_url(host: str, app: str) -> str:
    return f"http://{host}:{PORTS[app]}"


def get_admin_token(host: str) -> str:
    url = f"{base_url(host, 'shopping_admin')}/rest/V1/integration/admin/token"
    resp = requests.post(url, json={"username": ADMIN_USER, "password": ADMIN_PASS}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def admin_get(host: str, path: str, token: str, params: dict = None):
    url = f"{base_url(host, 'shopping_admin')}{path}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


# ── Pool builders ─────────────────────────────────────────────────────────────

def build_category_pool(host: str, token: str) -> list:
    """Template 1: leaf categories from GET /rest/V1/categories/list."""
    data = admin_get(host, "/rest/V1/categories/list", token, params={"searchCriteria[pageSize]": 500})
    items = data.get("items", [])
    pool = []
    for item in items:
        # include all named categories; caller can filter to leaf nodes if needed
        if item.get("name") and item.get("id"):
            pool.append({"name": item["name"], "category_id": item["id"]})
    return pool


def build_product_pool(host: str, token: str, max_items: int = 50) -> list:
    """Templates 3 & 6: simple, in-stock products."""
    data = admin_get(host, "/rest/V1/products", token, params={
        "searchCriteria[filterGroups][0][filters][0][field]": "type_id",
        "searchCriteria[filterGroups][0][filters][0][value]": "simple",
        "searchCriteria[filterGroups][0][filters][0][conditionType]": "eq",
        "searchCriteria[pageSize]": max_items,
    })
    items = data.get("items", [])
    pool = []
    for item in items:
        name = item.get("name", "").strip()
        sku  = item.get("sku", "").strip()
        if name and sku:
            pool.append({"name": name, "sku": sku})
    return pool


def build_wikipedia_pool(host: str) -> list:
    """Template 2: verify known articles exist in the ZIM snapshot."""
    base = base_url(host, "wikipedia")
    verified = []
    for display, search_query, expected_slug in WIKIPEDIA_TITLES:
        check_url = f"{base}/wikipedia_en_all_maxi_2022-05/A/{expected_slug}"
        try:
            r = requests.head(check_url, timeout=8, allow_redirects=True)
            if r.status_code == 200:
                verified.append({
                    "display": display,
                    "search_query": search_query,
                    "expected_slug": expected_slug,
                })
            else:
                print(f"  [wikipedia] WARNING: {expected_slug} → HTTP {r.status_code}, skipping")
        except Exception as e:
            print(f"  [wikipedia] WARNING: could not reach {check_url}: {e}")
    return verified


def build_forum_category_pool(host: str) -> list:
    """Templates 4 & 5: forum slugs with at least one submission."""
    base = base_url(host, "forum")
    pool = []
    page = 1
    while True:
        try:
            r = requests.get(f"{base}/api/forums", params={"page": page}, timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [forum] WARNING: could not reach forums API: {e}")
            break
        items = data if isinstance(data, list) else data.get("items", data.get("forums", []))
        if not items:
            break
        for item in items:
            name = item.get("name") or item.get("forum_name") or item.get("normalizedName")
            display = item.get("title") or item.get("displayName") or name
            if name:
                pool.append({"forum_name": name, "display_name": display or name})
        if len(items) < 20:
            break
        page += 1
    # deduplicate by forum_name
    seen = set()
    deduped = []
    for entry in pool:
        if entry["forum_name"] not in seen:
            seen.add(entry["forum_name"])
            deduped.append(entry)
    return deduped


# ── Template 7 pool ───────────────────────────────────────────────────────────

def build_admin_product_pool() -> list:
    """Template 7: fully generated SKU/price/name tuples. No API call needed."""
    specs = [
        ("HAR-TEST-001", 19.99,  "HAR Training Widget Alpha"),
        ("HAR-TEST-002", 34.50,  "HAR Training Widget Beta"),
        ("HAR-TEST-003", 9.99,   "HAR Economy Pack"),
        ("HAR-TEST-004", 49.00,  "HAR Premium Kit"),
        ("HAR-TEST-005", 7.75,   "HAR Starter Bundle"),
        ("HAR-TEST-006", 129.00, "HAR Deluxe Set"),
        ("HAR-TEST-007", 22.00,  "HAR Standard Unit"),
        ("HAR-TEST-008", 14.95,  "HAR Basic Module"),
        ("HAR-TEST-009", 59.99,  "HAR Advanced Pack"),
        ("HAR-TEST-010", 3.50,   "HAR Mini Component"),
        ("HAR-TEST-011", 89.00,  "HAR Pro Edition"),
        ("HAR-TEST-012", 11.25,  "HAR Lite Version"),
        ("HAR-TEST-013", 199.99, "HAR Enterprise Module"),
        ("HAR-TEST-014", 6.00,   "HAR Sample Item"),
        ("HAR-TEST-015", 45.00,  "HAR Mid-Range Pack"),
        ("HAR-TEST-016", 25.00,  "HAR Core Component"),
        ("HAR-TEST-017", 75.00,  "HAR Extended Kit"),
        ("HAR-TEST-018", 18.50,  "HAR Value Bundle"),
        ("HAR-TEST-019", 99.00,  "HAR Complete Suite"),
        ("HAR-TEST-020", 2.99,   "HAR Micro Unit"),
    ]
    return [{"sku": sku, "price": price, "product_name": name} for sku, price, name in specs]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build HARvestGym parameter pools from live EC2 apps.")
    parser.add_argument("--host",   default="ec2-16-59-2-56.us-east-2.compute.amazonaws.com")
    parser.add_argument("--output", default="parameter_pools.json")
    args = parser.parse_args()

    host   = args.host
    output = Path(args.output)

    print(f"Building parameter pools — host: {host}\n")

    # Admin token (needed for Shopping endpoints)
    print("[1/7] Fetching admin token...")
    token = get_admin_token(host)
    print("      OK\n")

    # Template 1 — category pool
    print("[2/7] Template 1: Shopping categories...")
    cat_pool = build_category_pool(host, token)
    print(f"      {len(cat_pool)} categories found\n")

    # Templates 3 & 6 — product pool
    print("[3/7] Templates 3 & 6: Shopping products (simple, in-stock)...")
    prod_pool = build_product_pool(host, token)
    print(f"      {len(prod_pool)} products found\n")

    # Template 2 — Wikipedia
    print("[4/7] Template 2: Verifying Wikipedia articles...")
    wiki_pool = build_wikipedia_pool(host)
    print(f"      {len(wiki_pool)} articles verified\n")

    # Templates 4 & 5 — Forum categories
    print("[5/7] Templates 4 & 5: Forum categories...")
    forum_pool = build_forum_category_pool(host)
    # template_5 category pool excludes any image-only forums — same list for now
    forum_pool_t5 = forum_pool
    print(f"      {len(forum_pool)} forums found\n")

    # Template 5 — post titles (static)
    print("[6/7] Template 5: Post titles (static list, no API call)...")
    print(f"      {len(FORUM_POST_TITLES)} titles loaded\n")

    # Template 7 — admin product specs (static)
    print("[7/7] Template 7: Admin product specs (generated, no API call)...")
    admin_pool = build_admin_product_pool()
    print(f"      {len(admin_pool)} product specs loaded\n")

    # ── Assemble output ───────────────────────────────────────────────────────
    pools = {
        "_meta": {
            "description": "Static parameter pools for the 7 HARvestGym task templates.",
            "generated_at": str(date.today()),
            "generated_from_host": host,
            "how_to_refresh": "python scripts/build_parameter_pools.py --host <EC2_HOST>",
            "source_apps": {
                "shopping":       f"http://{host}:{PORTS['shopping']}/",
                "shopping_admin": f"http://{host}:{PORTS['shopping_admin']}/admin",
                "forum":          f"http://{host}:{PORTS['forum']}/",
                "wikipedia":      f"http://{host}:{PORTS['wikipedia']}/",
            },
        },
        "template_1": {
            "description": "List products in category {category_name}",
            "tier": "Easy",
            "app": "shopping",
            "slots": ["category_name"],
            "source_endpoint": "GET /rest/V1/categories/list?searchCriteria[pageSize]=500",
            "note": "Only leaf categories are meaningful for product listing tasks. category_id is stored for grader use — not exposed in the task string.",
            "pool": {"category_name": cat_pool},
        },
        "template_2": {
            "description": "Retrieve article summary for {title}",
            "tier": "Easy",
            "app": "wikipedia",
            "slots": ["title"],
            "source_endpoint": "HEAD /wikipedia_en_all_maxi_2022-05/A/{slug} (verification only)",
            "note": "expected_slug is stored for grader verification. The agent must derive the slug independently via GET /search.",
            "pool": {"title": wiki_pool},
        },
        "template_3": {
            "description": "Add {product_name} to a guest cart",
            "tier": "Medium",
            "app": "shopping",
            "slots": ["product_name"],
            "source_endpoint": "GET /rest/V1/products?searchCriteria[pageSize]=50 (simple, in-stock only)",
            "note": "SKU stored for grader use — agent must independently discover it via product search.",
            "pool": {"product_name": prod_pool},
        },
        "template_4": {
            "description": "Retrieve all posts in {forum_category} (authed)",
            "tier": "Medium",
            "app": "forum",
            "slots": ["forum_category"],
            "source_endpoint": "GET /api/forums?page=1",
            "note": "forum_name is the URL slug; display_name is the human-readable label.",
            "pool": {"forum_category": forum_pool},
        },
        "template_5": {
            "description": "Create a post titled {title} in {category}",
            "tier": "Hard",
            "app": "forum",
            "slots": ["title", "category"],
            "source_endpoint": "GET /api/forums?page=1 (for category); titles are generated",
            "note": "title and category are sampled independently. category list excludes any image-only forums.",
            "pool": {
                "title":    FORUM_POST_TITLES,
                "category": forum_pool_t5,
            },
        },
        "template_6": {
            "description": "Guest checkout for {product_name}",
            "tier": "Hard",
            "app": "shopping",
            "slots": ["product_name"],
            "source_endpoint": "GET /rest/V1/products?searchCriteria[pageSize]=50 (same pool as template_3)",
            "note": "Guest checkout email is always test@example.com (STATIC). Grader queries /rest/V1/orders by email to confirm order creation.",
            "pool": {"product_name": prod_pool},
        },
        "template_7": {
            "description": "Create a new product with SKU {sku}, price {price}",
            "tier": "Hard",
            "app": "shopping_admin",
            "slots": ["sku", "price", "product_name"],
            "source_endpoint": "Fully generated — SKUs follow HAR-XXXXX pattern, no collision with existing catalog.",
            "note": "All slots sampled together as a product_spec tuple. attribute_set_id=4 (Default) is STATIC. Grader calls GET /rest/V1/products/{sku} to verify creation.",
            "pool": {"product_spec": admin_pool},
        },
    }

    output.write_text(json.dumps(pools, indent=2))
    print(f"Written to {output}  ({output.stat().st_size:,} bytes)")

    # Summary
    print("\n=== POOL SUMMARY ===")
    for tid in [k for k in pools if k.startswith("template")]:
        t = pools[tid]
        counts = {slot: len(vals) for slot, vals in t["pool"].items()}
        print(f"  {tid}: {counts}")


if __name__ == "__main__":
    main()
