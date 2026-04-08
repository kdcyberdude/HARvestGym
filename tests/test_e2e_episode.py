"""
End-to-End Episode Simulation: "Add Radiant Tee to a guest cart"

Simulates the full tool chain with mock data:
  browser_agent → search_endpoints → curl_exec → search_episode_data → done

Tests that values thread correctly between tools and that each tool's
output feeds properly into the next tool's input.
"""

import json
import os
import sys
import re

# Add tests dir to path
sys.path.insert(0, os.path.dirname(__file__))

from tool_browser_agent import browser_agent, extract_openapi_spec, spec_entry_to_text
from tool_search_endpoints import SearchEndpoints
from tool_curl_exec import mock_curl_exec, parse_curl_command
from tool_search_episode_data import EpisodeDataStore


def run_episode():
    """Simulate a full episode: Add 'Radiant Tee' to a guest cart."""

    print("=" * 70)
    print("E2E EPISODE: Add 'Radiant Tee' to a guest cart")
    print("URL: http://localhost:7770/")
    print("=" * 70)

    task = "Add 'Radiant Tee' to a guest cart"
    url = "http://localhost:7770/"
    mock_data_dir = os.path.join(os.path.dirname(__file__), "mock_data")

    # Episode state
    episode_index_docs = []
    episode_store = EpisodeDataStore()
    session_state = {}
    step = 0

    # -----------------------------------------------------------------------
    # STEP 1: browser_agent — discover endpoints
    # -----------------------------------------------------------------------
    step += 1
    print(f"\n{'='*50}")
    print(f"STEP {step}: browser_agent(\"{task}\", \"{url}\")")
    print(f"{'='*50}")

    # Load mock HAR directly (simulating HAR file exists)
    mock_har_path = os.path.join(mock_data_dir, "mock_har.json")
    with open(mock_har_path) as f:
        har_data = json.load(f)

    spec_entries = extract_openapi_spec(har_data, url)
    text_chunks = [spec_entry_to_text(e, "shopping") for e in spec_entries]

    # Build summary output (what RL agent sees)
    summary = {
        "app": "shopping",
        "endpoints": [{"method": e["method"], "path": e["path"]} for e in spec_entries],
        "total_endpoints": len(spec_entries),
        "note": "Use search_endpoints() for full details on any endpoint."
    }

    print(f"\nResult: {len(summary['endpoints'])} endpoints discovered:")
    for ep in summary["endpoints"]:
        print(f"  {ep['method']:6s} {ep['path']}")

    # Set up the search_endpoints tool with browser_agent output (NOT catalog ground truth)
    # In the real system, search_endpoints searches the GEMMA embeddings built by browser_agent
    # from HAR data. Here we use keyword search as a test fallback for GEMMA.
    search_tool = SearchEndpoints()
    search_tool.load_from_browser_agent(text_chunks)

    print(f"\n  → search_endpoints index built: {len(text_chunks)} docs from browser_agent HAR output")

    # -----------------------------------------------------------------------
    # STEP 2: search_endpoints — "how to find a product by name?"
    # -----------------------------------------------------------------------
    step += 1
    print(f"\n{'='*50}")
    print(f"STEP {step}: search_endpoints(\"find product by name get sku\")")
    print(f"{'='*50}")

    results = search_tool.search("find product by name get sku", top_k=3)
    print(f"\nTop-3 results:")
    for i, r in enumerate(results):
        ep_match = re.search(r'endpoint: (\S+ \S+)', r)
        ep_name = ep_match.group(1) if ep_match else "?"
        print(f"  [{i+1}] {ep_name}")
        print(f"      {r[:150]}...")

    # Agent decides: GET /rest/V1/products with searchCriteria filter
    print(f"\n  → Agent decides: GET /rest/V1/products with name filter")

    # -----------------------------------------------------------------------
    # STEP 3: curl_exec — search for Radiant Tee
    # -----------------------------------------------------------------------
    step += 1
    print(f"\n{'='*50}")
    print(f"STEP {step}: curl_exec(\"curl .../products?searchCriteria[...]=Radiant+Tee\")")
    print(f"{'='*50}")

    result = mock_curl_exec(
        "curl 'http://localhost:7770/rest/V1/products?searchCriteria[filter_groups][0][filters][0][field]=name&searchCriteria[filter_groups][0][filters][0][value]=Radiant+Tee'",
        step, episode_index_docs
    )
    episode_store.add_documents(episode_index_docs[-len(episode_index_docs):])

    print(f"\nResult: status={result['status_code']}")
    body = result["body"]
    if isinstance(body, dict):
        items = body.get("items", [])
        print(f"  items shown: {len(items)}, total: {body.get('total_count', '?')}")
        for item in items:
            print(f"    sku={item['sku']}, name={item['name']}, price={item['price']}")
        if "_list_truncated" in body:
            print(f"  [TRUNCATED] {body['_list_truncated']['note']}")

    # Agent extracts: sku="MH01" from response
    target_sku = "MH01"
    print(f"\n  → Agent extracts: sku='{target_sku}' for 'Radiant Tee'")

    # -----------------------------------------------------------------------
    # STEP 4: search_endpoints — "how to create a guest cart?"
    # -----------------------------------------------------------------------
    step += 1
    print(f"\n{'='*50}")
    print(f"STEP {step}: search_endpoints(\"create guest cart get cart id\")")
    print(f"{'='*50}")

    results = search_tool.search("create guest cart get cart id", top_k=3)
    print(f"\nTop result:")
    ep_match = re.search(r'endpoint: (\S+ \S+)', results[0])
    print(f"  {ep_match.group(1) if ep_match else results[0][:80]}")

    print(f"\n  → Agent decides: POST /rest/V1/guest-carts")

    # -----------------------------------------------------------------------
    # STEP 5: curl_exec — create guest cart
    # -----------------------------------------------------------------------
    step += 1
    print(f"\n{'='*50}")
    print(f"STEP {step}: curl_exec(\"curl -X POST .../guest-carts\")")
    print(f"{'='*50}")

    result = mock_curl_exec(
        "curl -X POST 'http://localhost:7770/rest/V1/guest-carts' -H 'Content-Type: application/json'",
        step, episode_index_docs
    )
    episode_store.add_documents(episode_index_docs[-1:])

    cart_id = result["body"]
    print(f"\nResult: status={result['status_code']}, cart_id={cart_id}")
    print(f"\n  → Agent extracts: cart_id='{cart_id}'")

    # -----------------------------------------------------------------------
    # STEP 6: search_endpoints — "how to add item to guest cart?"
    # -----------------------------------------------------------------------
    step += 1
    print(f"\n{'='*50}")
    print(f"STEP {step}: search_endpoints(\"add item to guest cart\")")
    print(f"{'='*50}")

    results = search_tool.search("add item to guest cart cartId sku", top_k=3)
    print(f"\nTop result:")
    print(f"  {results[0][:200]}...")

    print(f"\n  → Agent decides: POST /rest/V1/guest-carts/{{cartId}}/items")
    print(f"    cartId = {cart_id} (from step {step-1})")
    print(f"    sku = {target_sku} (from step 3)")
    print(f"    quote_id = {cart_id} (DERIVED, same as cartId)")

    # -----------------------------------------------------------------------
    # STEP 7: curl_exec — add Radiant Tee to cart
    # -----------------------------------------------------------------------
    step += 1
    print(f"\n{'='*50}")
    print(f"STEP {step}: curl_exec(\"curl -X POST .../guest-carts/{cart_id}/items\")")
    print(f"{'='*50}")

    result = mock_curl_exec(
        f'curl -X POST "http://localhost:7770/rest/V1/guest-carts/{cart_id}/items" '
        f'-H "Content-Type: application/json" '
        f'-d \'{{"cartItem":{{"sku":"{target_sku}","qty":1,"quote_id":"{cart_id}"}}}}\'',
        step, episode_index_docs
    )
    episode_store.add_documents(episode_index_docs[-2:])  # request + response docs

    print(f"\nResult: status={result['status_code']}")
    body = result["body"]
    if isinstance(body, dict):
        print(f"  item_id={body.get('item_id')}, sku={body.get('sku')}, qty={body.get('qty')}")

    # -----------------------------------------------------------------------
    # Test: search_episode_data — can we find values from prior steps?
    # -----------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"VERIFICATION: search_episode_data queries")
    print(f"{'='*50}")

    print(f"\nEpisode index: {episode_store.doc_count} documents total\n")

    # Can we find the product from step 2?
    results = episode_store.search("Radiant Tee sku", top_k=1)
    found_product = results and "MH01" in results[0]
    print(f"  Find 'Radiant Tee' sku from products response: {'PASS' if found_product else 'FAIL'}")
    if results:
        print(f"    → {results[0][:100]}...")

    # Can we find the cart ID from step 4?
    results = episode_store.search("guest-carts cart", top_k=3)
    found_cart = any("cart-mock" in r for r in results)
    print(f"\n  Find cart ID from create-cart response: {'PASS' if found_cart else 'FAIL'}")
    if results:
        print(f"    → {results[0][:100]}...")

    # Can we find the add-to-cart confirmation?
    results = episode_store.search("item_id sku MH01", top_k=1)
    found_confirm = results and "item_id" in results[0]
    print(f"\n  Find add-to-cart confirmation: {'PASS' if found_confirm else 'FAIL'}")
    if results:
        print(f"    → {results[0][:100]}...")

    # -----------------------------------------------------------------------
    # STEP 8: done
    # -----------------------------------------------------------------------
    step += 1
    print(f"\n{'='*50}")
    print(f"STEP {step}: done(\"Radiant Tee (MH01) added to guest cart {cart_id}\")")
    print(f"{'='*50}")
    print(f"\n  Episode complete. {step} steps total.")
    print(f"  Episode index: {episode_store.doc_count} documents indexed")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"EPISODE SUMMARY")
    print(f"{'='*70}")
    print(f"""
  Task:       {task}
  App:        shopping (port 7770)
  Steps:      {step}
  Tools used: browser_agent → search_endpoints (x3) → curl_exec (x3) → done

  Value Threading:
    Step 3: GET /products → sku='MH01' (Radiant Tee)
    Step 5: POST /guest-carts → cart_id='{cart_id}'
    Step 7: POST /guest-carts/{cart_id}/items
            sku='{target_sku}' (from step 3)
            quote_id='{cart_id}' (DERIVED from step 5)

  Episode Index: {episode_store.doc_count} documents
    - Categories, products (5 items), cart creation, add-to-cart
    - All searchable via search_episode_data()

  Result: item_id=5, sku=MH01, qty=1 added to cart {cart_id}
""")

    # Assertions
    assert found_product, "Failed to find product in episode data"
    assert found_cart, "Failed to find cart ID in episode data"
    assert found_confirm, "Failed to find add-to-cart confirmation"

    print("[PASS] End-to-end episode simulation completed successfully\n")


if __name__ == "__main__":
    run_episode()
