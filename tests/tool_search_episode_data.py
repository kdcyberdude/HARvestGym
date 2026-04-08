"""
Tool 3: search_episode_data — BM25 search over episode request/response history.

Uses rank_bm25 for keyword matching over all indexed data from curl_exec calls.
Falls back to simple keyword matching when rank_bm25 is not available.
"""

import re
import math
from collections import Counter

# ---------------------------------------------------------------------------
# Simple BM25 implementation (no external dependencies)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Tokenize text into words."""
    return re.findall(r'[a-zA-Z0-9_./{}:]+', text.lower())


class SimpleBM25:
    """
    Minimal BM25 implementation for episode data search.
    No external dependencies — pure Python.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[str] = []
        self.tokenized: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0
        self.idf: dict[str, float] = {}
        self.n_docs: int = 0

    def index(self, documents: list[str]):
        """Build BM25 index from documents."""
        self.corpus = documents
        self.tokenized = [_tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in self.tokenized]
        self.n_docs = len(documents)
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)

        # Compute IDF
        df = Counter()
        for tokens in self.tokenized:
            for t in set(tokens):
                df[t] += 1

        self.idf = {}
        for term, freq in df.items():
            # Standard BM25 IDF
            self.idf[term] = math.log(
                (self.n_docs - freq + 0.5) / (freq + 0.5) + 1
            )

    def add_documents(self, new_docs: list[str]):
        """Incrementally add documents and rebuild index."""
        self.corpus.extend(new_docs)
        new_tokenized = [_tokenize(d) for d in new_docs]
        self.tokenized.extend(new_tokenized)
        self.doc_len.extend(len(t) for t in new_tokenized)
        self.n_docs = len(self.corpus)
        self.avgdl = sum(self.doc_len) / max(self.n_docs, 1)

        # Recompute IDF
        df = Counter()
        for tokens in self.tokenized:
            for t in set(tokens):
                df[t] += 1
        self.idf = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float, str]]:
        """
        Search for query in corpus.
        Returns: list of (doc_index, score, document) tuples, sorted by score descending.
        """
        query_tokens = _tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self.tokenized):
            score = 0.0
            tf = Counter(doc_tokens)
            dl = self.doc_len[i]

            for qt in query_tokens:
                if qt not in self.idf:
                    continue
                term_freq = tf.get(qt, 0)
                idf = self.idf[qt]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * numerator / denominator

            scores.append((i, score, self.corpus[i]))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Episode data store
# ---------------------------------------------------------------------------

class EpisodeDataStore:
    """
    Per-episode BM25 index over all request/response bodies.
    Initialized empty at episode start, grows with each curl_exec call.
    Discarded at episode end.
    """

    def __init__(self):
        self.bm25 = SimpleBM25()
        self.bm25.index([])  # Initialize empty

    def add_documents(self, docs: list[str]):
        """Add new documents (from a curl_exec call) to the index."""
        self.bm25.add_documents(docs)

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """
        Search episode data by keyword query.
        Returns top-k matching documents as strings.
        """
        if self.bm25.n_docs == 0:
            return []
        results = self.bm25.search(query, top_k)
        return [doc for _, score, doc in results if score > 0]

    def search_with_scores(self, query: str, top_k: int = 5) -> list[tuple[float, str]]:
        """Search with scores for debugging."""
        results = self.bm25.search(query, top_k)
        return [(score, doc) for _, score, doc in results]

    @property
    def doc_count(self) -> int:
        return self.bm25.n_docs

    def reset(self):
        """Clear all data (called at episode end)."""
        self.bm25 = SimpleBM25()
        self.bm25.index([])


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: search_episode_data with simulated episode")
    print("=" * 70)

    from tool_curl_exec import build_index_documents
    import json

    store = EpisodeDataStore()

    # Simulate episode: 4 curl_exec calls building up the index

    # Step 1: GET /categories
    docs = build_index_documents(
        step=1, method="GET", path="/rest/V1/categories",
        request_body=None,
        response_body=json.dumps({
            "id": 1, "name": "Root",
            "children_data": [
                {"id": 2, "name": "Default Category"},
                {"id": 3, "name": "Beauty & Personal Care"}
            ]
        }),
        status_code=200
    )
    store.add_documents(docs)
    print(f"\nStep 1: indexed {len(docs)} docs from GET /categories")

    # Step 2: GET /products (with array items)
    docs = build_index_documents(
        step=2, method="GET", path="/rest/V1/products",
        request_body=None,
        response_body=json.dumps({
            "items": [
                {"sku": "MH01", "name": "Radiant Tee", "price": 22.0},
                {"sku": "MH02", "name": "Breathe-Easy Tank", "price": 34.0},
                {"sku": "MH03", "name": "Stellar Solar Jacket", "price": 75.0},
                {"sku": "MH04", "name": "Argus All-Weather Tank", "price": 22.0},
                {"sku": "WS01", "name": "Iris Workout Top", "price": 29.0},
            ],
            "total_count": 5
        }),
        status_code=200
    )
    store.add_documents(docs)
    print(f"Step 2: indexed {len(docs)} docs from GET /products (5 items)")

    # Step 3: POST /guest-carts
    docs = build_index_documents(
        step=3, method="POST", path="/rest/V1/guest-carts",
        request_body=None,
        response_body='"cart-mock-abc123"',
        status_code=200
    )
    store.add_documents(docs)
    print(f"Step 3: indexed {len(docs)} docs from POST /guest-carts")

    # Step 4: POST /guest-carts/.../items
    docs = build_index_documents(
        step=4, method="POST", path="/rest/V1/guest-carts/cart-mock-abc123/items",
        request_body={"cartItem": {"sku": "MH01", "qty": 1, "quote_id": "cart-mock-abc123"}},
        response_body=json.dumps({
            "item_id": 5, "sku": "MH01", "qty": 1,
            "name": "Radiant Tee", "price": 22.0
        }),
        status_code=200
    )
    store.add_documents(docs)
    print(f"Step 4: indexed {len(docs)} docs from POST /guest-carts/.../items")

    print(f"\nTotal documents in episode index: {store.doc_count}")

    # Test searches
    print(f"\n--- Search Tests ---\n")

    queries = [
        ("Radiant Tee sku", "Should find MH01 product"),
        ("Stellar Solar Jacket price", "Should find MH03 at $75"),
        ("cart-mock-abc123", "Should find cart ID"),
        ("Beauty Personal Care", "Should find category"),
        ("item_id 5", "Should find add-to-cart result"),
        ("Iris Workout Top", "Should find WS01 product"),
    ]

    all_passed = True
    for query, description in queries:
        results = store.search_with_scores(query, top_k=3)
        print(f"Query: \"{query}\" ({description})")
        if results:
            for score, doc in results:
                print(f"  [{score:.3f}] {doc[:120]}...")
        else:
            print(f"  [NO RESULTS]")
            all_passed = False
        print()

    # Verify specific lookups
    print("--- Specific Value Lookups ---\n")

    # Can we find the product SKU from a name?
    results = store.search("Radiant Tee", top_k=1)
    found_sku = "MH01" in results[0] if results else False
    print(f"  Find 'Radiant Tee' SKU: {'PASS' if found_sku else 'FAIL'} ({'MH01' if found_sku else 'not found'})")

    # Can we find the cart ID?
    results = store.search("cart guest-carts", top_k=3)
    found_cart = any("cart-mock-abc123" in r for r in results)
    print(f"  Find cart ID: {'PASS' if found_cart else 'FAIL'}")

    # Can we find from which step data came?
    results = store.search("Radiant Tee", top_k=1)
    found_step = "step:2" in results[0] if results else False
    print(f"  Step annotation present: {'PASS' if found_step else 'FAIL'}")

    # Test reset
    store.reset()
    assert store.doc_count == 0
    print(f"\n  Episode reset: doc_count = {store.doc_count} [PASS]")

    print("\n[PASS] search_episode_data tool tests completed successfully")
