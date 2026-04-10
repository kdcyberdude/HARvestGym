"""
Persistent embedding cache for HARvestGym.

Acts as a transparent proxy in front of the HF Inference API.
Before calling the API, checks a disk-backed cache keyed by SHA-256 of the
input text.  After fetching from the API, stores results so they survive
across episodes, restarts, and RL training runs.

Design:
  - Key   : first 24 hex chars of SHA-256(text.encode("utf-8"))
  - Value : L2-normalized float32 embedding vector (768-dim for embeddinggemma-300m)
  - Cap   : MAX_ENTRIES (2000) — FIFO eviction when full
  - Format: compressed numpy .npz  (~4 MB for 2000 × 768 float32 entries)
  - Location:
      1. $HARVGYM_CACHE_DIR/embed_cache.npz  (if env var set)
      2. <project_root>/.embed_cache/embed_cache.npz  (default, writable local dev)
      3. /tmp/harvgym_embed_cache/embed_cache.npz  (fallback for read-only containers)

Usage:
    from server.tools.embed_cache import get_cache

    cache = get_cache()
    emb = cache.get("some text")          # np.ndarray or None
    cache.put("some text", emb_array)     # single entry, saves immediately

    # Preferred: batch operations — single file write for all misses
    results, miss_idx = cache.get_batch(texts)
    # ... fetch API for miss_idx ...
    cache.put_batch([(texts[i], new_embs[j]) for j, i in enumerate(miss_idx)])
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_ENTRIES = 2000
_CACHE_FILENAME = "embed_cache.npz"
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_cache_path() -> Path:
    """Return a writable cache path, falling back gracefully."""
    # 1. Explicit env override
    env_dir = os.environ.get("HARVGYM_CACHE_DIR")
    if env_dir:
        p = Path(env_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / _CACHE_FILENAME

    # 2. Project root (works locally and persists across runs)
    project_cache = _PROJECT_ROOT / ".embed_cache"
    try:
        project_cache.mkdir(parents=True, exist_ok=True)
        # Quick write test
        test = project_cache / ".write_test"
        test.touch()
        test.unlink()
        return project_cache / _CACHE_FILENAME
    except OSError:
        pass

    # 3. /tmp fallback (ephemeral but always writable — e.g. HF Spaces)
    tmp = Path(tempfile.gettempdir()) / "harvgym_embed_cache"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp / _CACHE_FILENAME


# ---------------------------------------------------------------------------
# Cache class
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Thread-safe (single-process) disk-persistent embedding cache.

    Embeddings stored here are already L2-normalized — ready for direct
    cosine similarity with dot product.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or _resolve_cache_path()
        # OrderedDict preserves insertion order for FIFO eviction
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    # Key derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    # ------------------------------------------------------------------
    # Single-entry API
    # ------------------------------------------------------------------

    def get(self, text: str) -> np.ndarray | None:
        """Return cached L2-normalized embedding or None on miss."""
        return self._store.get(self._key(text))

    def put(self, text: str, embedding: np.ndarray, *, save: bool = True) -> None:
        """
        Cache one embedding (already L2-normalized).
        Evicts the oldest entry if at capacity.
        Set save=False when doing batch puts — call save() manually after.
        """
        key = self._key(text)
        if key in self._store:
            return  # idempotent
        if len(self._store) >= MAX_ENTRIES:
            self._store.popitem(last=False)  # FIFO: remove oldest
        self._store[key] = embedding.astype(np.float32)
        self._dirty = True
        if save:
            self._save()

    # ------------------------------------------------------------------
    # Batch API (preferred — single disk write for many entries)
    # ------------------------------------------------------------------

    def get_batch(
        self, texts: list[str]
    ) -> tuple[list[np.ndarray | None], list[int]]:
        """
        Look up multiple texts at once.

        Returns:
            results     : one entry per input text (np.ndarray or None)
            miss_indices: indices into texts that were not in cache
        """
        results: list[np.ndarray | None] = []
        miss_indices: list[int] = []
        for i, text in enumerate(texts):
            emb = self.get(text)
            results.append(emb)
            if emb is None:
                miss_indices.append(i)
        return results, miss_indices

    def put_batch(self, pairs: list[tuple[str, np.ndarray]]) -> None:
        """
        Cache a batch of (text, embedding) pairs — single disk write.
        All embeddings must already be L2-normalized.
        """
        for text, emb in pairs:
            self.put(text, emb, save=False)  # defer save
        if self._dirty:
            self._save()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self._store)

    def cache_path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = np.load(str(self._path), allow_pickle=True)
            keys: list[str] = data["keys"].tolist()
            values: np.ndarray = data["values"]
            for k, v in zip(keys, values):
                self._store[k] = v.astype(np.float32)
            print(
                f"[embed_cache] Loaded {len(self._store)}/{MAX_ENTRIES} entries "
                f"from {self._path}",
                flush=True,
            )
        except Exception as e:
            print(f"[embed_cache] Could not load cache ({e}) — starting empty.", flush=True)
            self._store.clear()

    def _save(self) -> None:
        if not self._store:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            keys = np.array(list(self._store.keys()), dtype=object)
            values = np.stack(list(self._store.values())).astype(np.float32)
            np.savez_compressed(str(self._path), keys=keys, values=values)
            self._dirty = False
        except Exception as e:
            print(f"[embed_cache] Save failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_cache: EmbeddingCache | None = None


def get_cache() -> EmbeddingCache:
    """Return the process-level singleton cache (loads from disk on first call)."""
    global _cache
    if _cache is None:
        _cache = EmbeddingCache()
    return _cache
