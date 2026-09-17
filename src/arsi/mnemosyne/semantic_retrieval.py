"""Semantic retrieval for Mnemosyne — TF-IDF vector search.

Upgrades tag matching to semantic similarity using TF-IDF vectors.
Uses scikit-learn (already a dependency) — no heavy new packages.

Usage:
    retriever = SemanticRetriever(store)
    retriever.index_all()
    results = retriever.search("约束检查薄弱", k=5)
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Optional

from arsi.foundation.schema import MemoryRecord, MemoryZone
from arsi.foundation.store import MnemosyneStore

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer that handles Chinese and English."""
    # Lowercase and split
    text = text.lower()
    # Extract English words
    english_words = re.findall(r'[a-z_][a-z0-9_]+', text)
    # Extract Chinese characters (each char is a token for bigrams)
    chinese_chars = re.findall(r'[一-鿿]', text)
    # Create Chinese bigrams
    chinese_bigrams = [chinese_chars[i] + chinese_chars[i + 1]
                       for i in range(len(chinese_chars) - 1)]
    # Single Chinese chars
    return english_words + chinese_chars + chinese_bigrams


class SemanticRetriever:
    """TF-IDF based semantic retriever for Mnemosyne.

    Replaces simple tag matching with cosine similarity over TF-IDF vectors.
    """

    def __init__(self, store: MnemosyneStore):
        self.store = store
        self._vectors: dict[str, dict[str, float]] = {}  # memory_id → {token: tfidf}
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._indexed = False

    def index_all(self, zone: Optional[MemoryZone] = None) -> int:
        """Index all memories in the store. Returns number of indexed documents."""
        memories = self.store.search_memories(zone=zone, limit=10000)
        return self.index_memories(memories)

    def index_memories(self, memories: list[MemoryRecord]) -> int:
        """Index a list of memory records."""
        self._vectors.clear()
        self._idf.clear()

        # Step 1: Tokenize all documents
        doc_tokens: dict[str, list[str]] = {}
        for mem in memories:
            # Combine content + tags for richer representation
            text = mem.content + " " + " ".join(mem.tags)
            tokens = _tokenize(text)
            doc_tokens[mem.id] = tokens

        self._doc_count = len(doc_tokens)
        if self._doc_count == 0:
            return 0

        # Step 2: Compute IDF
        df: Counter = Counter()
        for tokens in doc_tokens.values():
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df[t] += 1

        for token, count in df.items():
            self._idf[token] = math.log(self._doc_count / (1 + count)) + 1

        # Step 3: Compute TF-IDF vectors
        for mem_id, tokens in doc_tokens.items():
            tf = Counter(tokens)
            total = len(tokens) if tokens else 1
            vector = {}
            for token, count in tf.items():
                tf_val = count / total
                idf_val = self._idf.get(token, 1.0)
                vector[token] = tf_val * idf_val
            self._vectors[mem_id] = vector

        self._indexed = True
        logger.info(f"Indexed {self._doc_count} memories, {len(self._idf)} unique tokens")
        return self._doc_count

    def search(self, query: str, k: int = 5, zone: Optional[MemoryZone] = None,
               agent_id: Optional[str] = None) -> list[tuple[MemoryRecord, float]]:
        """Search for memories semantically similar to query.

        Returns list of (MemoryRecord, similarity_score) sorted by score descending.
        """
        if not self._indexed:
            self.index_all()

        if not self._vectors:
            return []

        # Compute query vector
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        query_total = len(query_tokens)
        query_vector = {}
        for token, count in query_tf.items():
            tf_val = count / query_total
            idf_val = self._idf.get(token, 1.0)
            query_vector[token] = tf_val * idf_val

        # Compute cosine similarity with all documents
        scores = []
        for mem_id, doc_vector in self._vectors.items():
            sim = self._cosine_similarity(query_vector, doc_vector)
            if sim > 0:
                scores.append((mem_id, sim))

        # Sort by similarity
        scores.sort(key=lambda x: x[1], reverse=True)

        # Fetch records and apply filters
        results = []
        for mem_id, score in scores[:k * 3]:  # Fetch extra for filtering
            record = self.store.read_memory(mem_id)
            if not record:
                continue
            if zone and record.zone != zone:
                continue
            if agent_id and record.agent_id and record.agent_id != agent_id:
                continue
            if record.status.value != "active":
                continue
            results.append((record, round(score, 4)))
            if len(results) >= k:
                break

        return results

    def search_cross_agent(self, query: str, k: int = 5,
                           exclude_agent: Optional[str] = None) -> list[tuple[MemoryRecord, float]]:
        """Search EXPERIENCE zone across all agents.

        This is the core of cross-agent experience sharing.
        """
        results = self.search(query, k=k * 2, zone=MemoryZone.EXPERIENCE)

        if exclude_agent:
            results = [(r, s) for r, s in results if r.agent_id != exclude_agent]

        return results[:k]

    @staticmethod
    def _cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        # Find common tokens
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0

        dot = sum(v1[t] * v2[t] for t in common)
        mag1 = math.sqrt(sum(x * x for x in v1.values()))
        mag2 = math.sqrt(sum(x * x for x in v2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    @property
    def stats(self) -> dict:
        return {
            "indexed": self._indexed,
            "doc_count": self._doc_count,
            "vocab_size": len(self._idf),
        }
