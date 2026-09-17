"""Tests for semantic retrieval."""
import pytest

from arsi.foundation.schema import MemoryRecord, MemoryZone
from arsi.foundation.store import MnemosyneStore
from arsi.mnemosyne.semantic_retrieval import SemanticRetriever, _tokenize


@pytest.fixture
def store():
    s = MnemosyneStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def retriever(store):
    return SemanticRetriever(store)


class TestTokenizer:
    def test_english(self):
        tokens = _tokenize("hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_chinese(self):
        tokens = _tokenize("约束检查薄弱")
        assert "约束" in tokens or "约" in tokens
        assert "检查" in tokens or "检" in tokens

    def test_mixed(self):
        tokens = _tokenize("learn 管道 failure")
        assert "learn" in tokens
        assert "failure" in tokens

    def test_empty(self):
        tokens = _tokenize("")
        assert tokens == []


class TestSemanticRetriever:
    def test_index_and_search(self, store, retriever):
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="在多文件重构中先跑测试再改代码成功率高",
            tags=["coding", "testing"],
            agent_id="a1",
        ))
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="写作任务先列大纲再写正文返工率低",
            tags=["writing", "planning"],
            agent_id="b1",
        ))
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="数据分析任务应该先检查数据质量",
            tags=["data", "quality"],
            agent_id="c1",
        ))

        count = retriever.index_all()
        assert count == 3

        results = retriever.search("测试代码", k=3)
        assert len(results) > 0
        # Should find the testing-related memory first
        assert "测试" in results[0][0].content or "代码" in results[0][0].content

    def test_cross_agent_search(self, store, retriever):
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="Agent A 发现约束检查是薄弱环节",
            tags=["decomposition"],
            agent_id="agent_a",
        ))
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="Agent B 学到了分解任务的技巧和约束检查方法",
            tags=["decomposition"],
            agent_id="agent_b",
        ))

        retriever.index_all()
        results = retriever.search_cross_agent("约束检查分解任务", exclude_agent="agent_a")

        # Should find Agent B's experience (Agent A excluded)
        assert len(results) >= 1
        for r, s in results:
            assert r.agent_id != "agent_a"

    def test_empty_store(self, retriever):
        results = retriever.search("anything")
        assert results == []

    def test_no_match(self, store, retriever):
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="completely unrelated content",
            tags=["test"],
        ))
        retriever.index_all()
        results = retriever.search("量子物理相对论", k=5)
        # May return results with low scores, that's ok
        assert isinstance(results, list)

    def test_zone_filter(self, store, retriever):
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="经验记忆内容",
            tags=["test"],
        ))
        store.write_memory(MemoryRecord(
            zone=MemoryZone.PROXY,
            content="代理记忆内容",
            tags=["test"],
            agent_id="a1",
        ))

        retriever.index_all()

        exp_results = retriever.search("记忆", k=5, zone=MemoryZone.EXPERIENCE)
        for r, s in exp_results:
            assert r.zone == MemoryZone.EXPERIENCE

        proxy_results = retriever.search("记忆", k=5, zone=MemoryZone.PROXY)
        for r, s in proxy_results:
            assert r.zone == MemoryZone.PROXY

    def test_similarity_ordering(self, store, retriever):
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="代码测试自动化",
            tags=["code", "test"],
        ))
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="烹饪食谱分享",
            tags=["cooking"],
        ))

        retriever.index_all()
        results = retriever.search("代码测试", k=2)

        if len(results) >= 2:
            # More relevant result should have higher score
            assert results[0][1] >= results[1][1]

    def test_stats(self, store, retriever):
        store.write_memory(MemoryRecord(zone=MemoryZone.EXPERIENCE, content="test", tags=["a"]))
        retriever.index_all()
        stats = retriever.stats
        assert stats["indexed"] is True
        assert stats["doc_count"] == 1
        assert stats["vocab_size"] > 0

    def test_incremental_reindex(self, store, retriever):
        store.write_memory(MemoryRecord(zone=MemoryZone.EXPERIENCE, content="first", tags=["a"]))
        retriever.index_all()
        assert retriever.stats["doc_count"] == 1

        store.write_memory(MemoryRecord(zone=MemoryZone.EXPERIENCE, content="second", tags=["b"]))
        retriever.index_all()
        assert retriever.stats["doc_count"] == 2
