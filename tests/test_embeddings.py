"""本地向量实现的确定性与归一化单元测试。"""

from app.rag.embeddings import LocalHashEmbeddings


def test_local_embeddings_are_deterministic_and_normalized():
    """相同文本必须得到相同单位向量，保证缓存和测试稳定。"""

    embeddings = LocalHashEmbeddings(128)
    first = embeddings.embed_query("RAG 检索增强生成")
    second = embeddings.embed_query("RAG 检索增强生成")
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6
