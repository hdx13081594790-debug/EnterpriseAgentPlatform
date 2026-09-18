from app.rag.embeddings import LocalHashEmbeddings


def test_local_embeddings_are_deterministic_and_normalized():
    embeddings = LocalHashEmbeddings(128)
    first = embeddings.embed_query("RAG 检索增强生成")
    second = embeddings.embed_query("RAG 检索增强生成")
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6

