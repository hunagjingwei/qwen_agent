import numpy as np

def test_vector_store_add_and_search():
    from agent.vector_store import VectorStore

    store = VectorStore(dimension=4)
    # 添加两条向量
    store.add(np.array([[0.1, 0.2, 0.3, 0.4]]), ["QA1 content"])
    store.add(np.array([[0.5, 0.6, 0.7, 0.8]]), ["QA2 content"])

    # 检索相似向量
    results = store.search(np.array([[0.1, 0.2, 0.3, 0.4]]), top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "QA1 content"