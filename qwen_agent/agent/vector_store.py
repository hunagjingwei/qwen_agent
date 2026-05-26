"""向量存储模块 - 使用 FAISS 管理向量索引"""
import os
import json
import numpy as np
import faiss


class VectorStore:
    """FAISS 向量存储管理类"""

    def __init__(self, dimension: int, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, vectors: np.ndarray, texts: list):
        """添加向量和对应文本"""
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query: np.ndarray, top_k: int = 5) -> list:
        """检索最相似的向量"""
        query = query.astype('float32')
        distances, indices = self.index.search(query, min(top_k, len(self.texts)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "distance": float(dist)
                })
        return results

    def save(self, path: str):
        """保存索引到磁盘"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        meta_path = path + ".meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({"texts": self.texts, "dimension": self.dimension}, f)

    def load(self, path: str):
        """从磁盘加载索引"""
        self.index = faiss.read_index(path)
        meta_path = path + ".meta.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            self.texts = meta["texts"]