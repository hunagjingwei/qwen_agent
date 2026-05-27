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
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, vectors: np.ndarray, texts: list):
        """添加向量和对应文本"""
        if len(vectors) != len(texts):
            raise ValueError("vectors and texts must have same length")
        vectors = vectors.astype('float32')
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query: np.ndarray, top_k: int = 5) -> list:
        """检索最相似的向量"""
        if self.index.ntotal == 0:
            return []
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
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            faiss.write_index(self.index, path)
            meta_path = path + ".meta.json"
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({"texts": self.texts, "dimension": self.dimension}, f)
        except Exception as e:
            raise IOError(f"Failed to save index to {path}: {e}")

    def load(self, path: str):
        """从磁盘加载索引"""
        try:
            self.index = faiss.read_index(path)
            meta_path = path + ".meta.json"
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                self.texts = meta["texts"]
        except Exception as e:
            raise IOError(f"Failed to load index from {path}: {e}")

    def remove_by_keywords(self, keywords: list) -> int:
        """删除包含关键词的文本，返回删除数量"""
        original_count = len(self.texts)
        # 找出不包含任何关键词的文本
        filtered_texts = []
        for text in self.texts:
            if not any(kw in text for kw in keywords):
                filtered_texts.append(text)
        self.texts = filtered_texts
        return original_count - len(self.texts)

    def rebuild_with_filter(self, filter_func):
        """根据过滤函数重建索引

        Args:
            filter_func: 接收 (text, vector) 返回 True 保留，False 删除

        Returns:
            删除的条目数量
        """
        # 收集保留的文本和向量
        kept_texts = []
        kept_vectors = []

        for i, text in enumerate(self.texts):
            vector = self.index.reconstruct(i)
            if filter_func(text, vector):
                kept_texts.append(text)
                kept_vectors.append(vector)

        removed_count = len(self.texts) - len(kept_texts)

        # 重建索引
        if kept_vectors:
            self.index.reset()
            self.index.add(np.array(kept_vectors).astype('float32'))
        else:
            self.index.reset()

        self.texts = kept_texts
        return removed_count