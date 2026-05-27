"""RAG 检索器 - 整合向量检索和 QA 提取"""
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from .vector_store import VectorStore
from .qa_extractor import QAExtractor


class RAGRetriever:
    """RAG 检索器 - 封装检索逻辑"""

    # 纠错检测关键词（当用户说这些词时，说明之前的回答是错的）
    CORRECTION_KEYWORDS = [
        "是错的", "错误", "不对", "不是这样", "应该", "正确的是",
        "纠正", "更正", "修正", "准确", "正确答案是", "不对，应该",
        "错了", "你说错了", "解释错了", "你的理解是错的"
    ]

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 3,
        similarity_threshold: float = 0.6,
        index_dir: str = "./vector_index"
    ):
        # If embedding_model is a string, use it as model name; otherwise use it directly as model
        if isinstance(embedding_model, str):
            self.embedding_model_name = embedding_model
            self.embedding_model = None
        else:
            self.embedding_model_name = None
            self.embedding_model = embedding_model

        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.index_dir = index_dir

        self.vector_store = None
        self.qa_extractor = QAExtractor()

    def _init_embedding_model(self):
        """延迟加载嵌入模型"""
        if self.embedding_model is None and self.embedding_model_name:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def _init_vector_store(self):
        """初始化向量存储"""
        if self.vector_store is None:
            self._init_embedding_model()
            dimension = self.embedding_model.get_embedding_dimension()
            self.vector_store = VectorStore(dimension=dimension)

    def index_conversation(self, messages: list):
        """为对话历史建立索引"""
        qa_pairs = self.qa_extractor.extract(messages)
        if not qa_pairs:
            return

        formatted_texts = self.qa_extractor.format_for_rag(qa_pairs)

        self._init_embedding_model()
        embeddings = self.embedding_model.encode(formatted_texts)

        self._init_vector_store()
        self.vector_store.add(embeddings, formatted_texts)

    def retrieve(self, query: str, top_k: int = None) -> list:
        """检索与查询相关的 QA"""
        top_k = top_k or self.top_k

        try:
            self._init_embedding_model()
            query_embedding = self.embedding_model.encode([query])

            self._init_vector_store()
            results = self.vector_store.search(query_embedding, top_k=top_k)

            if self.similarity_threshold > 0:
                # For L2 distance, lower is better. threshold = (1 - similarity_threshold) * 10
                # e.g., similarity_threshold=0.6 -> threshold=4.0
                threshold = (1 - self.similarity_threshold) * 10
                filtered_results = [
                    r for r in results
                    if r["distance"] < threshold
                ]
            else:
                filtered_results = results
            return filtered_results
        except Exception as e:
            print(f"[WARN] RAG retrieval failed: {e}")
            return []

    def format_context(self, retrieved_results: list) -> str:
        """格式化检索结果为 context 字符串"""
        if not retrieved_results:
            return ""

        context_parts = ["相关历史经验："]
        for i, result in enumerate(retrieved_results, 1):
            context_parts.append(f"{i}. {result['text']}")

        return "\n".join(context_parts)

    def save_index(self):
        """保存索引到磁盘"""
        if self.vector_store:
            os.makedirs(self.index_dir, exist_ok=True)
            index_path = f"{self.index_dir}/vector_store.faiss"
            self.vector_store.save(index_path)

    def load_index(self) -> bool:
        """从磁盘加载索引"""
        index_path = f"{self.index_dir}/vector_store.faiss"
        if not os.path.exists(index_path):
            return False
        try:
            self._init_embedding_model()
            dimension = self.embedding_model.get_embedding_dimension()
            self.vector_store = VectorStore(dimension=dimension)
            self.vector_store.load(index_path)
            return True
        except Exception as e:
            print(f"[WARN] Failed to load RAG index: {e}")
            return False

    def is_correction(self, user_message: str) -> bool:
        """检测用户是否在纠正之前的错误回答

        Args:
            user_message: 用户输入

        Returns:
            True 如果检测到纠错意图
        """
        user_lower = user_message.lower()
        return any(keyword in user_lower for keyword in self.CORRECTION_KEYWORDS)

    def cleanup_error_qa(self, user_message: str, conversation_history: list):
        """当检测到用户纠错时，清理之前错误的 QA 对

        Args:
            user_message: 用户当前的纠错消息
            assistant_response: 之前 Assistant 的错误回答

        Returns:
            清理的 QA 对数量
        """
        if not self.vector_store or self.vector_store.index.ntotal == 0:
            return 0

        # 从历史中找到最近的错误回答（这里简化处理，实际可能需要更复杂的匹配）
        # 策略：如果检测到纠错，删除最近索引的包含"错"字的 QA 对
        removed = self.vector_store.remove_by_keywords(["错", "不对"])
        if removed > 0:
            print(f"[INFO] Cleaned up {removed} error QA pairs from RAG index")
            self.save_index()
        return removed