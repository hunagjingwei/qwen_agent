# RAG 向量语义检索实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Qwen Agent 添加基于 FAISS 和嵌入模型的历史对话 QA 对语义检索功能

**Architecture:** 在现有 HistoryStorage 基础上，新增 VectorStore（FAISS 索引管理）、QAExtractor（QA 对提取）、RAGRetriever（检索逻辑）三个组件。Agent 在处理新查询时通过 RAGRetriever 检索相关历史 QA，注入到 prompt context 中。

**Tech Stack:** `sentence-transformers`（嵌入模型）+ `faiss`（向量索引）+ SQLite（现有持久化）

---

## 文件结构

```
qwen_agent/
├── agent/
│   ├── __init__.py
│   ├── core.py              # 修改：集成 RAGRetriever
│   ├── vector_store.py      # 新增：FAISS 索引管理
│   ├── qa_extractor.py     # 新增：QA 对提取
│   └── rag_retriever.py     # 新增：RAG 检索逻辑
├── docs/
│   └── superpowers/
│       ├── specs/           # 设计文档
│       └── plans/           # 本计划
└── tests/
    └── test_vector_store.py # 新增：测试
```

---

## Task 1: VectorStore — FAISS 索引管理

**Files:**
- Create: `agent/vector_store.py`
- Test: `tests/test_vector_store.py`

- [ ] **Step 1: 写测试**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_vector_store.py::test_vector_store_add_and_search -v`
Expected: FAIL — `agent.vector_store` 不存在

- [ ] **Step 3: 实现 VectorStore**

```python
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
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_vector_store.py::test_vector_store_add_and_search -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add agent/vector_store.py tests/test_vector_store.py
git commit -m "feat: add VectorStore for FAISS index management"
```

---

## Task 2: QAExtractor — QA 对提取

**Files:**
- Create: `agent/qa_extractor.py`
- Test: `tests/test_qa_extractor.py`

- [ ] **Step 1: 写测试**

```python
def test_extract_qa_pairs():
    from agent.qa_extractor import QAExtractor

    extractor = QAExtractor()
    messages = [
        {"role": "user", "content": "计算 1+1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "计算 2+2"},
        {"role": "assistant", "content": "4"},
    ]
    qa_pairs = extractor.extract(messages)
    assert len(qa_pairs) == 2
    assert "1+1" in qa_pairs[0]["question"]
    assert "2" in qa_pairs[0]["answer"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_qa_extractor.py::test_extract_qa_pairs -v`
Expected: FAIL — `agent.qa_extractor` 不存在

- [ ] **Step 3: 实现 QAExtractor**

```python
"""QA 提取器 - 从对话历史中提取问答对"""


class QAExtractor:
    """从对话历史中提取 QA 对"""

    def __init__(self):
        pass

    def extract(self, messages: list) -> list:
        """
        从消息列表中提取 QA 对

        Args:
            messages: 消息列表，每条消息包含 role 和 content

        Returns:
            QA 对列表，每条包含 question, answer, metadata
        """
        qa_pairs = []
        current_q = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                current_q = content
            elif role == "assistant" and current_q:
                qa_pairs.append({
                    "question": current_q,
                    "answer": content,
                    "metadata": {}
                })
                current_q = None

        return qa_pairs

    def format_for_rag(self, qa_pairs: list) -> list:
        """将 QA 对格式化为检索文本"""
        formatted = []
        for qa in qa_pairs:
            text = f"问：{qa['question']}\n答：{qa['answer']}"
            formatted.append(text)
        return formatted
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_qa_extractor.py::test_extract_qa_pairs -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add agent/qa_extractor.py tests/test_qa_extractor.py
git commit -m "feat: add QAExtractor for conversation QA pair extraction"
```

---

## Task 3: RAGRetriever — RAG 检索逻辑

**Files:**
- Create: `agent/rag_retriever.py`
- Test: `tests/test_rag_retriever.py`

- [ ] **Step 1: 写测试**

```python
from unittest.mock import Mock, patch

def test_rag_retriever_retrieve():
    from agent.rag_retriever import RAGRetriever

    with patch('agent.rag_retriever.SentenceTransformer') as mock_model:
        mock_instance = Mock()
        mock_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
        mock_model.return_value = mock_instance

        retriever = RAGRetriever(embedding_model=mock_model)
        retriever.vector_store.add = Mock()

        result = retriever.retrieve("计算问题", top_k=1)
        assert isinstance(result, list)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_rag_retriever.py::test_rag_retriever_retrieve -v`
Expected: FAIL

- [ ] **Step 3: 实现 RAGRetriever**

```python
"""RAG 检索器 - 整合向量检索和 QA 提取"""
import numpy as np
from sentence_transformers import SentenceTransformer
from .vector_store import VectorStore
from .qa_extractor import QAExtractor


class RAGRetriever:
    """RAG 检索器 - 封装检索逻辑"""

    def __init__(
        self,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 3,
        similarity_threshold: float = 0.6,
        index_dir: str = "./vector_index"
    ):
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.index_dir = index_dir

        self.embedding_model = None
        self.vector_store = None
        self.qa_extractor = QAExtractor()

    def _init_embedding_model(self):
        """延迟加载嵌入模型"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def _init_vector_store(self):
        """初始化向量存储"""
        if self.vector_store is None:
            self._init_embedding_model()
            dimension = self.embedding_model.get_sentence_embedding_dimension()
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

            filtered_results = [
                r for r in results
                if r["distance"] < (1 - self.similarity_threshold)
            ]
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
            import os
            os.makedirs(self.index_dir, exist_ok=True)
            index_path = f"{self.index_dir}/vector_store.faiss"
            self.vector_store.save(index_path)

    def load_index(self):
        """从磁盘加载索引"""
        import os
        index_path = f"{self.index_dir}/vector_store.faiss"
        if os.path.exists(index_path):
            self._init_embedding_model()
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.vector_store = VectorStore(dimension=dimension)
            self.vector_store.load(index_path)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_rag_retriever.py::test_rag_retriever_retrieve -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add agent/rag_retriever.py tests/test_rag_retriever.py
git commit -m "feat: add RAGRetriever for semantic retrieval"
```

---

## Task 4: Agent 集成 — 在 _build_messages 中调用 RAG

**Files:**
- Modify: `agent/core.py:566-582` (_build_messages 方法)
- Test: `tests/test_agent_rag.py` (新增)

- [ ] **Step 1: 写测试**

```python
def test_agent_build_messages_with_rag():
    from agent.core import Agent
    from unittest.mock import patch, Mock

    with patch('agent.core.vllm') as mock_vllm:
        mock_vllm.LLM.return_value = Mock()
        agent = Agent(model_path="/tmp/model")

        agent.rag_retriever = Mock()
        agent.rag_retriever.retrieve.return_value = [
            {"text": "问：1+1\n答：2", "distance": 0.3}
        ]
        agent.rag_retriever.format_context.return_value = "相关历史经验：\n1. 问：1+1\n答：2"

        messages = agent._build_messages("计算 1+1", [])
        assert any("相关历史经验" in str(m) for m in messages)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_agent_rag.py::test_agent_build_messages_with_rag -v`
Expected: FAIL

- [ ] **Step 3: 修改 Agent._build_messages**

在 `agent/core.py` 的 `_build_messages` 方法中添加 RAG 调用：

```python
def _build_messages(self, user_message: str, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构建消息列表"""
    messages = []

    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })

    # RAG 检索相关历史经验
    if self.rag_retriever:
        retrieved = self.rag_retriever.retrieve(user_message, top_k=3)
        if retrieved:
            rag_context = self.rag_retriever.format_context(retrieved)
            messages.append({
                "role": "system",
                "content": f"{rag_context}\n\n请参考上述历史经验回答用户问题。"
            })

    messages.extend(conversation_history)

    messages.append({
        "role": "user",
        "content": user_message
    })

    return messages
```

在 `Agent.__init__` 中初始化 `rag_retriever`：

```python
def __init__(
    self,
    model_path: str,
    tensor_parallel_size: int = 1,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.9,
    history_storage: Optional[HistoryStorage] = None
):
    # ... 现有初始化代码 ...

    # RAG 检索器
    from .rag_retriever import RAGRetriever
    self.rag_retriever = RAGRetriever()

    # ... 其余现有代码 ...
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_agent_rag.py::test_agent_build_messages_with_rag -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add agent/core.py tests/test_agent_rag.py
git commit -m "feat: integrate RAG retrieval into Agent._build_messages"
```

---

## Task 5: 对话结束后自动索引 QA

**Files:**
- Modify: `agent/core.py:876-878` (save_conversation 方法)

- [ ] **Step 1: 写测试**

```python
def test_agent_save_conversation_indexes_qa():
    from agent.core import Agent
    from unittest.mock import patch, Mock

    with patch('agent.core.vllm') as mock_vllm:
        mock_vllm.LLM.return_value = Mock()
        agent = Agent(model_path="/tmp/model")

        agent.rag_retriever = Mock()
        agent.history_storage = Mock()

        messages = [
            {"role": "user", "content": "计算 1+1"},
            {"role": "assistant", "content": "2"},
        ]

        agent.save_conversation("session1", messages)

        agent.rag_retriever.index_conversation.assert_called_once_with(messages)
        agent.history_storage.save.assert_called_once()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_agent_rag.py::test_agent_save_conversation_indexes_qa -v`
Expected: FAIL

- [ ] **Step 3: 修改 save_conversation 方法**

在 `agent/core.py` 的 `save_conversation` 方法末尾添加：

```python
def save_conversation(self, session_id: str, messages: List[Dict[str, Any]], metadata: Optional[Dict] = None):
    """保存对话到历史存储"""
    self.history_storage.save(session_id, messages, metadata)

    # 自动索引 QA 对到 RAG
    if self.rag_retriever:
        self.rag_retriever.index_conversation(messages)
        self.rag_retriever.save_index()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_agent_rag.py::test_agent_save_conversation_indexes_qa -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add agent/core.py
git commit -m "feat: auto-index QA pairs after saving conversation"
```

---

## Task 6: 启动时加载已有索引

**Files:**
- Modify: `agent/core.py` (Agent.__init__)

- [ ] **Step 1: 在 Agent.__init__ 中添加索引加载**

在 `Agent.__init__` 的 `self.rag_retriever = RAGRetriever()` 之后添加：

```python
# 尝试加载已有索引
try:
    self.rag_retriever.load_index()
    print("[INFO] RAG index loaded from disk")
except Exception as e:
    print(f"[INFO] No existing RAG index, will build new one: {e}")
```

- [ ] **Step 2: 运行测试**

Run: `pytest tests/test_agent_rag.py -v`
Expected: PASS (之前的测试仍然通过)

- [ ] **Step 3: 提交**

```bash
git add agent/core.py
git commit -m "feat: load existing RAG index on startup"
```

---

## 依赖安装

在开始前需要安装依赖：

```bash
pip install sentence-transformers faiss-cpu
```

---

## 计划自检

1. **Spec 覆盖：** ✓ VectorStore（Task 1）、QAExtractor（Task 2）、RAGRetriever（Task 3）、Agent 集成（Task 4-6）
2. **Placeholder 扫描：** ✓ 无 TBD/TODO
3. **类型一致性：** ✓ 方法名和签名在 Task 间一致

---

**Plan complete and saved to** `docs/superpowers/plans/2026-05-26-rag-vector-search-plan.md`

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**