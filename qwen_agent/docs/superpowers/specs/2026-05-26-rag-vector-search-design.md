# RAG 向量语义检索设计文档

**日期：** 2026-05-26
**目标：** 为 Qwen Agent 添加基于向量语义检索的历史经验检索功能

---

## 1. 整体架构

```
User Message → Agent → [RAG Module] → Retrieved QA Pairs → Context
                                    ↓
                           HistoryStorage (SQLite)
                                    ↓
                           VectorIndex (FAISS + 嵌入模型)
```

**核心流程：**
1. 对话结束后，将 QA 对文本向量化，存入 FAISS 索引
2. 新查询时，将用户问题向量化，在 FAISS 中检索相似 QA
3. 将检索结果注入 Agent 的 prompt context

---

## 2. 组件设计

### 2.1 新增组件

| 组件 | 文件 | 职责 |
|------|------|------|
| `VectorStore` | `agent/vector_store.py` | 管理 FAISS 索引，提供 add / search / save / load 接口 |
| `QAExtractor` | `agent/qa_extractor.py` | 从对话历史中提取并构建 QA 对 |
| `RAGRetriever` | `agent/rag_retriever.py` | 整合检索逻辑，对接 Agent 的 context 构建 |

### 2.2 修改组件

| 组件 | 修改内容 |
|------|----------|
| `Agent` (agent/core.py) | 在 `_build_messages()` 前调用 RAG 检索，注入 context |

---

## 3. 数据流

```
对话结束 → QAExtractor.extract(messages) → QA 对文本
                                      ↓
                           VectorStore.add(qa_texts) → FAISS 索引更新
                                      ↓
                           Storage.save(...) → SQLite 持久化

新查询 → query embedding → VectorStore.search(top_k=3) → 相关 QA
                                                  ↓
                                        RAGRetriever → context string
                                                  ↓
                                        Agent._build_messages() 注入 context
```

---

## 4. 持久化策略

| 数据 | 存储位置 | 说明 |
|------|----------|------|
| 向量索引 | `vector_store.faiss` | FAISS 二进制文件 |
| 向量元数据 | `vector_meta.json` | QA 对 ID ↔ 向量索引的映射 |
| QA 对原文 | 复用现有 HistoryStorage | SQLite |

---

## 5. 错误处理

- **嵌入模型加载失败** — 回退到纯关键词检索，Agent 继续工作
- **FAISS 查询失败** — 返回空结果，不阻塞 Agent
- **索引文件损坏** — 删除重建，触发全量重新索引

---

## 6. 配置参数

```python
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 轻量多语言模型
TOP_K = 3                      # 检索返回最相关的 3 条 QA
SIMILARITY_THRESHOLD = 0.6     # 低于此阈值的结果丢弃
INDEX_DIR = "./vector_index"  # 索引文件目录
```

---

## 7. 实现步骤

1. **VectorStore** — FAISS 索引管理，支持 add / search / save / load
2. **QAExtractor** — 从 messages 列表提取 QA 对
3. **RAGRetriever** — 封装检索逻辑，返回格式化 context
4. **Agent 集成** — 在 `_build_messages()` 中调用 RAG，注入检索结果