# Qwen3.5-4B Agent

基于 Qwen3.5-4B 的 CLI Agent，支持对话、编码、文档阅读、数学计算、翻译、天气查询功能。

## 架构

- **推理后端**: vLLM (FP16)
- **调用方式**: Function Calling
- **硬件**: RTX 3090 24GB

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python main.py
```

## 功能

| 功能 | 命令示例 |
|------|----------|
| 对话 | 你好，你是谁？ |
| 代码执行 | 帮我写一个快速排序 |
| 文档阅读 | 读取 /path/to/file.txt |
| 数学计算 | 计算 2+3*4 |
| 翻译 | 翻译 hello world 到中文 |
| 天气查询 | 今天广州天气怎样 |

## 技术细节

### 整体架构

```
main.py → Agent (core.py) → 工具注册表 → 各工具实现
                              ↓
                        RAGRetriever → VectorStore + QAExtractor
```

### 工具调用流程

```
用户输入 → 模型生成 Tool Call → core._execute_tool()
           ↓
    tool_registry 查找工具函数
           ↓
    若是 math_tool 类：传入 func_name 参数
           ↓
    math_tool() 的 lambda dispatch 路由到具体函数
           ↓
    返回结果给模型
```

### 数学工具

| 工具名 | 说明 | 示例 |
|--------|------|------|
| derivative | 求函数导数 | `derivative(expression="x**2+3*x", variable="x", n=1)` |
| find_extrema | 求极值 | `find_extrema(expression="x**3-5*x", variable="x")` |
| analyze_monotonic | 单调性分析 | `analyze_monotonic(expression="x**2-4")` |
| circle_area | 圆面积/周长 | `circle_area(diameter=10)` |
| rectangle_area | 矩形面积 | `rectangle_area(length=5, width=3)` |
| savings_find_year | 存款多少年达到目标 | `savings_find_year(annual_deposit=150000, annual_rate=0.03, target=1000000, initial_principal=400000)` |
| savings_accumulation | n年后存款总额 | `savings_accumulation(annual_deposit=150000, annual_rate=0.03, years=10)` |
| loan_repayment | 贷款计算 | `loan_repayment(principal=1000000, annual_rate=0.05, years=20)` |

### RAG 机制

对话保存时：
1. `QAExtractor.extract()` 从对话历史提取 QA 对
2. `format_for_rag()` 格式化
3. `SentenceTransformer` 向量化
4. 存入 FAISS 索引

检索时：query 编码 → FAISS 搜索 → 返回相似 QA 作为上下文

### RAG 纠错机制

用户输入包含"是错的"、"不对"等关键词时：
- `is_correction()` 检测纠错意图
- `cleanup_error_qa()` 删除包含"错"、"不对"的错误 QA 对
- 防止错误知识被记住

### prompts.py 关键规则

```
- 优先使用专用工具而不是 code_executor
- 工具返回结果必须使用，不要自己计算
- 存款、贷款、导数等必须调用对应工具
```

## 配置

模型路径在 `main.py` 中修改：

```python
MODEL_PATH = "/home/wuiie/LLM_project/Qwen3.5-4B"
```

天气 API Key 在 `tools/weather.py` 中修改：

```python
API_KEY = "your_api_key_here"
```

## 项目结构

```
qwen_agent/
├── main.py              # CLI 入口
├── requirements.txt    # 依赖
├── README.md           # 本文档
├── agent/
│   ├── core.py         # Agent 核心逻辑
│   ├── functions.py    # Function Calling 定义
│   ├── prompts.py      # 提示词模板
│   ├── rag_retriever.py # RAG 检索器
│   ├── vector_store.py # FAISS 向量存储
│   └── qa_extractor.py # QA 对提取
└── tools/
    ├── chat.py         # 对话工具
    ├── code.py         # 代码执行
    ├── document.py     # 文档读取
    ├── math_tool.py    # 数学计算
    ├── translator.py   # 翻译
    └── weather.py      # 天气查询
```