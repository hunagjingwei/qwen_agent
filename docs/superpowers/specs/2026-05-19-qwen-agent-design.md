# Qwen3.5-4B Agent 系统设计

> **日期**: 2026-05-19
> **模型**: Qwen3.5-4B (FP16)
> **硬件**: RTX 3090 24GB
> **架构**: Function Calling
> **推理后端**: vLLM
> **部署形式**: 命令行交互 (CLI)

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────┐
│                     CLI 入口                         │
│              (用户输入 → 结果展示)                    │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              主 Agent (Function Calling)              │
│  意图识别 → 函数选择 → 参数构造 → 执行 → 结果解析      │
└─────────────────────┬───────────────────────────────┘
                      │
      ┌───────────────┼───────────────┬───────────────┐
      ▼               ▼               ▼               ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│对话函数   │   │编码函数   │   │文档函数   │   │翻译/数学函数│
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              vLLM 推理引擎 (FP16)                     │
│              Qwen3.5-4B 模型                         │
└─────────────────────────────────────────────────────┘
```

---

## 2. Function Calling 函数定义

| 函数名 | 描述 | 参数 |
|--------|------|------|
| `chat` | 通用对话，直接返回模型回复 | `message: str` |
| `code_executor` | 执行 Python 或 Shell 代码 | `language: str`, `code: str` |
| `document_reader` | 读取文档内容 | `file_path: str`, `start_line: int`, `end_line: int` |
| `math_calc` | 数学表达式计算 | `expression: str` |
| `translator` | 多语言翻译 | `text: str`, `source_lang: str`, `target_lang: str` |

---

## 3. 技术栈

| 组件 | 技术选型 |
|------|----------|
| 模型推理 | vLLM + Qwen3.5-4B (FP16) |
| CLI 界面 | Python + rich (终端美化) |
| 代码执行 | `exec()` 沙箱 + 超时控制 |
| 文档解析 | PyPDF2 / 内置文本解析 |
| 数学计算 | sympy |
| 翻译 | argostranslate (离线) |

---

## 4. 文件结构

```
qwen_agent/
├── main.py              # CLI 入口
├── agent/
│   ├── __init__.py
│   ├── core.py          # Agent 核心逻辑
│   ├── functions.py     # Function Calling 定义
│   └── prompts.py       # 提示词模板
├── tools/
│   ├── __init__.py
│   ├── chat.py          # 对话工具
│   ├── code.py          # 代码执行
│   ├── document.py      # 文档读取
│   ├── math_tool.py     # 数学计算
│   └── translator.py    # 翻译
├── requirements.txt
└── README.md
```

---

## 5. 核心流程

### 5.1 单轮对话流程

```
用户输入 → vLLM (Function Calling) → 解析函数调用 → 执行工具 → 返回结果 → vLLM 生成最终回复 → 输出
```

### 5.2 工具执行逻辑

- `chat`: 直接调用 vLLM 的 chat completion 接口
- `code_executor`: 在沙箱中执行代码，支持 Python 和 Shell，5秒超时
- `document_reader`: 读取文件指定行范围
- `math_calc`: 使用 sympy 计算数学表达式
- `translator`: 使用 argostranslate 进行离线翻译

---

## 6. vLLM 配置

```python
{
    "model": "/home/wujie/LLM_project/Qwen3.5-4B",
    "tensor_parallel_size": 1,
    "dtype": "float16",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.9,
}
```

---

## 7. 退出机制

- 用户输入 `exit` 或 `quit` 退出
- Ctrl+C 优雅退出