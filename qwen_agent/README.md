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

## 工具

| 工具名 | 说明 | 依赖 |
|--------|------|------|
| chat | 通用对话 | vLLM |
| code_executor | 执行 Python/Shell 代码 | - |
| document_reader | 读取文本/PDF 文件 | PyPDF2 |
| math_calc | 数学表达式计算 | sympy |
| translator | 多语言翻译 | argostranslate |
| weather | 查询城市天气 | WeatherAPI.com |

## 配置

模型路径在 `main.py` 中修改：

```python
MODEL_PATH = "/home/wujie/LLM_project/Qwen3.5-4B"
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
│   └── prompts.py      # 提示词模板
└── tools/
    ├── chat.py         # 对话工具
    ├── code.py         # 代码执行
    ├── document.py     # 文档读取
    ├── math_tool.py    # 数学计算
    ├── translator.py   # 翻译
    └── weather.py      # 天气查询
```