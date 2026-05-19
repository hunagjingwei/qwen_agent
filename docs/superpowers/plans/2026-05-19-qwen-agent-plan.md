# Qwen3.5-4B Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个基于 Qwen3.5-4B 的 CLI Agent，支持对话、编码、文档阅读、数学计算、翻译功能，采用 Function Calling 架构。

**Architecture:** 主 Agent 通过 vLLM 加载 Qwen3.5-4B (FP16)，通过 Function Calling 调用各类工具函数，CLI 交互。

**Tech Stack:** vLLM, transformers, rich, sympy, argostranslate, PyPDF2

---

## 文件结构

```
/home/wujie/LLM_project/qwen-3.5/qwen_agent/
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

### Task 1: 项目初始化与环境准备

**Files:**
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/requirements.txt`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/README.md`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/agent/__init__.py`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/tools/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
vllm>=0.4.0
transformers>=4.37.0
torch>=2.0.0
rich>=13.0.0
sympy>=1.12
argostranslate>=1.8.0
PyPDF2>=3.0.0
```

- [ ] **Step 2: 创建目录结构和 __init__.py 文件**

```bash
mkdir -p /home/wujie/LLM_project/qwen-3.5/qwen_agent/agent
mkdir -p /home/wujie/LLM_project/qwen-3.5/qwen_agent/tools
touch /home/wujie/LLM_project/qwen-3.5/qwen_agent/agent/__init__.py
touch /home/wujie/LLM_project/qwen-3.5/qwen_agent/tools/__init__.py
```

- [ ] **Step 3: 创建 README.md**

```markdown
# Qwen3.5-4B Agent

基于 Qwen3.5-4B 的 CLI Agent，支持对话、编码、文档阅读、数学计算、翻译功能。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
python main.py
```

## 功能

- 对话
- 代码执行 (Python/Shell)
- 文档阅读
- 数学计算
- 翻译
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: initial project structure"
```

---

### Task 2: 工具函数实现

**Files:**
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/tools/chat.py`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/tools/code.py`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/tools/document.py`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/tools/math_tool.py`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/tools/translator.py`

- [ ] **Step 1: 实现 chat.py**

```python
"""对话工具 - 直接调用 vLLM"""
from typing import Generator
import vllm

def chat(message: str, model_path: str, template: str) -> str:
    """
    通用对话，直接调用 vLLM 返回模型回复

    Args:
        message: 用户消息
        model_path: 模型路径
        template: ChatML 模板

    Returns:
        模型回复文本
    """
    llm = vllm.LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )

    prompt = template.format(messages=[{"role": "user", "content": message}])
    sampling_params = vllm.SamplingParams(temperature=0.7, max_tokens=2048)
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text
```

- [ ] **Step 2: 实现 code.py**

```python
"""代码执行工具"""
import sys
import io
import traceback
import signal
from typing import Dict, Any

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("代码执行超时 (5秒)")

def code_executor(language: str, code: str) -> Dict[str, Any]:
    """
    执行 Python 或 Shell 代码

    Args:
        language: "python" 或 "shell"
        code: 要执行的代码

    Returns:
        {"success": bool, "output": str, "error": str}
    """
    if language == "python":
        return _execute_python(code)
    elif language == "shell":
        return _execute_shell(code)
    else:
        return {"success": False, "output": "", "error": f"不支持的语言: {language}"}

def _execute_python(code: str) -> Dict[str, Any]:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

    try:
        exec(code, {"__name__": "__main__"})
        output = sys.stdout.getvalue()
        signal.alarm(0)
        return {"success": True, "output": output, "error": ""}
    except TimeoutError:
        return {"success": False, "output": "", "error": "代码执行超时 (5秒)"}
    except Exception:
        error = traceback.format_exc()
        return {"success": False, "output": sys.stdout.getvalue(), "error": error}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        signal.alarm(0)

def _execute_shell(code: str) -> Dict[str, Any]:
    import subprocess
    try:
        result = subprocess.run(
            code, shell=True, capture_output=True, text=True, timeout=5
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else ""
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "代码执行超时 (5秒)"}
```

- [ ] **Step 3: 实现 document.py**

```python
"""文档读取工具"""
import os
from typing import Dict, Any, Optional

def document_reader(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> Dict[str, Any]:
    """
    读取文档内容

    Args:
        file_path: 文件路径
        start_line: 起始行号 (从1开始)
        end_line: 结束行号 (从1开始)

    Returns:
        {"success": bool, "content": str, "error": str}
    """
    if not os.path.exists(file_path):
        return {"success": False, "content": "", "error": f"文件不存在: {file_path}"}

    if file_path.endswith(".pdf"):
        return _read_pdf(file_path, start_line, end_line)
    else:
        return _read_text(file_path, start_line, end_line)

def _read_text(
    file_path: str,
    start_line: Optional[int],
    end_line: Optional[int]
) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start = (start_line - 1) if start_line else 0
        end = end_line if end_line else len(lines)

        content = "".join(lines[start:end])
        return {"success": True, "content": content, "error": ""}
    except Exception as e:
        return {"success": False, "content": "", "error": str(e)}

def _read_pdf(
    file_path: str,
    start_line: Optional[int],
    end_line: Optional[int]
) -> Dict[str, Any]:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)

        start = (start_line - 1) if start_line else 0
        end = end_line if end_line else len(reader.pages)

        content = []
        for i in range(start, min(end, len(reader.pages))):
            content.append(reader.pages[i].extract_text())

        return {"success": True, "content": "\n".join(content), "error": ""}
    except Exception as e:
        return {"success": False, "content": "", "error": str(e)}
```

- [ ] **Step 4: 实现 math_tool.py**

```python
"""数学计算工具"""
from typing import Dict, Any
import sympy
from sympy import symbols, sympify, evaluate

def math_calc(expression: str) -> Dict[str, Any]:
    """
    计算数学表达式

    Args:
        expression: 数学表达式字符串，如 "2+3*4" 或 "x**2 + 2*x + 1"

    Returns:
        {"success": bool, "result": str, "error": str}
    """
    try:
        result = sympify(expression)
        return {"success": True, "result": str(result), "error": ""}
    except Exception as e:
        return {"success": False, "result": "", "error": f"表达式解析错误: {str(e)}"}
```

- [ ] **Step 5: 实现 translator.py**

```python
"""翻译工具"""
from typing import Dict, Any

def translator(text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
    """
    翻译文本

    Args:
        text: 要翻译的文本
        source_lang: 源语言代码，如 "en", "zh"
        target_lang: 目标语言代码，如 "zh", "en"

    Returns:
        {"success": bool, "translated_text": str, "error": str}
    """
    try:
        import argostranslate

        available = argostranslate.get_installed_languages()
        source = None
        target = None

        for lang in available:
            if source_lang.lower() in lang.code.lower():
                source = lang
            if target_lang.lower() in lang.code.lower():
                target = lang

        if not source or not target:
            return {
                "success": False,
                "translated_text": "",
                "error": f"不支持的语言对: {source_lang} -> {target_lang}"
            }

        translator = argostranslate.translate.new_translator(source, target)
        result = translator.translate(text)
        return {"success": True, "translated_text": result, "error": ""}
    except Exception as e:
        return {"success": False, "translated_text": "", "error": str(e)}
```

- [ ] **Step 6: Commit**

```bash
git add tools/
git commit -m "feat: implement all tool functions (chat, code, document, math, translator)"
```

---

### Task 3: Agent 核心逻辑

**Files:**
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/agent/functions.py`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/agent/prompts.py`
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/agent/core.py`

- [ ] **Step 1: 实现 functions.py (Function Calling 定义)**

```python
"""Function Calling 函数定义"""
from typing import List, Dict, Any

FUNCTIONS = [
    {
        "name": "chat",
        "description": "通用对话，直接返回模型回复",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "用户消息"
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "code_executor",
        "description": "执行 Python 或 Shell 代码",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "shell"],
                    "description": "编程语言"
                },
                "code": {
                    "type": "string",
                    "description": "要执行的代码"
                }
            },
            "required": ["language", "code"]
        }
    },
    {
        "name": "document_reader",
        "description": "读取文档内容",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号 (从1开始)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号 (从1开始)"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "math_calc",
        "description": "计算数学表达式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 2+3*4"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "translator",
        "description": "翻译文本",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要翻译的文本"
                },
                "source_lang": {
                    "type": "string",
                    "description": "源语言代码，如 en, zh"
                },
                "target_lang": {
                    "type": "string",
                    "description": "目标语言代码，如 zh, en"
                }
            },
            "required": ["text", "source_lang", "target_lang"]
        }
    }
]
```

- [ ] **Step 2: 实现 prompts.py**

```python
"""提示词模板"""

SYSTEM_PROMPT = """你是一个智能助手，可以通过 Function Calling 调用各种工具来回答用户问题。

可用工具：
- chat: 通用对话
- code_executor: 执行 Python 或 Shell 代码
- document_reader: 读取文档内容
- math_calc: 计算数学表达式
- translator: 翻译文本

当用户请求执行某个操作时，你应该调用相应的工具函数。"""

CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""
```

- [ ] **Step 3: 实现 core.py**

```python
"""Agent 核心逻辑"""
import json
from typing import Dict, Any, List, Optional, Callable
import vllm

from .functions import FUNCTIONS
from .prompts import SYSTEM_PROMPT, CHAT_TEMPLATE

class Agent:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.9
    ):
        self.model_path = model_path
        self.llm = vllm.LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tools = {
            "chat": self._chat,
            "code_executor": self._code_executor,
            "document_reader": self._document_reader,
            "math_calc": self._math_calc,
            "translator": self._translator,
        }

    def run(self, message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """运行 Agent 处理用户消息"""
        messages = conversation_history or []
        messages.append({"role": "user", "content": message})

        # 构建 prompt
        prompt = self._build_prompt(messages)

        # 调用 vLLM
        sampling_params = vllm.SamplingParams(
            temperature=0.7,
            max_tokens=2048,
            stop=["<|im_end|>"]
        )
        outputs = self.llm.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text

        # 解析 Function Calling
        result = self._parse_function_call(response_text)

        if result:
            # 执行工具
            tool_name = result["name"]
            tool_args = result["arguments"]
            tool_result = self._execute_tool(tool_name, tool_args)

            # 将工具结果加入对话
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

            # 再次调用模型生成最终回复
            final_prompt = self._build_prompt(messages)
            outputs = self.llm.generate([final_prompt], sampling_params)
            final_response = outputs[0].outputs[0].text

            return {"response": final_response, "messages": messages}
        else:
            return {"response": response_text, "messages": messages}

    def _build_prompt(self, messages: List[Dict]) -> str:
        prompt = SYSTEM_PROMPT + "\n\n"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            elif role == "tool":
                prompt += f"<|im_start|>tool\nname={msg.get('name', 'unknown')}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def _parse_function_call(self, text: str) -> Optional[Dict[str, Any]]:
        """解析 Function Calling 格式"""
        try:
            if "function_call" in text or "Function" in text:
                # 简单解析 JSON 格式的函数调用
                start = text.find("{")
                end = text.rfind("}") + 1
                if start != -1 and end > start:
                    json_str = text[start:end]
                    data = json.loads(json_str)
                    return data
        except json.JSONDecodeError:
            pass
        return None

    def _execute_tool(self, name: str, args: Dict) -> Dict[str, Any]:
        """执行工具"""
        if name in self.tools:
            return self.tools[name](**args)
        return {"error": f"Unknown tool: {name}"}

    def _chat(self, message: str) -> str:
        """对话工具"""
        from tools.chat import chat
        return chat(message, self.model_path, CHAT_TEMPLATE)

    def _code_executor(self, language: str, code: str) -> Dict[str, Any]:
        """代码执行工具"""
        from tools.code import code_executor
        return code_executor(language, code)

    def _document_reader(self, file_path: str, start_line: int = None, end_line: int = None) -> Dict[str, Any]:
        """文档读取工具"""
        from tools.document import document_reader
        return document_reader(file_path, start_line, end_line)

    def _math_calc(self, expression: str) -> Dict[str, Any]:
        """数学计算工具"""
        from tools.math_tool import math_calc
        return math_calc(expression)

    def _translator(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """翻译工具"""
        from tools.translator import translator
        return translator(text, source_lang, target_lang)
```

- [ ] **Step 4: Commit**

```bash
git add agent/
git commit -m "feat: implement agent core with Function Calling"
```

---

### Task 4: CLI 入口

**Files:**
- Create: `/home/wujie/LLM_project/qwen-3.5/qwen_agent/main.py`

- [ ] **Step 1: 实现 main.py**

```python
#!/usr/bin/env python3
"""Qwen3.5-4B Agent CLI 入口"""
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.core import Agent

console = Console()

MODEL_PATH = "/home/wujie/LLM_project/Qwen3.5-4B"

def main():
    console.print(Panel("[bold blue]Qwen3.5-4B Agent[/bold blue]\n基于 Function Calling 的智能助手"))
    console.print()

    try:
        agent = Agent(
            model_path=MODEL_PATH,
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.9
        )
        console.print("[green]模型加载成功！[/green]\n")
    except Exception as e:
        console.print(f"[red]模型加载失败: {e}[/red]")
        sys.exit(1)

    conversation_history = []

    while True:
        try:
            user_input = console.input("[bold cyan]>>> [/bold cyan]")

            if user_input.lower() in ["exit", "quit", "退出"]:
                console.print("[yellow]再见！[/yellow]")
                break

            if not user_input.strip():
                continue

            with console.status("[bold green]思考中...[/bold green]"):
                result = agent.run(user_input, conversation_history)
                conversation_history = result["messages"]

            response = result["response"]
            console.print(Panel(response, title="[bold green]Assistant[/bold green]", border_style="green"))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]再见！[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            console.print()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add main.py
git commit -m "feat: add CLI entry point"
```

---

### Task 5: 完整测试验证

- [ ] **Step 1: 安装依赖**

```bash
cd /home/wujie/LLM_project/qwen-3.5/qwen_agent
pip install -r requirements.txt
```

- [ ] **Step 2: 测试运行**

```bash
python main.py
```

测试用例：
1. 对话："你好，你是谁？"
2. 代码执行："帮我写一个快速排序"
3. 文档读取：读取一个本地文件
4. 数学计算："计算 2+3*4"
5. 翻译："翻译 hello world 到中文"

---

## 自检清单

1. **Spec coverage**: 所有 Function Calling 函数已实现
2. **Placeholder scan**: 无 TBD/TODO
3. **Type consistency**: 函数签名一致

**Plan complete and saved to `/home/wujie/LLM_project/qwen-3.5/docs/superpowers/plans/2026-05-19-qwen-agent-plan.md`**