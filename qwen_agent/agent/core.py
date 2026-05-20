"""Agent 核心逻辑

当前实现说明：
1. 使用 vLLM 的 chat() API 进行对话
2. 工具调用通过文本解析实现（从模型输出的 <tool_call> 标签中提取）
   - 原因：vLLM + Qwen3.5 的原生 Function Calling 响应 finish_reason 始终为 "stop"，
     未正确触发 tool_calls 模式，因此改用文本解析方案
3. FUNCTIONS: 定义了工具的 JSON Schema，原本用于 vLLM 原生 Function Calling，
   现用于参考工具定义和文档说明
4. CHAT_TEMPLATE: Qwen 的 Jinja2 聊天模板格式，用于 generate() API，
   当前使用 chat() API 时由 vLLM 自动处理消息格式，故未使用
"""

import json
import re
from typing import Dict, Any, List, Optional
import vllm
from vllm import SamplingParams

from .functions import FUNCTIONS  # 工具定义，用于文档说明
from .prompts import SYSTEM_PROMPT, CHAT_TEMPLATE  # CHAT_TEMPLATE 当前未使用


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
            "weather": self._weather,
        }

    def run(self, message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """处理用户消息，支持工具调用

        流程：
        1. 添加 system prompt 到消息列表
        2. 调用 vLLM chat 获取回复
        3. 解析回复文本中的 <tool_call> 标签
        4. 如有工具调用，执行工具并将结果加入消息
        5. 再次调用 vLLM 生成最终回复
        """
        messages = conversation_history or []
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": message})

        sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
        outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response = outputs[0]
        response_text = response.outputs[0].text

        # 从文本中解析 <tool_call> 标签
        tool_call = self._parse_tool_call(response_text)

        if tool_call:
            tool_name = tool_call["name"]
            tool_args = tool_call["arguments"]
            tool_result = self._execute_tool(tool_name, tool_args)

            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{hash(tool_name) % 1000000}",
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

            final_outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
            final_response = final_outputs[0].outputs[0].text
            return {"response": final_response, "messages": messages}
        else:
            return {"response": response_text, "messages": messages}

    def _parse_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        """从模型输出的文本中解析工具调用

        支持的格式：
        <tool_call>
        <function=code_executor>
        <parameter=language>
        python
        </parameter>
        <parameter=code>
        print("hello")
        </parameter>
        </tool_call>
        """
        # 模式1：完整 XML 格式 <function=xxx><parameter=yyy>zzz</parameter>
        pattern1 = r'<function=(\w+)>\s*<parameter=(\w+)>\s*([^<]+?)(?:</parameter>|$)'
        try:
            # 先匹配整个 tool_call 块
            tool_block = re.search(r'<tool_call>\s*(<function=\w+>.*?)</tool_call>', text, re.DOTALL)
            if tool_block:
                block_content = tool_block.group(1)
                func_match = re.search(r'<function=(\w+)>', block_content)
                param_match = re.search(r'<parameter=(\w+)>\s*([^<]+?)\s*(?:</parameter>|$)', block_content)
                if func_match and param_match:
                    return {
                        "name": func_match.group(1),
                        "arguments": {param_match.group(1): param_match.group(2).strip()}
                    }
        except Exception:
            pass

        # 模式2：检测是否提到要使用某个工具但格式不正确
        # 如果文本中提到"用xxx计算"、"执行代码"等但没有正确格式
        text_lower = text.lower()
        if 'code_executor' in text_lower or ('代码' in text and '执行' in text):
            # 尝试提取代码块
            code_blocks = re.findall(r'```python\s*(.*?)```', text, re.DOTALL)
            if code_blocks:
                code = code_blocks[0].strip()
                return {
                    "name": "code_executor",
                    "arguments": {"language": "python", "code": code}
                }

        return None

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, str]) -> Any:
        """执行工具"""
        if tool_name in self.tools:
            return self.tools[tool_name](**tool_args)
        return {"error": f"Unknown tool: {tool_name}"}

    def _chat(self, text: str) -> str:
        return f"chat: {text}"

    def _code_executor(self, language: str, code: str) -> str:
        """执行 Python 代码"""
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout or result.stderr
        except Exception as e:
            return str(e)

    def _document_reader(self, file_path: str) -> str:
        """读取文档"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return str(e)

    def _math_calc(self, expression: str) -> str:
        """计算数学表达式"""
        try:
            import ast
            import operator
            ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
            tree = ast.parse(expression, mode="eval")
            return str(eval(compile(tree, "<string>", "eval")))
        except Exception as e:
            return str(e)

    def _translator(self, text: str, source_lang: str, target_lang: str) -> str:
        return f"translated: {text}"

    def _weather(self, city: str) -> str:
        return f"weather in {city}: sunny, 25C"
