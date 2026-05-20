"""Agent 核心逻辑"""
import json
import re
from typing import Dict, Any, List, Optional
import vllm
from vllm import SamplingParams

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
            "weather": self._weather,
        }

    def run(self, message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        messages = conversation_history or []
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": message})

        sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
        outputs = self.llm.chat(messages=messages, sampling_params=sampling_params)
        response = outputs[0]
        response_text = response.outputs[0].text

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
        try:
            pattern = r'<tool_call>\s*<function=(\w+)>\s*<parameter=(\w+)>\s*([^<]+)\s*</tool_call>'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return {
                    "name": match.group(1),
                    "arguments": match.group(3).strip()
                }
        except Exception:
            pass
        return None

    def _execute_tool(self, tool_name: str, tool_args: str) -> Any:
        if tool_name in self.tools:
            return self.tools[tool_name](tool_args)
        return {"error": f"Unknown tool: {tool_name}"}

    def _chat(self, text: str) -> str:
        return f"chat: {text}"

    def _code_executor(self, code: str) -> str:
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

    def _document_reader(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return str(e)

    def _math_calc(self, expr: str) -> str:
        try:
            import ast
            import operator
            ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
            tree = ast.parse(expr, mode="eval")
            return str(eval(compile(tree, "<string>", "eval")))
        except Exception as e:
            return str(e)

    def _translator(self, text: str) -> str:
        return f"translated: {text}"

    def _weather(self, city: str) -> str:
        return f"weather in {city}: sunny, 25C"
