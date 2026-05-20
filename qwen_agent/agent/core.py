"""Agent 核心逻辑"""
import json
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
        """运行 Agent 处理用户消息"""
        messages = conversation_history or []
        # 添加系统提示词
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": message})

        # 调用 vLLM chat（支持 Function Calling）
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=2048,
        )
        outputs = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            tools=FUNCTIONS,
        )
        response = outputs[0]

        # 检查是否触发了工具调用
        # vLLM 中 finish_reason 在 outputs[0] 上
        finish_reason = response.outputs[0].finish_reason if hasattr(response.outputs[0], 'finish_reason') else None
        if finish_reason == "tool_calls":
            tool_calls = response.outputs[0].tool_calls
            results = []

            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_args = tool_call.args

                # 执行工具
                tool_result = self._execute_tool(tool_name, tool_args)

                # 将工具调用和结果加入消息
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args)
                            }
                        }
                    ]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })
                results.append({"tool": tool_name, "result": tool_result})

            # 再次调用模型生成最终回复
            final_outputs = self.llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                tools=FUNCTIONS,
            )
            final_response = final_outputs[0].outputs[0].text
            return {"response": final_response, "messages": messages}
        else:
            return {"response": response.outputs[0].text, "messages": messages}

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

    def _weather(self, city: str) -> Dict[str, Any]:
        """天气查询工具"""
        from tools.weather import weather
        return weather(city)