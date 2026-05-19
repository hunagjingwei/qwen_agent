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