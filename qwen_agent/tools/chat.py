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