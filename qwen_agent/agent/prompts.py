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