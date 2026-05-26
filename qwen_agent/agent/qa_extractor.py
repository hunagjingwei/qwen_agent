"""QA 提取器 - 从对话历史中提取问答对"""


class QAExtractor:
    """从对话历史中提取 QA 对"""

    def __init__(self):
        pass

    def extract(self, messages: list) -> list:
        """
        从消息列表中提取 QA 对

        Args:
            messages: 消息列表，每条消息包含 role 和 content

        Returns:
            QA 对列表，每条包含 question, answer, metadata
        """
        qa_pairs = []
        current_q = None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                current_q = content
            elif role == "assistant" and current_q:
                qa_pairs.append({
                    "question": current_q,
                    "answer": content,
                    "metadata": {}
                })
                current_q = None

        return qa_pairs

    def format_for_rag(self, qa_pairs: list) -> list:
        """将 QA 对格式化为检索文本"""
        formatted = []
        for qa in qa_pairs:
            text = f"问：{qa['question']}\n答：{qa['answer']}"
            formatted.append(text)
        return formatted