"""Agent 核心模块 - 包含 HistoryStorage, HistoryRetriever 和 Agent 类"""
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import vllm

from .prompts import SYSTEM_PROMPT, CHAT_TEMPLATE
from .functions import FUNCTIONS
from .rag_retriever import RAGRetriever


class HistoryStorage:
    """对话历史存储类 - 用于持久化保存和加载对话历史"""

    def __init__(self, db_path: str = "./conversation_history.db"):
        """
        初始化历史存储

        Args:
            db_path: SQLite 数据库路径
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                messages TEXT NOT NULL,
                metadata TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON conversations(created_at)
        """)
        conn.commit()
        conn.close()

    def save(self, session_id: str, messages: List[Dict[str, Any]], metadata: Optional[Dict] = None) -> int:
        """
        保存对话历史

        Args:
            session_id: 会话 ID
            messages: 消息列表
            metadata: 元数据（可选）

        Returns:
            插入的对话 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        messages_json = json.dumps(messages, ensure_ascii=False)
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        cursor.execute("""
            INSERT INTO conversations (session_id, created_at, updated_at, messages, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, now, now, messages_json, metadata_json))

        conversation_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return conversation_id

    def update(self, conversation_id: int, messages: List[Dict[str, Any]], metadata: Optional[Dict] = None):
        """
        更新对话历史

        Args:
            conversation_id: 对话 ID
            messages: 消息列表
            metadata: 元数据（可选）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        messages_json = json.dumps(messages, ensure_ascii=False)
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None

        cursor.execute("""
            UPDATE conversations
            SET updated_at = ?, messages = ?, metadata = ?
            WHERE id = ?
        """, (now, messages_json, metadata_json, conversation_id))

        conn.commit()
        conn.close()

    def load(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        加载指定会话的对话历史

        Args:
            session_id: 会话 ID
            limit: 返回的最近对话数量

        Returns:
            对话历史列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT messages, created_at, metadata
            FROM conversations
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, limit))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            messages = json.loads(row[0])
            created_at = row[1]
            metadata = json.loads(row[2]) if row[2] else None
            results.append({
                "messages": messages,
                "created_at": created_at,
                "metadata": metadata
            })

        return list(reversed(results))

    def search(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        搜索包含关键词的对话历史

        Args:
            keyword: 搜索关键词
            limit: 返回结果数量限制

        Returns:
            匹配的对话历史列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, messages, created_at, metadata
            FROM conversations
            WHERE messages LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{keyword}%", limit))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            session_id = row[0]
            messages = json.loads(row[1])
            created_at = row[2]
            metadata = json.loads(row[3]) if row[3] else None
            results.append({
                "session_id": session_id,
                "messages": messages,
                "created_at": created_at,
                "metadata": metadata
            })

        return results

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        列出所有会话

        Args:
            limit: 返回数量限制

        Returns:
            会话列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, created_at, updated_at, metadata
            FROM conversations
            ORDER BY updated_at DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "session_id": row[0],
                "created_at": row[1],
                "updated_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else None
            })

        return results

    def delete(self, session_id: str) -> int:
        """
        删除指定会话的所有对话

        Args:
            session_id: 会话 ID

        Returns:
            删除的对话数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM conversations WHERE session_id = ?
        """, (session_id,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count


class HistoryRetriever:
    """对话历史检索类 - 用于搜索和检索历史对话"""

    def __init__(self, storage: HistoryStorage):
        """
        初始化历史检索器

        Args:
            storage: HistoryStorage 实例
        """
        self.storage = storage

    def retrieve_by_keyword(self, keyword: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        通过关键词检索对话

        Args:
            keyword: 搜索关键词
            max_results: 最大返回结果数

        Returns:
            匹配的对话列表
        """
        return self.storage.search(keyword, limit=max_results)

    def retrieve_by_session(self, session_id: str, max_messages: int = 50) -> List[Dict[str, Any]]:
        """
        检索指定会话的对话历史

        Args:
            session_id: 会话 ID
            max_messages: 最大消息数

        Returns:
            对话历史列表
        """
        return self.storage.load(session_id, limit=max_messages)

    def retrieve_recent(self, max_sessions: int = 10) -> List[Dict[str, Any]]:
        """
        检索最近的会话

        Args:
            max_sessions: 最大会话数

        Returns:
            最近会话列表
        """
        return self.storage.list_sessions(limit=max_sessions)

    def retrieve_similar(
        self,
        query: str,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        检索相似的对话（基于关键词匹配）

        Args:
            query: 查询文本
            max_results: 最大返回结果数
            similarity_threshold: 相似度阈值（0-1）

        Returns:
            相似对话列表
        """
        query_keywords = self._extract_keywords(query)

        if not query_keywords:
            return []

        all_sessions = self.storage.list_sessions(limit=100)
        scored_results = []

        for session in all_sessions:
            session_keywords = self._extract_keywords_from_messages(session.get("messages", []))
            similarity = self._calculate_similarity(query_keywords, session_keywords)

            if similarity >= similarity_threshold:
                session["similarity"] = similarity
                scored_results.append(session)

        scored_results.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_results[:max_results]

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词"""
        words = re.findall(r'\w+', text.lower())
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "什么", "怎么", "为什么", "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must"}
        return [w for w in words if len(w) > 1 and w not in stop_words]

    def _extract_keywords_from_messages(self, messages: List[Dict[str, Any]]) -> List[str]:
        """从消息列表中提取关键词"""
        all_text = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                all_text.append(str(msg["content"]))
            elif isinstance(msg, dict):
                all_text.append(str(msg))
            else:
                all_text.append(str(msg))

        combined_text = " ".join(all_text)
        return self._extract_keywords(combined_text)

    def _calculate_similarity(self, keywords1: List[str], keywords2: List[str]) -> float:
        """计算两组关键词的相似度（Jaccard 相似系数）"""
        if not keywords1 or not keywords2:
            return 0.0

        set1 = set(keywords1)
        set2 = set(keywords2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0


class Agent:
    """Agent 主类 - 负责对话、工具调用和历史管理"""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.9,
        history_storage: Optional[HistoryStorage] = None
    ):
        """
        初始化 Agent

        Args:
            model_path: 模型路径
            tensor_parallel_size: 张量并行大小
            max_model_len: 最大模型长度
            gpu_memory_utilization: GPU 内存利用率
            history_storage: 可选的历史存储实例
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

        self.llm = vllm.LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="float16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.sampling_params = vllm.SamplingParams(temperature=0.7, max_tokens=2048)

        self.history_storage = history_storage or HistoryStorage()
        self.history_retriever = HistoryRetriever(self.history_storage)
        # RAG 检索器
        self.rag_retriever = RAGRetriever()

        # 尝试加载已有索引
        try:
            self.rag_retriever.load_index()
            print("[INFO] RAG index loaded from disk")
        except Exception as e:
            print(f"[INFO] No existing RAG index, will build new one: {e}")

        self.tool_registry = {
            "chat": self._chat_tool,
            "code_executor": self._code_executor_tool,
            "document_reader": self._document_reader_tool,
            "math_tool": self._math_tool,
            "translator": self._translator_tool,
            "weather": self._weather_tool,
            # 数学函数直接映射到 _math_tool
            "derivative": self._math_tool,
            "calculate": self._math_tool,
            "indefinite_integral": self._math_tool,
            "definite_integral": self._math_tool,
            "limit_calc": self._math_tool,
            "solve_equation": self._math_tool,
            "solve_inequality": self._math_tool,
            "find_extrema": self._math_tool,
            "find_inflection_points": self._math_tool,
            "analyze_monotonic": self._math_tool,
            "compound_interest": self._math_tool,
            "simple_interest": self._math_tool,
            "loan_repayment": self._math_tool,
            "sequence_sum": self._math_tool,
            "probability_calc": self._math_tool,
            "triangle_area": self._math_tool,
            "circle_area": self._math_tool,
            "rectangle_area": self._math_tool,
            "sphere_volume": self._math_tool,
            "cylinder_volume": self._math_tool,
            "cone_volume": self._math_tool,
            "cuboid_volume": self._math_tool,
            "matrix_operation": self._math_tool,
            "vector_operation": self._math_tool,
        }

    def run(self, user_message: str, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        运行 Agent 处理用户消息

        Args:
            user_message: 用户输入的消息
            conversation_history: 对话历史列表

        Returns:
            包含 response 和 messages 的字典
        """
        messages = self._build_messages(user_message, conversation_history)

        max_turns = 10
        turn_count = 0
        tool_executed = False

        while turn_count < max_turns:
            turn_count += 1

            response_text = self._generate_response(messages)

            # 检测并过滤异常长的响应（可能是模型生成垃圾输出）
            if len(response_text) > 5000 or response_text.count('</parameter>') > 20:
                print(f"[WARN] Response too long or malformed ({len(response_text)} chars, </parameter> x{response_text.count('</parameter>')}), truncating", file=sys.stderr)
                import re
                # 如果有完整的 </tool_call>，保留到它为止
                first_end = response_text.find('</tool_call>')
                if first_end != -1 and first_end < 4000:
                    response_text = response_text[:first_end + len('</tool_call>')]
                else:
                    # 没有完整闭合，手动补全最后一个参数块
                    # 找到最后一个 </parameter> 之后的内容并截断
                    last_param_end = response_text.rfind('</parameter>')
                    if last_param_end != -1 and last_param_end > 100:
                        response_text = response_text[:last_param_end + len('</parameter>')]
                        # 补全 tool_call 结构
                        if '<tool_call>' in response_text and '</tool_call>' not in response_text:
                            response_text += '</tool_call>'
                # 清理重复的 </parameter> 标签
                response_text = re.sub(r'(</parameter>\s*)+', '</parameter>', response_text)
                response_text = response_text.strip()

            assistant_message = {
                "role": "assistant",
                "content": response_text
            }
            messages.append(assistant_message)

            tool_calls = self._parse_tool_call(response_text)

            if not tool_calls:
                # 没有更多工具调用，结束
                return {
                    "response": response_text,
                    "messages": messages,
                    "tool_calls": []
                }

            # 执行工具
            tool_executed = True
            tool_results = []
            for tool_call in tool_calls:
                result = self._execute_tool(tool_call)
                tool_results.append(result)

                tool_message = {
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "tool_call_id": tool_call.get("id", "")
                }
                messages.append(tool_message)

            # 如果已经执行过工具，不要再循环，直接返回
            if tool_executed:
                print(f"[DEBUG] tool_executed=True, tool_results count={len(tool_results)}", file=sys.stderr)
                if tool_results:
                    tr0 = tool_results[0]
                    print(f"[DEBUG] tool_results[0] keys={tr0.keys()}", file=sys.stderr)
                    print(f"[DEBUG] success={tr0.get('success')}, output length={len(tr0.get('output', ''))}", file=sys.stderr)

                # 检查工具执行结果
                if tool_results and tool_results[0].get("success"):
                    # 兼容 output (code_executor) 和 result (math_tool) 两种格式
                    output = tool_results[0].get("output") or tool_results[0].get("result", "")
                    # 对于其他格式（如 area, volume, monotonic_intervals 等），转为字符串
                    if not output:
                        # 提取有意义的值（按优先级排序）
                        for key in ['area', 'volume', 'sum', 'total_amount', 'monthly_payment',
                                    'monotonic_intervals', 'critical_points', 'extrema',
                                    'inflection_points', 'derivative', 'limit', 'solution', 'solutions',
                                    'result', 'output']:
                            if key in tool_results[0]:
                                val = tool_results[0][key]
                                if isinstance(val, list):
                                    output = ', '.join(str(v) for v in val)
                                else:
                                    output = str(val)
                                break
                    print(f"[DEBUG] checking output, is truthy={bool(output)}, output={repr(output)[:50]}", file=sys.stderr)
                    if output:
                        print(f"[DEBUG] returning DIRECT OUTPUT", file=sys.stderr)
                        # 直接使用工具输出作为回答
                        return {
                            "response": output,
                            "messages": messages,
                            "tool_calls": tool_calls
                        }
                print(f"[DEBUG] falling through to _generate_response", file=sys.stderr)
                # 让模型根据工具结果生成最终回答
                response_text = self._generate_response(messages)
                return {
                    "response": response_text,
                    "messages": messages,
                    "tool_calls": tool_calls
                }

            if turn_count >= max_turns:
                return {
                    "response": "已达到最大对话轮次限制",
                    "messages": messages,
                    "tool_calls": tool_calls
                }

        return {
            "response": response_text,
            "messages": messages,
            "tool_calls": tool_calls
        }

    def _build_messages(self, user_message: str, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """构建消息列表"""
        messages = []

        messages.append({
            "role": "system",
            "content": SYSTEM_PROMPT
        })

        # RAG 检索相关历史经验
        if self.rag_retriever:
            retrieved = self.rag_retriever.retrieve(user_message, top_k=3)
            if retrieved:
                rag_context = self.rag_retriever.format_context(retrieved)
                messages.append({
                    "role": "system",
                    "content": f"{rag_context}\n\n请参考上述历史经验回答用户问题。"
                })

        messages.extend(conversation_history)

        messages.append({
            "role": "user",
            "content": user_message
        })

        return messages

    def _generate_response(self, messages: List[Dict[str, Any]]) -> str:
        """调用模型生成响应"""
        prompt = self._build_prompt(messages)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text

    def _build_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """构建 ChatML 格式的提示词"""
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted_messages.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_messages.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_messages.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "tool":
                formatted_messages.append(f"<|im_start|>tool\n{content}<|im_end|>")

        prompt = "\n".join(formatted_messages) + "\n<|im_start|>assistant\n"
        return prompt

    def _parse_tool_call(self, response_text: str) -> List[Dict[str, Any]]:
        """
        解析模型输出中的工具调用

        Args:
            response_text: 模型响应文本

        Returns:
            工具调用列表
        """
        tool_calls = []

        # 清理响应文本：移除重复的 XML 垃圾
        # 只保留第一个完整的 <tool_call>...</tool_call> 块
        first_tool_call_end = response_text.find('</tool_call>')
        if first_tool_call_end != -1:
            # 截取到第一个 </tool_call> 之后，再找下一个 <tool_call> 开始
            response_text = response_text[:first_tool_call_end + len('</tool_call>')]
            # 移除可能的重复垃圾
            response_text = re.sub(r'(</parameter>\s*)+', '</parameter>', response_text)
            response_text = re.sub(r'(</tool_call>\s*)+', '</tool_call>', response_text)

        # 先提取每个 <tool_call>...</tool_call> 块
        tool_call_blocks = re.findall(
            r'<tool_call>\s*<function=(\w+)>(.*?)</tool_call>',
            response_text,
            re.DOTALL
        )

        print(f"[DEBUG] _parse_tool_call: found {len(tool_call_blocks)} tool_call blocks", file=sys.stderr)

        if not tool_call_blocks:
            # 尝试备用格式
            xml_pattern = r'<function>(\w+)</function>\s*<parameter>(\w+)</parameter>\s*<value>(.*?)</value>'
            xml_matches = re.findall(xml_pattern, response_text, re.DOTALL)
            if xml_matches:
                for match in xml_matches:
                    tool_calls.append({
                        "name": match[0],
                        "arguments": {match[1]: match[2].strip()}
                    })
            return tool_calls

        for func_name, params_block in tool_call_blocks:
            # 在每个块中提取所有 <parameter=name>value</parameter>
            param_matches = re.findall(
                r'<parameter=(\w+)>\s*(.*?)\s*</parameter>',
                params_block,
                re.DOTALL
            )

            if param_matches:
                arguments = {param_name: param_value.strip() for param_name, param_value in param_matches}
                tool_calls.append({
                    "name": func_name,
                    "arguments": arguments
                })

        return tool_calls

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行工具调用

        Args:
            tool_call: 工具调用字典，包含 name 和 arguments

        Returns:
            工具执行结果
        """
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        print(f"[DEBUG] _execute_tool called: tool_name={tool_name}, arguments keys={list(arguments.keys())}", file=sys.stderr)

        if tool_name not in self.tool_registry:
            return {
                "success": False,
                "error": f"未知工具: {tool_name}",
                "tool_name": tool_name
            }

        try:
            tool_func = self.tool_registry[tool_name]
            print(f"[DEBUG] calling tool_func with tool_name={tool_name}, arguments: {list(arguments.keys())}", file=sys.stderr)
            # 只有 math_tool 及其别名需要 func_name 参数
            if tool_name == "math_tool" or tool_name in [
                "derivative", "calculate", "indefinite_integral", "definite_integral",
                "limit_calc", "solve_equation", "solve_inequality", "find_extrema",
                "find_inflection_points", "analyze_monotonic", "compound_interest",
                "simple_interest", "loan_repayment", "sequence_sum", "probability_calc",
                "triangle_area", "circle_area", "rectangle_area", "sphere_volume",
                "cylinder_volume", "cone_volume", "cuboid_volume", "matrix_operation",
                "vector_operation"
            ]:
                result = tool_func(func_name=tool_name, **arguments)
            else:
                result = tool_func(**arguments)
            print(f"[DEBUG] tool_func returned result keys={result.keys()}, success={result.get('success')}", file=sys.stderr)
            result["tool_name"] = tool_name
            return result
        except Exception as e:
            import traceback
            print(f"[DEBUG] _execute_tool exception: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }

    def _chat_tool(self, message: str) -> Dict[str, Any]:
        """通用对话工具"""
        from ..tools.chat import chat

        try:
            result = chat(message, self.model_path, CHAT_TEMPLATE)
            return {
                "success": True,
                "response": result,
                "error": ""
            }
        except Exception as e:
            return {
                "success": False,
                "response": "",
                "error": str(e)
            }

    def _code_executor_tool(self, language: str, code: str) -> Dict[str, Any]:
        """代码执行工具"""
        import sys
        sys.path.insert(0, '/home/wujie/LLM_project/qwen-3.5')
        from qwen_agent.tools.code import code_executor

        print(f"[DEBUG] _code_executor_tool called: language={language}", file=sys.stderr)
        print(f"[DEBUG] code length={len(code)}, code preview: {code[:200]}", file=sys.stderr)

        try:
            result = code_executor(language, code)
            print(f"[DEBUG] code_executor result success={result.get('success')}", file=sys.stderr)
            print(f"[DEBUG] code_executor output length={len(result.get('output', ''))}", file=sys.stderr)
            print(f"[DEBUG] code_executor error: {result.get('error', '')[:200] if result.get('error') else 'none'}", file=sys.stderr)

            if result["success"]:
                return {
                    "success": True,
                    "output": result["output"],
                    "error": ""
                }
            else:
                return {
                    "success": False,
                    "output": result["output"],
                    "error": result["error"]
                }
        except Exception as e:
            print(f"[DEBUG] code_executor exception: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    def _document_reader_tool(
        self,
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """文档读取工具"""
        from ..tools.document import document_reader

        try:
            result = document_reader(file_path, start_line, end_line)
            if result["success"]:
                return {
                    "success": True,
                    "content": result["content"],
                    "error": ""
                }
            else:
                return {
                    "success": False,
                    "content": "",
                    "error": result["error"]
                }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": str(e)
            }

    def _math_tool(self, func_name: str = None, **kwargs) -> Dict[str, Any]:
        """统一数学工具入口"""
        from qwen_agent.tools.math_tool import math_tool

        print(f"[DEBUG] _math_tool called: func_name={func_name}, kwargs={kwargs}", file=sys.stderr)
        try:
            result = math_tool(func_name, **kwargs)
            print(f"[DEBUG] math_tool returned: {result}", file=sys.stderr)
            return result
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _translator_tool(self, text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
        """翻译工具"""
        from ..tools.translator import translator

        try:
            result = translator(text, source_lang, target_lang)
            if result["success"]:
                return {
                    "success": True,
                    "translated_text": result["translated_text"],
                    "error": ""
                }
            else:
                return {
                    "success": False,
                    "translated_text": "",
                    "error": result["error"]
                }
        except Exception as e:
            return {
                "success": False,
                "translated_text": "",
                "error": str(e)
            }

    def _weather_tool(self, city: str) -> Dict[str, Any]:
        """天气查询工具"""
        from ..tools.weather import weather

        try:
            result = weather(city)
            if result["success"]:
                return {
                    "success": True,
                    "weather": result["weather"],
                    "error": ""
                }
            else:
                return {
                    "success": False,
                    "weather": "",
                    "error": result["error"]
                }
        except Exception as e:
            return {
                "success": False,
                "weather": "",
                "error": str(e)
            }

    def save_conversation(self, session_id: str, messages: List[Dict[str, Any]], metadata: Optional[Dict] = None):
        """保存对话到历史存储"""
        self.history_storage.save(session_id, messages, metadata)

        # 自动索引 QA 对到 RAG
        if self.rag_retriever:
            self.rag_retriever.index_conversation(messages)
            self.rag_retriever.save_index()

    def load_conversation(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """加载对话历史"""
        return self.history_storage.load(session_id, limit)

    def search_history(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索历史对话"""
        return self.history_retriever.retrieve_by_keyword(keyword, limit)

    def retrieve_similar_conversations(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """检索相似对话"""
        return self.history_retriever.retrieve_similar(query, max_results)
