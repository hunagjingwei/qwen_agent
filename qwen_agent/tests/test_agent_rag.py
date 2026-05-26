"""Test RAG integration in Agent"""
from unittest.mock import patch, Mock, MagicMock

# Mock vllm and sentence_transformers before importing agent.core
import sys
sys.modules['vllm'] = Mock()
sys.modules['sentence_transformers'] = MagicMock()

from agent.core import Agent


def test_agent_build_messages_with_rag():
    """Test that _build_messages integrates RAG retrieval"""
    with patch('agent.core.vllm') as mock_vllm:
        with patch('agent.core.RAGRetriever') as mock_rag_class:
            mock_vllm.LLM.return_value = Mock()
            agent = Agent(model_path="/tmp/model")

            agent.rag_retriever = Mock()
            agent.rag_retriever.retrieve.return_value = [
                {"text": "问：1+1\n答：2", "distance": 0.3}
            ]
            agent.rag_retriever.format_context.return_value = "相关历史经验：\n1. 问：1+1\n答：2"

            messages = agent._build_messages("计算 1+1", [])
            assert any("相关历史经验" in str(m) for m in messages), f"Expected RAG context in messages, got: {messages}"