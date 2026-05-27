from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Mock sentence_transformers module before importing
sys.modules['sentence_transformers'] = MagicMock()


def test_rag_retriever_retrieve():
    """Test RAGRetriever retrieve method with properly mocked dependencies"""
    from agent.rag_retriever import RAGRetriever

    # Create mock embedding model
    mock_st_instance = Mock()
    mock_st_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_st_instance.get_sentence_embedding_dimension.return_value = 4

    # Create retriever with pre-loaded embedding model
    retriever = RAGRetriever(embedding_model=mock_st_instance)

    # Properly mock vector_store.search to return actual results
    # distance 0.3 < (1 - 0.6) = 0.4, so this should pass the filter
    mock_vs = Mock()
    mock_vs.search.return_value = [
        {"text": "问：1+1\n答：2", "distance": 0.3}
    ]
    retriever.vector_store = mock_vs

    result = retriever.retrieve("计算问题", top_k=1)

    # Validate result is a list
    assert isinstance(result, list), f"Expected list, got {type(result)}"

    # Validate correct filtering
    assert len(result) == 1, f"Expected 1 result, got {len(result)}"
    assert result[0]["text"] == "问：1+1\n答：2"


def test_rag_retriever_retrieve_filters_by_threshold():
    """Test that retrieve properly filters results by similarity threshold"""
    from agent.rag_retriever import RAGRetriever

    mock_st_instance = Mock()
    mock_st_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_st_instance.get_sentence_embedding_dimension.return_value = 4

    mock_vs = Mock()
    # L2 distances for 384-dim embeddings typically range from 0 to ~50
    # similarity_threshold=0.6 -> threshold=(1-0.6)*10=4.0
    # distance < 4.0 passes, distance >= 4.0 fails
    mock_vs.search.return_value = [
        {"text": "相关问题", "distance": 2.5},  # passes filter (2.5 < 4.0)
        {"text": "不相关问题", "distance": 6.0},  # fails filter (6.0 >= 4.0)
    ]

    retriever = RAGRetriever(embedding_model=mock_st_instance, similarity_threshold=0.6)
    retriever.vector_store = mock_vs

    result = retriever.retrieve("查询", top_k=2)

    # Only the first result should pass the distance < 0.4 filter
    assert len(result) == 1, f"Expected 1 filtered result, got {len(result)}"
    assert result[0]["text"] == "相关问题"


def test_format_context():
    """Test format_context properly formats retrieved results"""
    from agent.rag_retriever import RAGRetriever

    retriever = RAGRetriever()

    results = [
        {"text": "问：1+1\n答：2", "distance": 0.3},
        {"text": "问：2+2\n答：4", "distance": 0.5},
    ]

    context = retriever.format_context(results)

    assert "相关历史经验：" in context
    assert "问：1+1" in context
    assert "问：2+2" in context
    assert "答：2" in context
    assert "答：4" in context


def test_format_context_empty():
    """Test format_context returns empty string for empty results"""
    from agent.rag_retriever import RAGRetriever

    retriever = RAGRetriever()
    context = retriever.format_context([])

    assert context == ""


def test_index_conversation():
    """Test index_conversation properly indexes QA pairs"""
    from agent.rag_retriever import RAGRetriever

    mock_st_instance = Mock()
    mock_st_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_st_instance.get_sentence_embedding_dimension.return_value = 4

    mock_vs = Mock()
    mock_vs.add = Mock()

    with patch('agent.rag_retriever.VectorStore', return_value=mock_vs):
        retriever = RAGRetriever(embedding_model=mock_st_instance)

        messages = [
            {"role": "user", "content": "1+1等于几"},
            {"role": "assistant", "content": "2"},
        ]

        retriever.index_conversation(messages)

        # Verify vector_store.add was called with embeddings and texts
        mock_vs.add.assert_called_once()


def test_save_index():
    """Test save_index creates directory and saves index"""
    from agent.rag_retriever import RAGRetriever

    mock_st_instance = Mock()
    mock_st_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_st_instance.get_sentence_embedding_dimension.return_value = 4

    mock_vs = Mock()
    mock_vs.save = Mock()

    with patch('agent.rag_retriever.VectorStore', return_value=mock_vs):
        retriever = RAGRetriever(embedding_model=mock_st_instance, index_dir="/tmp/test_index")
        retriever.vector_store = mock_vs

        retriever.save_index()

        mock_vs.save.assert_called_once()
        call_args = mock_vs.save.call_args[0][0]
        assert "vector_store.faiss" in call_args


def test_load_index():
    """Test load_index properly loads index from disk"""
    from agent.rag_retriever import RAGRetriever

    mock_st_instance = Mock()
    mock_st_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_st_instance.get_sentence_embedding_dimension.return_value = 4

    mock_vs = Mock()
    mock_vs.load = Mock()

    with patch('agent.rag_retriever.VectorStore', return_value=mock_vs):
        with patch('os.path.exists', return_value=True):
            retriever = RAGRetriever(embedding_model=mock_st_instance, index_dir="/tmp/test_index")

            retriever.load_index()

            mock_vs.load.assert_called_once()
            call_args = mock_vs.load.call_args[0][0]
            assert "vector_store.faiss" in call_args


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])