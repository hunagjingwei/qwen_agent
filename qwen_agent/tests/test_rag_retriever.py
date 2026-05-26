from unittest.mock import Mock, patch, MagicMock
import sys


def test_rag_retriever_retrieve():
    """Test RAGRetriever retrieve method with mocked dependencies"""
    # Create mock for sentence_transformers module
    mock_st_module = MagicMock()
    mock_st_instance = Mock()
    mock_st_instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4]]
    mock_st_instance.get_sentence_embedding_dimension.return_value = 4
    mock_st_module.SentenceTransformer.return_value = mock_st_instance

    # Patch sys.modules before importing rag_retriever
    with patch.dict(sys.modules, {'sentence_transformers': mock_st_module}):
        # Clear any cached imports
        for mod in list(sys.modules.keys()):
            if mod.startswith('agent'):
                del sys.modules[mod]

        from agent.rag_retriever import RAGRetriever

        # Create retriever with pre-loaded embedding model
        retriever = RAGRetriever(embedding_model=mock_st_instance)
        retriever.vector_store = Mock()

        result = retriever.retrieve("计算问题", top_k=1)
        assert isinstance(result, list)