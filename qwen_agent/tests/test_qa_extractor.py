def test_extract_qa_pairs():
    from agent.qa_extractor import QAExtractor

    extractor = QAExtractor()
    messages = [
        {"role": "user", "content": "计算 1+1"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "计算 2+2"},
        {"role": "assistant", "content": "4"},
    ]
    qa_pairs = extractor.extract(messages)
    assert len(qa_pairs) == 2
    assert "1+1" in qa_pairs[0]["question"]
    assert "2" in qa_pairs[0]["answer"]


def test_format_for_rag():
    from agent.qa_extractor import QAExtractor

    extractor = QAExtractor()
    qa_pairs = [
        {"question": "1+1=?", "answer": "2", "metadata": {}},
        {"question": "2+2=?", "answer": "4", "metadata": {}}
    ]
    formatted = extractor.format_for_rag(qa_pairs)
    assert len(formatted) == 2
    assert "问：1+1=?" in formatted[0]
    assert "答：2" in formatted[0]