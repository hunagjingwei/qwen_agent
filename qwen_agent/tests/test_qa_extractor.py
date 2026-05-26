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