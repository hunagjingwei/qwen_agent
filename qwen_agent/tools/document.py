"""文档读取工具"""
import os
from typing import Dict, Any, Optional

def document_reader(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None
) -> Dict[str, Any]:
    """
    读取文档内容

    Args:
        file_path: 文件路径
        start_line: 起始行号 (从1开始)
        end_line: 结束行号 (从1开始)

    Returns:
        {"success": bool, "content": str, "error": str}
    """
    if not os.path.exists(file_path):
        return {"success": False, "content": "", "error": f"文件不存在: {file_path}"}

    if file_path.endswith(".pdf"):
        return _read_pdf(file_path, start_line, end_line)
    else:
        return _read_text(file_path, start_line, end_line)

def _read_text(
    file_path: str,
    start_line: Optional[int],
    end_line: Optional[int]
) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start = (start_line - 1) if start_line else 0
        end = end_line if end_line else len(lines)

        content = "".join(lines[start:end])
        return {"success": True, "content": content, "error": ""}
    except Exception as e:
        return {"success": False, "content": "", "error": str(e)}

def _read_pdf(
    file_path: str,
    start_line: Optional[int],
    end_line: Optional[int]
) -> Dict[str, Any]:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)

        start = (start_line - 1) if start_line else 0
        end = end_line if end_line else len(reader.pages)

        content = []
        for i in range(start, min(end, len(reader.pages))):
            content.append(reader.pages[i].extract_text())

        return {"success": True, "content": "\n".join(content), "error": ""}
    except Exception as e:
        return {"success": False, "content": "", "error": str(e)}