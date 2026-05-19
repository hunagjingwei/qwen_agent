"""代码执行工具"""
import sys
import io
import traceback
import signal
from typing import Dict, Any

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("代码执行超时 (5秒)")

def code_executor(language: str, code: str) -> Dict[str, Any]:
    """
    执行 Python 或 Shell 代码

    Args:
        language: "python" 或 "shell"
        code: 要执行的代码

    Returns:
        {"success": bool, "output": str, "error": str}
    """
    if language == "python":
        return _execute_python(code)
    elif language == "shell":
        return _execute_shell(code)
    else:
        return {"success": False, "output": "", "error": f"不支持的语言: {language}"}

def _execute_python(code: str) -> Dict[str, Any]:
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

    try:
        exec(code, {"__name__": "__main__"})
        output = sys.stdout.getvalue()
        signal.alarm(0)
        return {"success": True, "output": output, "error": ""}
    except TimeoutError:
        return {"success": False, "output": "", "error": "代码执行超时 (5秒)"}
    except Exception:
        error = traceback.format_exc()
        return {"success": False, "output": sys.stdout.getvalue(), "error": error}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        signal.alarm(0)

def _execute_shell(code: str) -> Dict[str, Any]:
    import subprocess
    try:
        result = subprocess.run(
            code, shell=True, capture_output=True, text=True, timeout=5
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else ""
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "output": "", "error": "代码执行超时 (5秒)"}