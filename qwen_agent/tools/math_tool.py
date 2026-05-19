"""数学计算工具"""
from typing import Dict, Any
import sympy
from sympy import symbols, sympify, evaluate

def math_calc(expression: str) -> Dict[str, Any]:
    """
    计算数学表达式

    Args:
        expression: 数学表达式字符串，如 "2+3*4" 或 "x**2 + 2*x + 1"

    Returns:
        {"success": bool, "result": str, "error": str}
    """
    try:
        result = sympify(expression)
        return {"success": True, "result": str(result), "error": ""}
    except Exception as e:
        return {"success": False, "result": "", "error": f"表达式解析错误: {str(e)}"}