"""Function Calling 函数定义"""
from typing import List, Dict, Any

FUNCTIONS = [
    {
        "name": "chat",
        "description": "通用对话，直接返回模型回复",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "用户消息"
                }
            },
            "required": ["message"]
        }
    },
    {
        "name": "code_executor",
        "description": "执行 Python 或 Shell 代码",
        "parameters": {
            "type": "object",
            "properties": {
                "language": {
                    "type": "string",
                    "enum": ["python", "shell"],
                    "description": "编程语言"
                },
                "code": {
                    "type": "string",
                    "description": "要执行的代码"
                }
            },
            "required": ["language", "code"]
        }
    },
    {
        "name": "document_reader",
        "description": "读取文档内容",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "start_line": {
                    "type": "integer",
                    "description": "起始行号 (从1开始)"
                },
                "end_line": {
                    "type": "integer",
                    "description": "结束行号 (从1开始)"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "math_calc",
        "description": "计算数学表达式",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 2+3*4"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "translator",
        "description": "翻译文本",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "要翻译的文本"
                },
                "source_lang": {
                    "type": "string",
                    "description": "源语言代码，如 en, zh"
                },
                "target_lang": {
                    "type": "string",
                    "description": "目标语言代码，如 zh, en"
                }
            },
            "required": ["text", "source_lang", "target_lang"]
        }
    },
    {
        "name": "weather",
        "description": "查询城市天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如 '广州', 'Beijing', 'Guangzhou'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "savings_find_year",
        "description": "逐月或逐年存款计算，多少年后达到目标金额",
        "parameters": {
            "type": "object",
            "properties": {
                "annual_deposit": {
                    "type": "number",
                    "description": "每年存款金额(如 150000)"
                },
                "annual_rate": {
                    "type": "number",
                    "description": "年利率(如 0.03)"
                },
                "target": {
                    "type": "number",
                    "description": "目标金额(如 1000000)"
                },
                "initial_principal": {
                    "type": "number",
                    "description": "初始本金(已有存款，如 400000)"
                },
                "monthly_deposit": {
                    "type": "number",
                    "description": "每月存款金额(如 15000)，与 annual_deposit 二选一"
                }
            },
            "required": ["annual_rate", "target"]
        }
    },
    {
        "name": "savings_accumulation",
        "description": "逐年存款计算，n年后的存款总额",
        "parameters": {
            "type": "object",
            "properties": {
                "annual_deposit": {
                    "type": "number",
                    "description": "每年存款金额(如 150000)"
                },
                "annual_rate": {
                    "type": "number",
                    "description": "年利率(如 0.03)"
                },
                "years": {
                    "type": "integer",
                    "description": "存款年数(如 10)"
                }
            },
            "required": ["annual_deposit", "annual_rate", "years"]
        }
    }
]