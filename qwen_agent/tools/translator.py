"""翻译工具"""
from typing import Dict, Any

def translator(text: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
    """
    翻译文本

    Args:
        text: 要翻译的文本
        source_lang: 源语言代码，如 "en", "zh"
        target_lang: 目标语言代码，如 "zh", "en"

    Returns:
        {"success": bool, "translated_text": str, "error": str}
    """
    try:
        import argostranslate

        available = argostranslate.get_installed_languages()
        source = None
        target = None

        for lang in available:
            if source_lang.lower() in lang.code.lower():
                source = lang
            if target_lang.lower() in lang.code.lower():
                target = lang

        if not source or not target:
            return {
                "success": False,
                "translated_text": "",
                "error": f"不支持的语言对: {source_lang} -> {target_lang}"
            }

        translator = argostranslate.translate.new_translator(source, target)
        result = translator.translate(text)
        return {"success": True, "translated_text": result, "error": ""}
    except Exception as e:
        return {"success": False, "translated_text": "", "error": str(e)}