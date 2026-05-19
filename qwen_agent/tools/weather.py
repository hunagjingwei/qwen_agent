"""天气查询工具"""
from typing import Dict, Any
import urllib.request
import urllib.parse
import json

API_KEY = "79e53696133a4ea587081357261905"

def weather(city: str) -> Dict[str, Any]:
    """
    查询城市天气

    Args:
        city: 城市名称，如 "广州", "Beijing", "Guangzhou"

    Returns:
        {"success": bool, "weather": str, "error": str}
    """
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={urllib.parse.quote(city)}&aqi=no"

        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        location = data["location"]["name"]
        country = data["location"]["country"]
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        humidity = data["current"]["humidity"]
        wind_kph = data["current"]["wind_kph"]
        feelslike_c = data["current"]["feelslike_c"]

        weather_info = f"""天气信息 - {location}, {country}:
- 温度: {temp_c}°C (体感 {feelslike_c}°C)
- 天气状况: {condition}
- 湿度: {humidity}%
- 风速: {wind_kph} km/h"""

        return {"success": True, "weather": weather_info, "error": ""}

    except urllib.error.URLError as e:
        return {"success": False, "weather": "", "error": f"网络错误: {str(e)}"}
    except KeyError as e:
        return {"success": False, "weather": "", "error": f"数据解析错误: {str(e)}"}
    except Exception as e:
        return {"success": False, "weather": "", "error": f"未知错误: {str(e)}"}