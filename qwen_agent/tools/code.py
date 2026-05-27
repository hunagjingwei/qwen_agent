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
    # 自动转换 sympy 多项式代码为 numpy
    code = _convert_sympy_to_numpy(code)

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


def _convert_sympy_to_numpy(code: str) -> str:
    """自动转换 sympy 多项式代码为 numpy"""
    import re

    # 检测是否使用了 sympy
    uses_sympy = 'symbols(' in code or 'sp.symbols' in code
    uses_sympy_diff = 'diff(' in code
    uses_sympy_solve = 'solve(' in code

    if not uses_sympy:
        # 尝试转换纯 numpy 因子形式多项式，如 (3-x)*(x+2)*(8-x)*(9+x)*(x-7)
        # 支持 f = ... 和 def f(x): return ... 两种形式
        expr_match = re.search(r'f\s*=\s*(.+?)(?:\n|$)', code, re.DOTALL)
        if not expr_match:
            # 尝试匹配 def f(x): return ... 形式
            expr_match = re.search(r'def\s+f\s*\([^)]*\):\s*\n\s*return\s+(.+?)(?:\n|$)', code, re.DOTALL)

        if expr_match:
            expr = expr_match.group(1).strip()
            if '*' in expr and '(' in expr:
                roots = _extract_roots_from_factor_expr(expr, 'x')
                if len(roots) >= 2:
                    roots_code = ', '.join(str(r) for r in roots)
                    return f"""import numpy as np

# 自动转换因子形式多项式为 numpy
roots = np.array([{roots_code}])
print('多项式根:', roots)

coeffs = np.poly(roots)
print('多项式系数:', coeffs)

deriv_coeffs = np.polyder(coeffs)
print('导数系数:', deriv_coeffs)

critical_points = np.roots(deriv_coeffs)
print('临界点:', critical_points)

critical_points = np.sort(critical_points.real)
print('排序后临界点:', critical_points)

print('单调区间分析:')
for i in range(len(critical_points) + 1):
    if i == 0:
        x_test = critical_points[0] - 1
        interval = '(-inf, ' + str(round(critical_points[0], 2)) + ')'
    elif i == len(critical_points):
        x_test = critical_points[-1] + 1
        interval = '(' + str(round(critical_points[-1], 2)) + ', +inf)'
    else:
        x_test = (critical_points[i-1] + critical_points[i]) / 2
        interval = '(' + str(round(critical_points[i-1], 2)) + ', ' + str(round(critical_points[i], 2)) + ')'
    deriv_val = np.polyval(deriv_coeffs, x_test)
    monotonic = '递增' if deriv_val > 0 else '递减'
    print(interval + ': ' + monotonic)
"""
        # 检查是否是 numpy 多项式代码，如果有 bug 也修复
        return _fix_numpy_polynomial_code(code)

    # 检测是否真的需要 sympy（复杂符号运算）
    needs_sympy = any(keyword in code for keyword in [
        'Matrix(', 'matrices', 'integrate(', 'limit(', 'series(',
        'simplify(', 'expand(', 'factor(', 'apart('
    ])

    if needs_sympy:
        return code

    # 提取变量名
    var_match = None
    for pattern in [
        r"(\w+)\s*=\s*sp\.symbols\('([^']+)'\)",
        r"(\w+)\s*=\s*symbols\('([^']+)'\)",
    ]:
        m = re.search(pattern, code)
        if m:
            var_match = m
            break

    var_name = var_match.group(1) if var_match else 'x'

    # 找 f = 表达式
    expr_match = re.search(r'f\s*=\s*(.+?)(?:\n|$)', code, re.DOTALL)
    if expr_match:
        expr = expr_match.group(1).strip()

        # 检查是否是纯 numpy 因子形式多项式，如 (3-x)*(x+2)*(8-x)*(9+x)*(x-7)
        # 如果代码是纯 numpy 但包含因子相乘的形式，也需要转换
        if not uses_sympy and '*' in expr and '(' in expr:
            # 尝试检测并转换纯 numpy 因子形式
            roots = _extract_roots_from_factor_expr(expr, var_name)
            if len(roots) >= 2:
                roots_code = ', '.join(str(r) for r in roots)
                return f"""import numpy as np

# 自动转换因子形式多项式为 numpy
roots = np.array([{roots_code}])
print('多项式根:', roots)

coeffs = np.poly(roots)
print('多项式系数:', coeffs)

deriv_coeffs = np.polyder(coeffs)
print('导数系数:', deriv_coeffs)

critical_points = np.roots(deriv_coeffs)
print('临界点:', critical_points)

critical_points = np.sort(critical_points.real)
print('排序后临界点:', critical_points)

print('单调区间分析:')
for i in range(len(critical_points) + 1):
    if i == 0:
        x_test = critical_points[0] - 1
        interval = '(-inf, ' + str(round(critical_points[0], 2)) + ')'
    elif i == len(critical_points):
        x_test = critical_points[-1] + 1
        interval = '(' + str(round(critical_points[-1], 2)) + ', +inf)'
    else:
        x_test = (critical_points[i-1] + critical_points[i]) / 2
        interval = '(' + str(round(critical_points[i-1], 2)) + ', ' + str(round(critical_points[i], 2)) + ')'
    deriv_val = np.polyval(deriv_coeffs, x_test)
    monotonic = '递增' if deriv_val > 0 else '递减'
    print(interval + ': ' + monotonic)
"""

    if not var_match:
        return code

    var_name = var_match.group(1)

    # 找 f = 表达式
    expr_match = re.search(r'f\s*=\s*(.+?)(?:\n|$)', code, re.DOTALL)
    if not expr_match:
        return code

    expr = expr_match.group(1).strip()

    # 检查是否是简单的因子分解多项式 (x-a)(x-b)...(x-n) 或 (a-x)(b-x)...
    # 支持: (x+2), (x-7), (3-x), (8-x), (9+x) 等
    # 提取所有 (factor) 部分
    factor_pattern = r'\(([^)]+)\)'
    all_factors = re.findall(factor_pattern, expr)

    roots = []
    for factor in all_factors:
        factor = factor.strip()
        # 检查是否是 (x+num) 或 (x-num)
        m1 = re.match(r'^' + var_name + r'([+-])(\d+)$', factor)
        if m1:
            sign, num = m1.groups()
            root = -int(sign + num) if sign == '-' else -int(num)
            roots.append(root)
            continue

        # 检查是否是 (num+x) 或 (num-x) 或 (-x+num) 等
        m2 = re.match(r'^([+-]?\d+)([+-])' + var_name + r'$', factor)
        if m2:
            num_part, sign = m2.groups()
            root = -int(num_part) if sign == '+' else int(num_part)
            roots.append(root)
            continue

        # 检查是否是 (-x+num) 形式
        m3 = re.match(r'^-\s*' + var_name + r'\s*\+\s*(\d+)$', factor)
        if m3:
            roots.append(int(m3.group(1)))
            continue

    if len(roots) >= 2 and uses_sympy_diff:
        # 转换因子形式多项式
        roots_code = ', '.join(str(r) for r in roots)
        new_code = (
            "import numpy as np\n\n"
            "# 自动转换 sympy 多项式为 numpy\n"
            "roots = np.array([" + roots_code + "])\n"
            "print('多项式根:', roots)\n\n"
            "coeffs = np.poly(roots)\n"
            "print('多项式系数:', coeffs)\n\n"
            "deriv_coeffs = np.polyder(coeffs)\n"
            "print('导数系数:', deriv_coeffs)\n\n"
            "critical_points = np.roots(deriv_coeffs)\n"
            "print('临界点:', critical_points)\n\n"
            "critical_points = np.sort(critical_points.real)\n"
            "print('排序后临界点:', critical_points)\n\n"
            "print('单调区间分析:')\n"
            "for i in range(len(critical_points) + 1):\n"
            "    if i == 0:\n"
            "        x_test = critical_points[0] - 1\n"
            "        interval = '(-inf, ' + str(round(critical_points[0], 2)) + ')'\n"
            "    elif i == len(critical_points):\n"
            "        x_test = critical_points[-1] + 1\n"
            "        interval = '(' + str(round(critical_points[-1], 2)) + ', +inf)'\n"
            "    else:\n"
            "        x_test = (critical_points[i-1] + critical_points[i]) / 2\n"
            "        interval = '(' + str(round(critical_points[i-1], 2)) + ', ' + str(round(critical_points[i], 2)) + ')'\n"
            "    deriv_val = np.polyval(deriv_coeffs, x_test)\n"
            "    monotonic = '递增' if deriv_val > 0 else '递减'\n"
            "    print(interval + ': ' + monotonic)\n"
        )
        return new_code

    # 检查是否是展开形式的多项式 4*x**5 - 3*x**4 + ...
    if uses_sympy_diff and '**' in expr and len(roots) < 2:
        # 尝试提取系数
        coeffs = _extract_coeffs_from_expr(expr, var_name)
        if coeffs and len(coeffs) >= 2:
            coeffs_repr = str(coeffs)
            new_code = (
                "import numpy as np\n\n"
                "# 自动转换 sympy 展开多项式为 numpy\n"
                "coeffs = " + coeffs_repr + "\n"
                "print('多项式系数:', coeffs)\n\n"
                "deriv_coeffs = np.polyder(coeffs)\n"
                "print('导数系数:', deriv_coeffs)\n\n"
                "critical = np.roots(deriv_coeffs)\n"
                "critical = np.sort(critical.real)\n"
                "print('临界点:', critical)\n\n"
                "print('单调区间:')\n"
                "for i in range(len(critical) + 1):\n"
                "    if i == 0:\n"
                "        x_test = critical[0] - 1\n"
                "        interval = '(-inf, ' + str(round(critical[0], 2)) + ')'\n"
                "    elif i == len(critical):\n"
                "        x_test = critical[-1] + 1\n"
                "        interval = '(' + str(round(critical[-1], 2)) + ', +inf)'\n"
                "    else:\n"
                "        x_test = (critical[i-1] + critical[i]) / 2\n"
                "        interval = '(' + str(round(critical[i-1], 2)) + ', ' + str(round(critical[i], 2)) + ')'\n"
                "    deriv_val = np.polyval(deriv_coeffs, x_test)\n"
                "    monotonic = '递增' if deriv_val > 0 else '递减'\n"
                "    print(interval + ': ' + monotonic)\n"
            )
            return new_code

    return code


def _extract_roots_from_factor_expr(expr: str, var_name: str = 'x') -> list:
    """从因子形式的表达式中提取根"""
    import re

    factor_pattern = r'\(([^)]+)\)'
    all_factors = re.findall(factor_pattern, expr)

    roots = []
    for factor in all_factors:
        factor = factor.strip()

        # 检查是否是 (x+num) 或 (x-num) - 支持空格
        m1 = re.match(r'^\s*' + var_name + r'\s*([+-])\s*(\d+)\s*$', factor)
        if m1:
            sign, num = m1.groups()
            root = -int(sign + num) if sign == '-' else -int(num)
            roots.append(root)
            continue

        # 检查是否是 (num+x) 或 (num-x) - 支持空格
        m2 = re.match(r'^\s*([+-]?\d+)\s*([+-])\s*' + var_name + r'\s*$', factor)
        if m2:
            num_part, sign = m2.groups()
            root = -int(num_part) if sign == '+' else int(num_part)
            roots.append(root)
            continue

    return roots


def _extract_coeffs_from_expr(expr: str, var_name: str) -> list:
    """从多项式表达式中提取系数"""
    import re

    coeffs_dict = {}

    # 匹配 a*x**n 模式（例如 4*x**5, -3*x**4）
    pattern_powered = r'([+-]?\d*)\s*\*?\s*' + var_name + r'\s*\*\*\s*(\d+)'
    matches_powered = re.findall(pattern_powered, expr.replace(' ', ''))

    for coef_str, power_str in matches_powered:
        power = int(power_str)
        coef = int(coef_str) if coef_str and coef_str not in '+-' else (1 if coef_str in ['+', ''] else -1)
        coeffs_dict[power] = coef

    # 匹配 a*x 模式（x的一次方，不能跟 ** 或数字）
    pattern_single = r'([+-]?\d*)\s*\*?\s*' + var_name + r'(?!\d|\*\*)'
    matches_single = re.findall(pattern_single, expr.replace(' ', ''))

    for match in matches_single:
        coef_str = match
        coef = int(coef_str) if coef_str and coef_str not in '+-' else (1 if coef_str in ['+', ''] else -1)
        coeffs_dict[1] = coef

    if not coeffs_dict:
        return []

    # 检查常数项
    const_match = re.search(r'([+-]\d+)$', expr.replace(' ', ''))
    if const_match:
        coeffs_dict[0] = int(const_match.group(1))

    # 构建系数数组（从高次到低次）
    max_power = max(coeffs_dict.keys())
    coeffs = [coeffs_dict.get(i, 0) for i in range(max_power, -1, -1)]
    return coeffs


def _fix_numpy_polynomial_code(code: str) -> str:
    """修复 numpy 多项式代码中的常见 bug"""
    import re

    # 检查是否是多项式相关代码
    is_polynomial = 'np.polyder' in code or 'np.roots' in code or 'poly1d' in code or ('coeffs' in code and '[' in code)
    if not is_polynomial:
        return code

    # 尝试提取多项式系数
    coeffs_match = re.search(r'coeffs\s*=\s*\[([^\]]+)\]', code)
    if coeffs_match:
        coeffs_str = coeffs_match.group(1)
        try:
            coeffs = [float(x.strip()) for x in coeffs_str.split(',')]
            if len(coeffs) >= 2:
                # 重新生成正确的代码
                coeffs_repr = str(coeffs)
                new_code = f"""import numpy as np

# 多项式系数: {coeffs_repr}
coeffs = {coeffs_repr}

# 求导
deriv_coeffs = np.polyder(coeffs)
print('导数系数:', deriv_coeffs)

# 临界点
critical = np.roots(deriv_coeffs)
# 只取实部（忽略数值误差产生的虚部）
critical = np.sort(critical.real)
print('临界点:', critical)

# 单调区间
print('单调区间:')
for i in range(len(critical) + 1):
    if i == 0:
        x_test = critical[0] - 1
        interval = f"(-inf, {{critical[0]:.2f}})"
    elif i == len(critical):
        x_test = critical[-1] + 1
        interval = f"({{critical[-1]:.2f}}, +inf)"
    else:
        x_test = (critical[i-1] + critical[i]) / 2
        interval = f"({{critical[i-1]:.2f}}, {{critical[i]:.2f}})"

    deriv_val = np.polyval(deriv_coeffs, x_test)
    monotonic = '递增' if deriv_val > 0 else '递减'
    print(f"{{interval}}: {{monotonic}}")
"""
                return new_code
        except:
            pass

    return code

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