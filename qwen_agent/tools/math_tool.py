"""数学计算工具 - 通用数学函数和问题求解"""
from typing import Dict, Any, List, Optional, Tuple, Union
import sympy
from sympy import (symbols, sympify, diff, integrate, limit, solve,
                   solve_univariate_inequality, is_increasing, is_decreasing,
                   Matrix, sqrt, Abs, sin, cos, tan, exp, ln, log,
                   asin, acos, atan, oo, N, refine, Q, ask, pi, E)
from sympy.core.relational import Relational
from sympy.plotting import plot
import io
import numpy as np


# ============== 基础数学函数 ==============

def calculate(expression: str) -> Dict[str, Any]:
    """计算数学表达式的值"""
    try:
        result = sympify(expression)
        return {"success": True, "result": str(result), "error": ""}
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def derivative(expression: str, variable: str = 'x', n: int = 1) -> Dict[str, Any]:
    """求导数

    Args:
        expression: 函数表达式
        variable: 自变量
        n: 导数阶数，默认1阶
    """
    try:
        x = symbols(variable)
        f = sympify(expression)
        f_n = diff(f, x, n)
        return {
            "success": True,
            "result": str(f_n),
            "expression": expression,
            "variable": variable,
            "order": n,
            "error": ""
        }
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def indefinite_integral(expression: str, variable: str = 'x') -> Dict[str, Any]:
    """求不定积分

    Args:
        expression: 函数表达式
        variable: 自变量
    """
    try:
        x = symbols(variable)
        f = sympify(expression)
        F = integrate(f, x)
        return {
            "success": True,
            "result": str(F) + " + C",
            "expression": expression,
            "variable": variable,
            "error": ""
        }
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def definite_integral(expression: str, variable: str = 'x',
                      lower: float = None, upper: float = None) -> Dict[str, Any]:
    """求定积分

    Args:
        expression: 函数表达式
        variable: 自变量
        lower: 下限
        upper: 上限
    """
    try:
        x = symbols(variable)
        f = sympify(expression)

        if lower is not None and upper is not None:
            result = integrate(f, (x, lower, upper))
            # 尝试获取数值
            try:
                numeric_result = float(N(result))
                result_str = f"{numeric_result:.6f}"
            except:
                result_str = str(result)

            return {
                "success": True,
                "result": result_str,
                "expression": expression,
                "variable": variable,
                "lower": lower,
                "upper": upper,
                "antiderivative": str(integrate(f, x)),
                "error": ""
            }
        else:
            return {"success": False, "result": "", "error": "必须提供 lower 和 upper 参数"}
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def limit_calc(expression: str, variable: str = 'x',
               point: Union[str, float] = 0) -> Dict[str, Any]:
    """求极限

    Args:
        expression: 函数表达式
        variable: 自变量
        point: 趋近的点（如 0, 'oo', '+oo', '-oo'）
    """
    try:
        x = symbols(variable)
        f = sympify(expression)

        if isinstance(point, str):
            if point == 'oo' or point == '+oo':
                pt = oo
            elif point == '-oo':
                pt = -oo
            else:
                pt = sympify(point)
        else:
            pt = point

        result = limit(f, x, pt)
        return {
            "success": True,
            "result": str(result),
            "expression": expression,
            "variable": variable,
            "point": str(pt),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def solve_equation(equation: str, variable: str = 'x') -> Dict[str, Any]:
    """解方程

    Args:
        equation: 方程表达式，如 "x**2 - 4 = 0"
        variable: 自变量
    """
    try:
        x = symbols(variable)
        # 解析方程
        if '=' in equation:
            left, right = equation.split('=')
            eq = sympify(left) - sympify(right)
        else:
            eq = sympify(equation)

        solutions = solve(eq, x)
        # 过滤实数解
        real_solutions = []
        for sol in solutions:
            try:
                val = complex(sol)
                if abs(val.imag) < 1e-10:
                    real_solutions.append(float(val.real))
            except:
                real_solutions.append(str(sol))

        return {
            "success": True,
            "solutions": real_solutions,
            "count": len(real_solutions),
            "equation": equation,
            "variable": variable,
            "error": ""
        }
    except Exception as e:
        return {"success": False, "solutions": [], "error": str(e)}


def solve_inequality(inequality: str, variable: str = 'x') -> Dict[str, Any]:
    """解不等式

    Args:
        inequality: 不等式，如 "x**2 - 4 < 0"
        variable: 自变量
    """
    try:
        x = symbols(variable)
        # 解析不等式
        if '<' in inequality:
            left, right = inequality.split('<')
            if inequality.count('<') > 1:
                # 可能是 <=
                left, right = inequality.split('<=')
                expr = sympify(left) - sympify(right)
                relation = sympy.Le
            else:
                expr = sympify(left) - sympify(right)
                relation = sympy.Lt
        elif '>' in inequality:
            left, right = inequality.split('>')
            if inequality.count('>') > 1:
                left, right = inequality.split('>=')
                expr = sympify(left) - sympify(right)
                relation = sympy.Ge
            else:
                expr = sympify(left) - sympify(right)
                relation = sympy.Gt
        elif '>=' in inequality:
            left, right = inequality.split('>=')
            expr = sympify(left) - sympify(right)
            relation = sympy.Ge
        elif '<=' in inequality:
            left, right = inequality.split('<=')
            expr = sympify(left) - sympify(right)
            relation = sympy.Le
        else:
            return {"success": False, "result": "", "error": "无法识别不等式符号"}

        result = solve_univariate_inequality(relation(expr, 0), x)
        return {
            "success": True,
            "result": str(result),
            "inequality": inequality,
            "variable": variable,
            "error": ""
        }
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def find_extrema(expression: str, variable: str = 'x') -> Dict[str, Any]:
    """求极值点

    Args:
        expression: 函数表达式
        variable: 自变量
    """
    try:
        x = symbols(variable)
        f = sympify(expression)
        f_prime = diff(f, x)

        # 求临界点
        critical_points = solve(f_prime, x)
        f_double = diff(f, x, 2)

        extrema = []
        for pt in critical_points:
            try:
                val = float(N(pt))
                second_deriv = float(N(f_double.subs(x, pt)))
                if second_deriv > 0:
                    extrema_type = "极小值"
                elif second_deriv < 0:
                    extrema_type = "极大值"
                else:
                    extrema_type = "可能是拐点"

                f_val = float(N(f.subs(x, pt)))
                extrema.append({
                    "point": round(val, 6),
                    "value": round(f_val, 6),
                    "type": extrema_type
                })
            except:
                pass

        return {
            "success": True,
            "extrema": extrema,
            "critical_points": [str(pt) for pt in critical_points],
            "expression": expression,
            "variable": variable,
            "error": ""
        }
    except Exception as e:
        return {"success": False, "extrema": [], "error": str(e)}


def find_inflection_points(expression: str, variable: str = 'x') -> Dict[str, Any]:
    """求拐点

    Args:
        expression: 函数表达式
        variable: 自变量
    """
    try:
        x = symbols(variable)
        f = sympify(expression)
        f_double = diff(f, x, 2)
        f_triple = diff(f, x, 3)

        # 求二阶导数为零的点
        inflection_points = solve(f_double, x)

        result = []
        for pt in inflection_points:
            try:
                # 验证三阶导数不为零
                if f_triple.subs(x, pt) != 0:
                    val = float(N(pt))
                    result.append(round(val, 6))
            except:
                pass

        return {
            "success": True,
            "inflection_points": result,
            "expression": expression,
            "variable": variable,
            "error": ""
        }
    except Exception as e:
        return {"success": False, "inflection_points": [], "error": str(e)}


def analyze_monotonic(expression: str, variable: str = 'x') -> Dict[str, Any]:
    """分析函数单调性

    Args:
        expression: 函数表达式
        variable: 自变量
    """
    try:
        x = symbols(variable)
        f = sympify(expression)
        f_prime = diff(f, x)

        # 求临界点（保留精确符号形式）
        critical_points_raw = solve(f_prime, x)

        # 分离精确符号形式和数值近似
        critical_points_exact = []  # 精确符号形式
        critical_points = []        # 数值近似

        for pt in critical_points_raw:
            # 检查是否是实数
            if pt.is_real or (isinstance(pt, float) and not math.isnan(pt)):
                try:
                    val = complex(pt)
                    if val.imag == 0:
                        real_val = val.real
                        if real_val != float('inf') and real_val != float('-inf'):
                            critical_points_exact.append(str(pt))
                            critical_points.append(float(real_val))
                except:
                    pass

        critical_points = sorted(critical_points)

        # 构建友好的格式（如 -4/3 ≈ -1.33）
        critical_points_friendly = []
        for i, cp in enumerate(critical_points):
            if i < len(critical_points_exact):
                exact = critical_points_exact[i]
                # 如果是分数形式，添加小数近似
                if '/' in exact:
                    critical_points_friendly.append(f"{exact} ≈ {round(cp, 2)}")
                else:
                    critical_points_friendly.append(f"{round(cp, 2)}")
            else:
                critical_points_friendly.append(f"{round(cp, 2)}")

        # 分析单调区间（使用精确符号形式）
        intervals = []
        test_points = critical_points if critical_points else [0]

        # 尝试构建精确的区间端点
        exact_points = critical_points_exact if len(critical_points_exact) == len(critical_points) else None
        friendly_points = critical_points_friendly if len(critical_points_friendly) == len(critical_points) else None

        for i in range(len(test_points) + 1):
            if i == 0:
                test_x = test_points[0] - 1 if test_points else -1
                if exact_points and friendly_points:
                    interval_str = f"(-∞, {friendly_points[0]})"
                elif exact_points:
                    interval_str = f"(-∞, {exact_points[0]})"
                else:
                    interval_str = f"(-∞, {round(test_points[0], 2)})" if test_points else "(-∞, +∞)"
            elif i == len(test_points):
                test_x = test_points[-1] + 1
                if exact_points and friendly_points:
                    interval_str = f"({friendly_points[-1]}, +∞)"
                elif exact_points:
                    interval_str = f"({exact_points[-1]}, +∞)"
                else:
                    interval_str = f"({round(test_points[-1], 2)}, +∞)"
            else:
                test_x = (test_points[i-1] + test_points[i]) / 2
                if exact_points and friendly_points:
                    interval_str = f"({friendly_points[i-1]}, {friendly_points[i]})"
                elif exact_points:
                    interval_str = f"({exact_points[i-1]}, {exact_points[i]})"
                else:
                    interval_str = f"({round(test_points[i-1], 2)}, {round(test_points[i], 2)})"

            try:
                deriv_val = float(N(f_prime.subs(x, test_x)))
                monotonic = "递增" if deriv_val > 0 else "递减"
            except:
                monotonic = "未知"

            intervals.append({"interval": interval_str, "monotonic": monotonic})

        return {
            "success": True,
            "derivative": str(f_prime),
            "critical_points": critical_points,
            "critical_points_friendly": critical_points_friendly,
            "monotonic_intervals": intervals,
            "expression": expression,
            "variable": variable,
            "error": ""
        }
    except Exception as e:
        return {"success": False, "derivative": "", "critical_points": [],
                "monotonic_intervals": [], "error": str(e)}


# ============== 应用题计算 ==============

def compound_interest(principal: float, rate: float, years: float,
                      frequency: int = 12, monthly: bool = False) -> Dict[str, Any]:
    """复利计算

    Args:
        principal: 本金
        rate: 年利率（如 0.03 表示 3%）
        years: 年数
        frequency: 每年复利次数，默认12（按月）
        monthly: 是否按月还款/计息
    """
    try:
        principal = float(principal)
        rate = float(rate)
        years = float(years)
        frequency = int(frequency) if frequency else 12
        if monthly:
            # 按月复利
            r_monthly = rate / 12
            n_months = years * 12
            amount = principal * (1 + r_monthly) ** n_months
            interest = amount - principal
        else:
            # 按年复利
            amount = principal * (1 + rate / frequency) ** (frequency * years)
            interest = amount - principal

        return {
            "success": True,
            "principal": float(principal),
            "rate": float(rate),
            "years": float(years),
            "frequency": int(frequency),
            "monthly": bool(monthly),
            "total_amount": round(float(amount), 2),
            "interest": round(float(interest), 2),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "total_amount": 0, "interest": 0, "error": str(e)}


def simple_interest(principal: float, rate: float, years: float) -> Dict[str, Any]:
    """单利计算

    Args:
        principal: 本金
        rate: 年利率
        years: 年数
    """
    try:
        principal = float(principal)
        rate = float(rate)
        years = float(years)
        interest = principal * rate * years
        total = principal + interest
        return {
            "success": True,
            "principal": float(principal),
            "rate": float(rate),
            "years": float(years),
            "interest": round(float(interest), 2),
            "total_amount": round(float(total), 2),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "interest": 0, "total_amount": 0, "error": str(e)}


def loan_repayment(principal: float, annual_rate: float, years: int,
                   payment_type: str = "equal_principal") -> Dict[str, Any]:
    """贷款还款计算

    Args:
        principal: 贷款本金
        annual_rate: 年利率
        years: 还款年数
        payment_type: "equal_principal" 等额本金, "equal_payment" 等额本息
    """
    try:
        principal = float(principal)
        annual_rate = float(annual_rate)
        years = int(years)
        monthly_rate = annual_rate / 12
        months = years * 12

        if payment_type == "equal_principal":
            # 等额本金
            monthly_principal = principal / months
            total_interest = 0
            monthly_payments = []

            remaining = principal
            for month in range(1, months + 1):
                interest = remaining * monthly_rate
                payment = monthly_principal + interest
                total_interest += interest
                remaining -= monthly_principal
                monthly_payments.append({
                    "month": month,
                    "principal": round(monthly_principal, 2),
                    "interest": round(interest, 2),
                    "payment": round(payment, 2),
                    "remaining": round(remaining, 2)
                })

            return {
                "success": True,
                "type": "等额本金",
                "principal": float(principal),
                "annual_rate": float(annual_rate),
                "years": int(years),
                "monthly_payment_first": round(float(monthly_payments[0]['payment']), 2),
                "monthly_payment_last": round(float(monthly_payments[-1]['payment']), 2),
                "total_interest": round(float(total_interest), 2),
                "total_amount": round(float(principal) + float(total_interest), 2),
                "monthly_payments_sample": monthly_payments[:3] + monthly_payments[-1:],
                "error": ""
            }

        elif payment_type == "equal_payment":
            # 等额本息
            if monthly_rate == 0:
                monthly_payment = principal / months
            else:
                monthly_payment = (principal * monthly_rate * (1 + monthly_rate)**months) / ((1 + monthly_rate)**months - 1)

            total_amount = monthly_payment * months
            total_interest = total_amount - principal

            return {
                "success": True,
                "type": "等额本息",
                "principal": float(principal),
                "annual_rate": float(annual_rate),
                "years": int(years),
                "monthly_payment": round(float(monthly_payment), 2),
                "total_interest": round(float(total_interest), 2),
                "total_amount": round(float(total_amount), 2),
                "error": ""
            }
        else:
            return {"success": False, "result": "", "error": "不支持的还款类型"}
    except Exception as e:
        return {"success": False, "result": "", "error": str(e)}


def sequence_sum(sequence_type: str, n: int, **params) -> Dict[str, Any]:
    """数列求和

    Args:
        sequence_type: "arithmetic" 等差, "geometric" 等比, "general" 一般数列
        n: 求和项数
        params: 其他参数
            arithmetic: a1(首项), d(公差)
            geometric: a1(首项), r(公比)
            general: formula(通项公式)
    """
    try:
        n = int(n)
        if sequence_type == "arithmetic":
            a1 = float(params.get('a1', 0))
            d = float(params.get('d', 0))
            Sn = n * a1 + d * n * (n - 1) / 2
            an = a1 + (n - 1) * d
            return {
                "success": True,
                "type": "等差数列",
                "n": int(n),
                "first_term": float(a1),
                "common_difference": float(d),
                "general_term": f"{a1} + (n-1)*{d}",
                "nth_term": float(an),
                "sum": round(float(Sn), 6),
                "error": ""
            }

        elif sequence_type == "geometric":
            a1 = float(params.get('a1', 0))
            r = float(params.get('r', 0))
            if r == 1:
                Sn = n * a1
            else:
                Sn = a1 * (1 - r**n) / (1 - r)
            an = a1 * r**(n - 1)
            return {
                "success": True,
                "type": "等比数列",
                "n": int(n),
                "first_term": float(a1),
                "common_ratio": float(r),
                "general_term": f"{a1} * {r}^(n-1)",
                "nth_term": float(an),
                "sum": round(float(Sn), 6),
                "error": ""
            }

        elif sequence_type == "general":
            formula = params.get('formula', '')
            x = symbols('n')
            f = sympify(formula)
            total = sum([float(N(f.subs(x, i))) for i in range(1, n + 1)])
            return {
                "success": True,
                "type": "一般数列",
                "n": n,
                "general_term": formula,
                "sum": round(total, 6),
                "error": ""
            }
        else:
            return {"success": False, "sum": 0, "error": "不支持的数列类型"}
    except Exception as e:
        return {"success": False, "sum": 0, "error": str(e)}


def probability(n: int, k: int, p: float = 0.5,
                calculation: str = "binomial") -> Dict[str, Any]:
    """概率计算

    Args:
        n: 试验次数
        k: 成功次数
        p: 成功概率
        calculation: "binomial" 二项分布, "normal" 正态分布近似
    """
    try:
        from math import comb, sqrt, exp, pi, erf

        if calculation == "binomial":
            # 二项分布 P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
            prob = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

            # 累积概率 P(X <= k)
            cum_prob = sum([comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1)])

            return {
                "success": True,
                "type": "二项分布",
                "n": n,
                "k": k,
                "p": p,
                "P(X=k)": round(prob, 8),
                "P(X<=k)": round(cum_prob, 8),
                "E(X)": round(n * p, 4),
                "Var(X)": round(n * p * (1 - p), 4),
                "error": ""
            }

        elif calculation == "normal":
            # 正态分布近似（当 n 较大时）
            mean = n * p
            std = sqrt(n * p * (1 - p))
            z = (k - mean) / std

            # 标准正态分布累积概率
            from scipy.special import erf
            cum_prob = 0.5 * (1 + erf(z / sqrt(2)))

            return {
                "success": True,
                "type": "正态分布近似",
                "n": n,
                "k": k,
                "p": p,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "z_score": round(z, 4),
                "P(X<=k)": round(cum_prob, 8),
                "error": ""
            }
        else:
            return {"success": False, "result": 0, "error": "不支持的计算类型"}
    except Exception as e:
        return {"success": False, "result": 0, "error": str(e)}


def savings_find_year(annual_deposit: float, annual_rate: float,
                      target: float, initial_principal: float = 0,
                      max_years: int = 100) -> Dict[str, Any]:
    """计算存款多少年后超过目标金额（逐年存款+复利）

    Args:
        annual_deposit: 每年存款金额
        annual_rate: 年利率（如 0.03 表示 3%）
        target: 目标金额
        initial_principal: 初始本金（默认0）
        max_years: 最大计算年数（默认100年）

    Returns:
        包含第几年超过目标、当时存款总额等信息
    """
    try:
        annual_deposit = float(annual_deposit)
        annual_rate = float(annual_rate)
        target = float(target)
        initial_principal = float(initial_principal)
        max_years = int(max_years)

        if annual_rate < 0:
            return {"success": False, "error": "利率不能为负"}

        balance = initial_principal
        year = 0
        yearly_data = []

        while year < max_years:
            year += 1
            # 复利：本金 + 利息 + 新存款
            balance = balance * (1 + annual_rate) + annual_deposit
            yearly_data.append({
                "year": year,
                "deposit": annual_deposit,
                "interest": round(balance * annual_rate / (1 + annual_rate) if balance > 0 else 0, 2),
                "balance": round(balance, 2)
            })

            if balance >= target:
                break

        if balance >= target:
            total_deposited = initial_principal + year * annual_deposit
            return {
                "success": True,
                "type": "逐年存款复利计算",
                "annual_deposit": float(annual_deposit),
                "annual_rate": float(annual_rate),
                "initial_principal": float(initial_principal),
                "target": float(target),
                "target_year": year,
                "target_balance": round(float(balance), 2),
                "total_deposited": round(float(total_deposited), 2),
                "total_interest": round(float(balance - total_deposited), 2),
                "yearly_sample": yearly_data[:3] + [yearly_data[-1]] if len(yearly_data) > 4 else yearly_data,
                "error": ""
            }
        else:
            return {
                "success": False,
                "error": f"经过 {max_years} 年仍未达到目标金额，当前存款 {balance:.2f}"
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def savings_accumulation(annual_deposit: float, annual_rate: float,
                         years: int) -> Dict[str, Any]:
    """计算n年后的存款总额（逐年存款+复利）

    Args:
        annual_deposit: 每年存款金额
        annual_rate: 年利率（如 0.03 表示 3%）
        years: 存款年数

    Returns:
        包含n年后存款总额、总存款、总利息等信息
    """
    try:
        annual_deposit = float(annual_deposit)
        annual_rate = float(annual_rate)
        years = int(years)

        balance = 0
        yearly_data = []

        for year in range(1, years + 1):
            balance = balance * (1 + annual_rate) + annual_deposit
            yearly_data.append({
                "year": year,
                "balance": round(float(balance), 2)
            })

        total_deposited = years * annual_deposit
        total_interest = balance - total_deposited

        return {
            "success": True,
            "type": "逐年存款复利",
            "annual_deposit": float(annual_deposit),
            "annual_rate": float(annual_rate),
            "years": int(years),
            "final_balance": round(float(balance), 2),
            "total_deposited": round(float(total_deposited), 2),
            "total_interest": round(float(total_interest), 2),
            "yearly_sample": yearly_data[:3] + [yearly_data[-1]] if len(yearly_data) > 4 else yearly_data,
            "error": ""
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============== 几何计算 ==============

def triangle_area(base: float = None, height: float = None,
                  a: float = None, b: float = None, c: float = None,
                  area_type: str = "base_height") -> Dict[str, Any]:
    """三角形面积

    Args:
        base, height: 底和高
        a, b, c: 三边长（海伦公式）
        area_type: "base_height" 底高, "heron" 海伦公式
    """
    try:
        if area_type == "base_height" and base is not None and height is not None:
            area = 0.5 * float(base) * float(height)
            return {"success": True, "area": round(float(area), 4), "type": "底×高/2", "error": ""}

        elif area_type == "heron" and a and b and c:
            a = float(a); b = float(b); c = float(c)
            s = (a + b + c) / 2
            area = sqrt(s * (s - a) * (s - b) * (s - c))
            return {"success": True, "area": round(float(area), 4), "sides": [float(a), float(b), float(c)], "type": "海伦公式", "error": ""}

        else:
            return {"success": False, "area": 0, "error": "参数不足"}
    except Exception as e:
        return {"success": False, "area": 0, "error": str(e)}


def circle_area(radius: float = None, diameter: float = None) -> Dict[str, Any]:
    """圆面积和周长"""
    try:
        if radius is not None:
            radius = float(radius)
        elif diameter is not None:
            radius = float(diameter) / 2
        else:
            return {"success": False, "area": 0, "error": "需要提供 radius 或 diameter"}

        area = float(pi) * radius ** 2
        circumference = float(pi) * 2 * radius

        return {
            "success": True,
            "radius": float(radius),
            "area": round(float(area), 4),
            "circumference": round(float(circumference), 4),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "area": 0, "error": str(e)}


def rectangle_area(length: float, width: float) -> Dict[str, Any]:
    """矩形面积和周长"""
    try:
        length = float(length)
        width = float(width)
        area = length * width
        perimeter = 2 * (length + width)
        diagonal = sqrt(length ** 2 + width ** 2)

        return {
            "success": True,
            "length": float(length),
            "width": float(width),
            "area": round(float(area), 4),
            "perimeter": round(float(perimeter), 4),
            "diagonal": round(float(diagonal), 4),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "area": 0, "error": str(e)}


def sphere_volume(radius: float) -> Dict[str, Any]:
    """球体积和表面积"""
    try:
        radius = float(radius)
        volume = (4 / 3) * pi * radius ** 3
        surface_area = 4 * pi * radius ** 2

        return {
            "success": True,
            "radius": float(radius),
            "volume": round(float(volume), 4),
            "surface_area": round(float(surface_area), 4),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "volume": 0, "error": str(e)}


def cylinder_volume(radius: float, height: float) -> Dict[str, Any]:
    """圆柱体积和表面积"""
    try:
        radius = float(radius)
        height = float(height)
        volume = pi * radius ** 2 * height
        surface_area = 2 * pi * radius * (radius + height)

        return {
            "success": True,
            "radius": float(radius),
            "height": float(height),
            "volume": round(float(volume), 4),
            "surface_area": round(float(surface_area), 4),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "volume": 0, "error": str(e)}


def cone_volume(radius: float, height: float) -> Dict[str, Any]:
    """圆锥体积和表面积"""
    try:
        radius = float(radius)
        height = float(height)
        volume = (1 / 3) * pi * radius ** 2 * height
        slant_height = sqrt(radius ** 2 + height ** 2)
        surface_area = pi * radius * (radius + slant_height)

        return {
            "success": True,
            "radius": float(radius),
            "height": float(height),
            "volume": round(float(volume), 4),
            "surface_area": round(float(surface_area), 4),
            "slant_height": round(float(slant_height), 4),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "volume": 0, "error": str(e)}


def cuboid_volume(length: float, width: float, height: float) -> Dict[str, Any]:
    """长方体体积和表面积"""
    try:
        length = float(length)
        width = float(width)
        height = float(height)
        volume = length * width * height
        surface_area = 2 * (length * width + width * height + height * length)
        diagonal = sqrt(length ** 2 + width ** 2 + height ** 2)

        return {
            "success": True,
            "length": float(length),
            "width": float(width),
            "height": float(height),
            "volume": round(float(volume), 4),
            "surface_area": round(float(surface_area), 4),
            "diagonal": round(float(diagonal), 4),
            "error": ""
        }
    except Exception as e:
        return {"success": False, "volume": 0, "error": str(e)}


# ============== 矩阵和向量 ==============

def matrix_operation(matrix1: List[List[float]],
                    matrix2: List[List[float]] = None,
                    operation: str = "determinant") -> Dict[str, Any]:
    """矩阵运算

    Args:
        matrix1: 矩阵1
        matrix2: 矩阵2（用于乘法、加法）
        operation: "determinant", "inverse", "transpose", "add", "multiply"
    """
    try:
        M1 = Matrix(matrix1)

        if operation == "determinant":
            det = M1.det()
            return {"success": True, "result": float(det), "operation": "行列式", "error": ""}

        elif operation == "inverse":
            inv = M1.inv()
            return {"success": True, "result": [[float(x) for x in row] for row in inv.tolist()],
                    "operation": "逆矩阵", "error": ""}

        elif operation == "transpose":
            trans = M1.T
            return {"success": True, "result": [[float(x) for x in row] for row in trans.tolist()],
                    "operation": "转置矩阵", "error": ""}

        elif operation == "multiply" and matrix2 is not None:
            M2 = Matrix(matrix2)
            result = M1 * M2
            return {"success": True, "result": [[float(x) for x in row] for row in result.tolist()],
                    "operation": "矩阵乘法", "error": ""}

        elif operation == "add" and matrix2 is not None:
            M2 = Matrix(matrix2)
            result = M1 + M2
            return {"success": True, "result": [[float(x) for x in row] for row in result.tolist()],
                    "operation": "矩阵加法", "error": ""}

        else:
            return {"success": False, "result": None, "error": "不支持的操作或参数不足"}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


def vector_operation(vector1: List[float],
                     vector2: List[float] = None,
                     operation: str = "magnitude") -> Dict[str, Any]:
    """向量运算

    Args:
        vector1: 向量1
        vector2: 向量2（用于点积、叉积）
        operation: "magnitude", "dot", "cross", "add", "subtract"
    """
    try:
        v1 = [float(x) for x in vector1]
        v1_matrix = Matrix(v1)

        if operation == "magnitude":
            mag = sqrt(sum(x**2 for x in v1))
            return {"success": True, "result": float(mag), "operation": "向量模", "error": ""}

        elif operation == "normalize":
            mag = sqrt(sum(x**2 for x in v1))
            norm = [x/mag for x in v1]
            return {"success": True, "result": [float(x) for x in norm], "operation": "单位向量", "error": ""}

        elif operation == "dot" and vector2 is not None:
            v2 = [float(x) for x in vector2]
            v2_matrix = Matrix(v2)
            dot_product = v1_matrix.dot(v2_matrix)
            return {"success": True, "result": float(dot_product), "operation": "点积", "error": ""}

        elif operation == "cross" and vector2 is not None:
            if len(v1) != 3 or len(v2) != 3:
                return {"success": False, "result": None, "error": "叉积仅支持3维向量"}
            v2 = [float(x) for x in vector2]
            v2_matrix = Matrix(v2)
            cross_product = v1_matrix.cross(v2_matrix)
            return {"success": True, "result": [float(x) for x in cross_product.tolist()],
                    "operation": "叉积", "error": ""}

        elif operation == "add" and vector2 is not None:
            v2 = [float(x) for x in vector2]
            v2_matrix = Matrix(v2)
            result = v1_matrix + v2_matrix
            return {"success": True, "result": [float(x) for x in result.tolist()],
                    "operation": "向量加法", "error": ""}

        else:
            return {"success": False, "result": None, "error": "不支持的操作或参数不足"}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}


# ============== 主入口函数 ==============

def math_tool(func_name: str, **kwargs) -> Dict[str, Any]:
    """数学工具统一入口

    Args:
        func_name: 函数名
        **kwargs: 函数参数
    """
    dispatch = {
        # 基础函数
        "calculate": lambda: calculate(kwargs.get("expression", "")),
        "derivative": lambda: derivative(kwargs.get("expression", ""),
                                        kwargs.get("variable", "x"),
                                        kwargs.get("n", 1)),
        "indefinite_integral": lambda: indefinite_integral(kwargs.get("expression", ""),
                                                          kwargs.get("variable", "x")),
        "definite_integral": lambda: definite_integral(kwargs.get("expression", ""),
                                                        kwargs.get("variable", "x"),
                                                        kwargs.get("lower"),
                                                        kwargs.get("upper")),
        "limit_calc": lambda: limit_calc(kwargs.get("expression", ""),
                                         kwargs.get("variable", "x"),
                                         kwargs.get("point", 0)),
        "solve_equation": lambda: solve_equation(kwargs.get("equation", ""),
                                                 kwargs.get("variable", "x")),
        "solve_inequality": lambda: solve_inequality(kwargs.get("inequality", ""),
                                                   kwargs.get("variable", "x")),
        "find_extrema": lambda: find_extrema(kwargs.get("expression", ""),
                                            kwargs.get("variable", "x")),
        "find_inflection_points": lambda: find_inflection_points(kwargs.get("expression", ""),
                                                               kwargs.get("variable", "x")),
        "analyze_monotonic": lambda: analyze_monotonic(kwargs.get("expression", ""),
                                                       kwargs.get("variable", "x")),

        # 应用题
        "compound_interest": lambda: compound_interest(kwargs.get("principal", 0),
                                                       kwargs.get("rate", 0),
                                                       kwargs.get("years", 0),
                                                       kwargs.get("frequency", 12),
                                                       kwargs.get("monthly", False)),
        "simple_interest": lambda: simple_interest(kwargs.get("principal", 0),
                                                   kwargs.get("rate", 0),
                                                   kwargs.get("years", 0)),
        "loan_repayment": lambda: loan_repayment(kwargs.get("principal", 0),
                                                 kwargs.get("annual_rate", 0),
                                                 kwargs.get("years", 0),
                                                 kwargs.get("payment_type", "equal_principal")),

        # 存款计算
        "savings_find_year": lambda: savings_find_year(kwargs.get("annual_deposit", 0),
                                                       kwargs.get("annual_rate", 0),
                                                       kwargs.get("target", 0),
                                                       kwargs.get("initial_principal", 0),
                                                       kwargs.get("max_years", 100)),
        "savings_accumulation": lambda: savings_accumulation(kwargs.get("annual_deposit", 0),
                                                             kwargs.get("annual_rate", 0),
                                                             kwargs.get("years", 0)),

        "sequence_sum": lambda: sequence_sum(kwargs.get("sequence_type", "arithmetic"),
                                            kwargs.get("n", 10),
                                            a1=kwargs.get("a1", 1),
                                            d=kwargs.get("d", 1),
                                            r=kwargs.get("r", 2),
                                            formula=kwargs.get("formula", "")),
        "probability_calc": lambda: probability(kwargs.get("n", 10),
                                               kwargs.get("k", 5),
                                               kwargs.get("p", 0.5),
                                               kwargs.get("calculation", "binomial")),

        # 几何
        "triangle_area": lambda: triangle_area(kwargs.get("base"), kwargs.get("height"),
                                            kwargs.get("a"), kwargs.get("b"), kwargs.get("c"),
                                            kwargs.get("area_type", "heron") if (kwargs.get("a") and kwargs.get("b") and kwargs.get("c")) else "base_height"),
        "circle_area": lambda: circle_area(kwargs.get("radius"), kwargs.get("diameter")),
        "rectangle_area": lambda: rectangle_area(kwargs.get("length", 0),
                                                kwargs.get("width", 0)),
        "sphere_volume": lambda: sphere_volume(kwargs.get("radius", 0)),
        "cylinder_volume": lambda: cylinder_volume(kwargs.get("radius", 0),
                                                  kwargs.get("height", 0)),
        "cone_volume": lambda: cone_volume(kwargs.get("radius", 0),
                                           kwargs.get("height", 0)),
        "cuboid_volume": lambda: cuboid_volume(kwargs.get("length", 0),
                                             kwargs.get("width", 0),
                                             kwargs.get("height", 0)),

        # 矩阵和向量
        "matrix_operation": lambda: matrix_operation(kwargs.get("matrix1", []),
                                                    kwargs.get("matrix2"),
                                                    kwargs.get("operation", "determinant")),
        "vector_operation": lambda: vector_operation(kwargs.get("vector1", []),
                                                    kwargs.get("vector2"),
                                                    kwargs.get("operation", "magnitude")),
    }

    if func_name in dispatch:
        return dispatch[func_name]()
    else:
        return {"success": False, "result": None, "error": f"未知函数: {func_name}"}
