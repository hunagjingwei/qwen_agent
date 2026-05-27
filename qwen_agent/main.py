#!/usr/bin/env python3
"""Qwen3.5-4B Agent CLI 入口"""
import sys
import os

# Add parent directory to path so 'qwen_agent' imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.core import Agent

console = Console()

MODEL_PATH = "/home/wujie/LLM_project/Qwen3.5-4B"


def truncate_history(messages, max_rounds=2):
    """截断对话历史，保留 system prompt + 最近 N 轮完整对话

    Args:
        messages: 对话消息列表
        max_rounds: 最多保留的对话轮数（默认2轮，避免token超限）

    Returns:
        截断后的消息列表
    """
    if not messages:
        return messages

    # 始终保留 system prompt（第一条消息）
    system_prompt = messages[0] if messages[0].get("role") == "system" else None

    # 如果没有 system prompt 或只有 system prompt，直接返回
    if system_prompt is None or len(messages) == 1:
        return messages

    # 提取非 system 消息
    non_system = messages[1:]

    # 找到所有 user message 的位置（每轮对话的开始）
    user_positions = []
    for i, msg in enumerate(non_system):
        if msg.get("role") == "user":
            user_positions.append(i)

    # 如果对话轮数未超过限制，直接返回
    if len(user_positions) <= max_rounds:
        return messages

    # 计算需要保留的起始位置
    # 保留最后 max_rounds 轮对话
    start_pos = user_positions[-max_rounds]

    # 重新构建消息列表
    result = [system_prompt] + non_system[start_pos:]

    print(f"[DEBUG] History truncated: {len(messages)} -> {len(result)} messages ({len(user_positions)} -> {max_rounds} rounds)", file=sys.stderr)

    return result


def main():
    console.print(Panel("[bold blue]Qwen3.5-4B Agent[/bold blue]\n基于 Function Calling 的智能助手"))
    console.print("[dim]输入 'clear' 清空对话历史，输入 'exit' 退出[/dim]\n")

    try:
        agent = Agent(
            model_path=MODEL_PATH,
            tensor_parallel_size=1,
            max_model_len=8192,
            gpu_memory_utilization=0.9
        )
        console.print("[green]模型加载成功！[/green]\n")
    except Exception as e:
        console.print(f"[red]模型加载失败: {e}[/red]")
        sys.exit(1)

    conversation_history = []
    MAX_ROUNDS = 2  # 最多保留2轮对话（避免token超限）

    while True:
        try:
            user_input = console.input("[bold cyan]>>> [/bold cyan]")

            # 处理特殊命令
            if user_input.lower() in ["exit", "quit"]:
                console.print("[yellow]再见！[/yellow]")
                break

            if user_input.lower() == "clear":
                conversation_history = []
                console.print("[yellow]对话历史已清空[/yellow]\n")
                continue

            if not user_input.strip():
                continue

            with console.status("[bold green]思考中...[/bold green]"):
                result = agent.run(user_input, conversation_history)

            # Debug: 打印结果
            print(f"[DEBUG] result keys: {result.keys()}", file=sys.stderr)
            print(f"[DEBUG] response length: {len(result.get('response', ''))}", file=sys.stderr)
            print(f"[DEBUG] response preview: {result.get('response', '')[:300]}", file=sys.stderr)
            print(f"[DEBUG] messages count: {len(result.get('messages', []))}", file=sys.stderr)
            if result.get('tool_calls'):
                tc = result['tool_calls'][0]
                print(f"[DEBUG] tool_call name: {tc.get('name')}", file=sys.stderr)
                args = tc.get('arguments', {})
                print(f"[DEBUG] tool_call args keys: {args.keys()}", file=sys.stderr)
                if 'code' in args:
                    code_preview = args['code'][:100] if args['code'] else 'EMPTY'
                    print(f"[DEBUG] code preview: {code_preview}...", file=sys.stderr)

            conversation_history = truncate_history(result["messages"], MAX_ROUNDS)

            response = result["response"]
            console.print(Panel(response, title="[bold green]Assistant[/bold green]", border_style="green"))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]再见！[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]错误: {e}[/red]")
            console.print()

if __name__ == "__main__":
    main()
