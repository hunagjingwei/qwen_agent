#!/usr/bin/env python3
"""Qwen3.5-4B Agent CLI 入口"""
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from agent.core import Agent

console = Console()

MODEL_PATH = "/home/wujie/LLM_project/Qwen3.5-4B"

def main():
    console.print(Panel("[bold blue]Qwen3.5-4B Agent[/bold blue]\n基于 Function Calling 的智能助手"))
    console.print()

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

    while True:
        try:
            user_input = console.input("[bold cyan]>>> [/bold cyan]")

            if user_input.lower() in ["exit", "quit", "退出"]:
                console.print("[yellow]再见！[/yellow]")
                break

            if not user_input.strip():
                continue

            with console.status("[bold green]思考中...[/bold green]"):
                result = agent.run(user_input, conversation_history)
                conversation_history = result["messages"]

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
