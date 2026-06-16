"""ipywidgets-based notebook chat UI for AutoETF."""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from typing import Callable, Optional

from src.chat_state import create_chat_state, format_context_for_display, suggest_follow_up


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def run_chat_turn_with_captured_logs(
    runner: Callable,
    *,
    user_input: str,
    config,
    use_llm: bool,
    api_key: Optional[str],
    model: str,
    state: dict,
) -> dict:
    """Run one chat turn while keeping tool stdout/stderr out of the chat UI."""
    log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            result = runner(
                user_input=user_input,
                config=config,
                use_llm=use_llm,
                api_key=api_key,
                model=model,
                state=state,
            )
        return {
            "ok": True,
            "result": result,
            "logs": log_buffer.getvalue().strip(),
            "error": None,
            "traceback": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "result": None,
            "logs": log_buffer.getvalue().strip(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def launch_notebook_chat(
    config,
    *,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    chat_runner: Optional[Callable] = None,
):
    """Render a minimal notebook chat box with persistent state."""
    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
    except ImportError as exc:  # pragma: no cover - notebook-only dependency
        raise RuntimeError("ipywidgets / IPython are required for notebook chat UI.") from exc

    chat_state = create_chat_state(config)
    if chat_runner is None:
        from src.chat_agent import run_research_chat

        runner = run_research_chat
    else:
        runner = chat_runner

    title = widgets.HTML("<h3>AutoETF Notebook Chat</h3>")
    instruction = widgets.HTML(
        "<p>输入研究问题后点击发送。支持连续追问，例如‘加上成交额上升’、‘换成未来10日’、‘改成放量突破’。</p>"
    )
    input_box = widgets.Text(
        value="",
        placeholder="例如：缩量上涨后，ETF 未来1日上涨概率怎么样？",
        description="问题",
        layout=widgets.Layout(width="100%"),
    )
    send_button = widgets.Button(description="发送", button_style="primary")
    reset_button = widgets.Button(description="重置上下文", button_style="")
    transcript_output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="10px",
            height="360px",
            overflow_y="auto",
        )
    )
    log_output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="10px",
            height="180px",
            overflow_y="auto",
        )
    )
    context_output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #ddd",
            padding="10px",
            height="360px",
            overflow_y="auto",
        )
    )
    status_label = widgets.HTML("")

    def render_context() -> None:
        context_output.clear_output(wait=True)
        with context_output:
            display(Markdown("### 当前上下文"))
            display(Markdown(f"```json\n{format_context_for_display(chat_state)}\n```"))
            suggestions = suggest_follow_up(chat_state)
            if suggestions:
                display(Markdown("### 可继续追问"))
                display(Markdown("\n".join(f"- {item}" for item in suggestions)))
            last_summary = chat_state.get("current_context", {}).get("last_result_summary")
            if last_summary:
                display(Markdown("### 上轮结果摘要"))
                display(Markdown(f"```json\n{json.dumps(last_summary, ensure_ascii=False, indent=2, default=str)}\n```"))

    def handle_send(_=None):
        nonlocal chat_state
        user_input = input_box.value.strip()
        if not user_input:
            return

        send_button.disabled = True
        reset_button.disabled = True
        status_label.value = "<span style='color:#555'>运行中...</span>"
        input_box.value = ""
        with transcript_output:
            display(Markdown(f"**你：** {user_input}"))
        try:
            turn = run_chat_turn_with_captured_logs(
                runner,
                user_input=user_input,
                config=config,
                use_llm=use_llm,
                api_key=api_key,
                model=model,
                state=chat_state,
            )
            if turn["logs"]:
                with log_output:
                    display(Markdown(f"### {user_input}"))
                    display(Markdown(f"```text\n{turn['logs']}\n```"))

            if not turn["ok"]:
                with transcript_output:
                    display(Markdown("**发生错误：**"))
                    display(Markdown(f"```text\n{turn['error']}\n```"))
                with log_output:
                    display(Markdown(f"### {user_input}"))
                    display(Markdown(f"```text\n{turn['traceback']}\n```"))
                status_label.value = "<span style='color:#d93025'>出错，详情见工具日志</span>"
                return

            result = turn["result"]
            chat_state = result["state"]
            with transcript_output:
                display(Markdown(f"**工具：** `{result['tool_name']}`"))
                display(Markdown(f"**路由：** {result['decision'].get('reason')}"))
                display(Markdown("**回答：**"))
                display(Markdown(result["reply"]))
            render_context()
            status_label.value = "<span style='color:#188038'>完成</span>"
        except Exception as exc:
            with transcript_output:
                display(Markdown("**发生错误：**"))
                display(Markdown(f"```text\n{type(exc).__name__}: {exc}\n```"))
            with log_output:
                display(Markdown(f"### {user_input}"))
                display(Markdown(f"```text\n{traceback.format_exc()}\n```"))
            status_label.value = "<span style='color:#d93025'>出错，详情见工具日志</span>"
        finally:
            send_button.disabled = False
            reset_button.disabled = False

    def handle_reset(_=None):
        nonlocal chat_state
        chat_state = create_chat_state(config)
        input_box.value = ""
        transcript_output.clear_output(wait=True)
        log_output.clear_output(wait=True)
        status_label.value = ""
        render_context()
        with transcript_output:
            display(Markdown("**上下文已重置。**"))

    send_button.on_click(handle_send)
    reset_button.on_click(handle_reset)
    input_box.on_submit(lambda _: handle_send())

    controls = widgets.HBox([send_button, reset_button])
    chat_panel = widgets.VBox(
        [
            title,
            instruction,
            input_box,
            controls,
            status_label,
            widgets.HTML("<h4>聊天记录</h4>"),
            transcript_output,
            widgets.HTML("<h4>工具日志</h4>"),
            log_output,
            widgets.HTML("<h4>当前上下文</h4>"),
            context_output,
        ]
    )
    display(chat_panel)
    render_context()

    return {
        "state": chat_state,
        "input_box": input_box,
        "output": transcript_output,
        "log_output": log_output,
        "context_output": context_output,
        "send_button": send_button,
        "reset_button": reset_button,
        "status_label": status_label,
        "chat_panel": chat_panel,
    }
