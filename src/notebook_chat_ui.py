"""ipywidgets-based notebook chat UI for AutoETF."""

from __future__ import annotations

import contextlib
from html import escape
import io
import traceback
from typing import Callable, Optional

from src.chat_state import create_chat_state


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


def _append_chat_message(output, label: str, text: str, *, color: str = "#111") -> None:
    """Append a wrapped chat message into an ipywidgets Output area."""
    from IPython.display import HTML, display

    safe_text = escape(text).replace("\n", "<br>")
    block = (
        "<div style='margin:0 0 12px 0; line-height:1.65; "
        "white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word;'>"
        f"<strong style='color:{color};'>{escape(label)}</strong>"
        f"<span style='color:{color};'>{safe_text}</span>"
        "</div>"
    )
    with output:
        display(HTML(block))


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
        from IPython.display import display
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
            width="100%",
            overflow_y="visible",
            overflow_x="hidden",
        )
    )
    status_label = widgets.HTML("")

    def handle_send(_=None):
        nonlocal chat_state
        user_input = input_box.value.strip()
        if not user_input:
            return

        send_button.disabled = True
        reset_button.disabled = True
        status_label.value = "<span style='color:#555'>运行中...</span>"
        input_box.value = ""
        _append_chat_message(transcript_output, "你：", user_input, color="#111")
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
            if not turn["ok"]:
                _append_chat_message(transcript_output, "发生错误：", turn["error"], color="#d93025")
                status_label.value = "<span style='color:#d93025'>出错</span>"
                return

            result = turn["result"]
            chat_state = result["state"]
            _append_chat_message(transcript_output, "回答：", result["reply"], color="#111")
            status_label.value = "<span style='color:#188038'>完成</span>"
        except Exception as exc:
            _append_chat_message(transcript_output, "发生错误：", f"{type(exc).__name__}: {exc}", color="#d93025")
            status_label.value = "<span style='color:#d93025'>出错</span>"
        finally:
            send_button.disabled = False
            reset_button.disabled = False

    def handle_reset(_=None):
        nonlocal chat_state
        chat_state = create_chat_state(config)
        input_box.value = ""
        transcript_output.clear_output(wait=True)
        status_label.value = ""
        _append_chat_message(transcript_output, "系统：", "上下文已重置。", color="#188038")

    send_button.on_click(handle_send)
    reset_button.on_click(handle_reset)
    input_box.on_submit(lambda _: handle_send())

    controls = widgets.HBox([send_button, reset_button])
    panel_children = [
        title,
        instruction,
        input_box,
        controls,
        status_label,
        widgets.HTML("<h4>聊天记录</h4>"),
        transcript_output,
    ]
    chat_panel = widgets.VBox(panel_children)
    display(chat_panel)

    return {
        "state": chat_state,
        "input_box": input_box,
        "output": transcript_output,
        "send_button": send_button,
        "reset_button": reset_button,
        "status_label": status_label,
        "chat_panel": chat_panel,
    }
