"""ipywidgets-based notebook chat UI for AutoETF."""

from __future__ import annotations

import contextlib
from html import escape
import io
from time import monotonic
import traceback
from typing import Callable, Optional

from src.chat_state import create_chat_state
from src.notebook_report_render import build_report_html


DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
_ACTIVE_CHAT_UI_NS_KEY = "_autoetf_active_chat_ui"
_ACTIVE_CHAT_UI = None
_SEND_IN_FLIGHT = False


def _ipython_user_ns() -> dict | None:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            return ip.user_ns
    except Exception:
        return None
    return None


def _get_active_chat_ui() -> dict | None:
    ns = _ipython_user_ns()
    if ns is not None and ns.get(_ACTIVE_CHAT_UI_NS_KEY) is not None:
        return ns.get(_ACTIVE_CHAT_UI_NS_KEY)
    return _ACTIVE_CHAT_UI


def _set_active_chat_ui(chat_ui: dict | None) -> None:
    global _ACTIVE_CHAT_UI
    _ACTIVE_CHAT_UI = chat_ui
    ns = _ipython_user_ns()
    if ns is not None:
        if chat_ui is None:
            ns.pop(_ACTIVE_CHAT_UI_NS_KEY, None)
        else:
            ns[_ACTIVE_CHAT_UI_NS_KEY] = chat_ui


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
        "white-space:pre-wrap; overflow-wrap:anywhere; word-break:break-word; "
        "width:100%; box-sizing:border-box;'>"
        f"<strong style='color:{color};'>{escape(label)}</strong>"
        f"<span style='color:{color};'>{safe_text}</span>"
        "</div>"
    )
    with output:
        display(HTML(block))


def _render_report_html_into_output(output, html_block: str) -> None:
    """Render a chunk of HTML into the given ipywidgets ``Output`` widget.

    The renderer is isolated so tests can monkeypatch it without wrestling
    with the ``IPython.display`` import dance inside ``handle_send``.
    """
    if not html_block:
        return
    from IPython.display import HTML, display

    with output:
        display(HTML(html_block))


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

    global _ACTIVE_CHAT_UI
    _close_active_chat_ui()

    chat_state = create_chat_state(config)
    if chat_runner is None:
        from src.chat_agent import run_research_chat

        runner = run_research_chat
    else:
        runner = chat_runner

    chat_instance_id = monotonic()

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
            height="auto",
            overflow_y="visible",
            overflow_x="hidden",
        )
    )
    status_label = widgets.HTML("")
    last_submission = {"text": None, "time": 0.0}

    def _on_input_change(change):
        # Re-enable the send button whenever the user types new text, so an
        # accidental second click on the same text doesn't fire a duplicate
        # submission. The button is auto-disabled again at submit time.
        if change.get("name") == "value" and change.get("new") != change.get("old"):
            send_button.disabled = False

    input_box.observe(_on_input_change, names="value")

    def handle_send(_=None):
        nonlocal chat_state
        global _SEND_IN_FLIGHT
        if _get_active_chat_ui() is not chat_ui:
            return
        user_input = input_box.value.strip()
        if not user_input:
            return
        now = monotonic()
        # Stronger dedup: same text within 5s (covers double-click / impatient
        # user re-presses) and busy guard against concurrent calls.
        if last_submission["text"] == user_input and (now - last_submission["time"]) < 5.0:
            status_label.value = "<span style='color:#888'>已发送过同样的内容，请修改后再发送</span>"
            return
        if _SEND_IN_FLIGHT or getattr(handle_send, "_busy", False):
            return

        _SEND_IN_FLIGHT = True
        handle_send._busy = True
        last_submission["text"] = user_input
        last_submission["time"] = now
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
            # Render the full Markdown report + chart grid inline so the
            # chat panel no longer hides the actual research outputs.
            tool_name = result.get("tool_name")
            if tool_name == "run_factor_research_pipeline":
                chat_task_type = "factor_research"
            elif tool_name == "run_condition_research_pipeline":
                chat_task_type = "condition_research"
            else:
                chat_task_type = None
            tool_result = result.get("tool_result") or {}
            if chat_task_type and tool_result:
                html_block = build_report_html(
                    tool_result,
                    chat_task_type,
                    outputs_dir=getattr(config, "outputs_dir", "outputs"),
                )
                _render_report_html_into_output(transcript_output, html_block)
            status_label.value = "<span style='color:#188038'>完成</span>"
        except Exception as exc:
            _append_chat_message(transcript_output, "发生错误：", f"{type(exc).__name__}: {exc}", color="#d93025")
            status_label.value = "<span style='color:#d93025'>出错</span>"
        finally:
            _SEND_IN_FLIGHT = False
            handle_send._busy = False
            send_button.disabled = False
            reset_button.disabled = False

    def handle_reset(_=None):
        nonlocal chat_state
        if _get_active_chat_ui() is not chat_ui:
            return
        chat_state = create_chat_state(config)
        input_box.value = ""
        transcript_output.clear_output(wait=True)
        status_label.value = ""
        _append_chat_message(transcript_output, "系统：", "上下文已重置。", color="#188038")

    send_button.on_click(handle_send)
    reset_button.on_click(handle_reset)

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

    chat_ui = {
        "state": chat_state,
        "input_box": input_box,
        "output": transcript_output,
        "send_button": send_button,
        "reset_button": reset_button,
        "status_label": status_label,
        "chat_panel": chat_panel,
        "_handle_send": handle_send,
        "_handle_reset": handle_reset,
        "_instance_id": chat_instance_id,
    }
    _set_active_chat_ui(chat_ui)
    return chat_ui


def _close_active_chat_ui() -> None:
    """Close the previous widget tree so repeated launches don't stack duplicate UIs."""
    global _SEND_IN_FLIGHT
    active = _get_active_chat_ui()
    if not active:
        return
    send_button = active.get("send_button")
    reset_button = active.get("reset_button")
    handle_send = active.get("_handle_send")
    handle_reset = active.get("_handle_reset")
    if send_button is not None and handle_send is not None:
        try:
            send_button.on_click(handle_send, remove=True)
        except Exception:
            pass
    if reset_button is not None and handle_reset is not None:
        try:
            reset_button.on_click(handle_reset, remove=True)
        except Exception:
            pass
    for key in ("send_button", "reset_button", "input_box", "output", "status_label"):
        widget = active.get(key)
        if widget is not None:
            try:
                widget.close()
            except Exception:
                pass
    panel = active.get("chat_panel")
    if panel is not None:
        try:
            panel.close()
        except Exception:
            pass
    _set_active_chat_ui(None)
    _SEND_IN_FLIGHT = False
