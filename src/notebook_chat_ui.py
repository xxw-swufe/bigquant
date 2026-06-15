"""ipywidgets-based notebook chat UI for AutoETF."""

from __future__ import annotations

import json
from typing import Any, Optional

from src.chat_agent import DEFAULT_DEEPSEEK_MODEL, run_research_chat
from src.chat_state import create_chat_state, format_context_for_display, suggest_follow_up


def launch_notebook_chat(
    config,
    *,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
):
    """Render a minimal notebook chat box with persistent state."""
    try:
        import ipywidgets as widgets
        from IPython.display import Markdown, display
    except ImportError as exc:  # pragma: no cover - notebook-only dependency
        raise RuntimeError("ipywidgets / IPython are required for notebook chat UI.") from exc

    chat_state = create_chat_state(config)

    title = widgets.HTML("<h3>AutoETF Notebook Chat</h3>")
    instruction = widgets.HTML(
        "<p>输入研究问题后点击发送。你可以继续追问，例如‘加上成交额上升’、‘换成未来10日’、‘改成放量突破’。</p>"
    )
    input_box = widgets.Textarea(
        value="",
        placeholder="例如：缩量上涨后，ETF 未来5日上涨概率怎么样？",
        description="问题",
        layout=widgets.Layout(width="100%", height="88px"),
    )
    send_button = widgets.Button(description="发送", button_style="primary")
    reset_button = widgets.Button(description="重置上下文", button_style="")
    output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="10px"))
    context_output = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", padding="10px"))

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

        output.clear_output(wait=True)
        with output:
            display(Markdown(f"**你：** {user_input}"))
            result = run_research_chat(
                user_input=user_input,
                config=config,
                use_llm=use_llm,
                api_key=api_key,
                model=model,
                state=chat_state,
            )
            chat_state = result["state"]
            display(Markdown(f"**工具：** `{result['tool_name']}`"))
            display(Markdown(f"**路由：** {result['decision'].get('reason')}"))
            display(Markdown("**回答：**"))
            display(Markdown(result["reply"]))
        render_context()
        input_box.value = ""

    def handle_reset(_=None):
        nonlocal chat_state
        chat_state = create_chat_state(config)
        input_box.value = ""
        output.clear_output(wait=True)
        render_context()
        with output:
            display(Markdown("**上下文已重置。**"))

    send_button.on_click(handle_send)
    reset_button.on_click(handle_reset)
    input_box.on_submit(lambda _: handle_send())

    controls = widgets.HBox([send_button, reset_button])
    display(widgets.VBox([title, instruction, input_box, controls, output, context_output]))
    render_context()

    return {
        "state": chat_state,
        "input_box": input_box,
        "output": output,
        "context_output": context_output,
        "send_button": send_button,
        "reset_button": reset_button,
    }
