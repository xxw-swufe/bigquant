"""Test that the notebook chat UI injects the report HTML after a research turn."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.notebook_chat_ui import launch_notebook_chat
from src.config import DEFAULT_CONFIG


class _RecorderWidget:
    """Minimal stand-in for ``ipywidgets.Output`` that records what was displayed."""

    def __init__(self) -> None:
        self.displayed: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Recorder:
    """Records every widget creation and every click handler."""

    def __init__(self) -> None:
        self.outputs: list[_RecorderWidget] = []

    def Output(self, **kwargs):
        rec = _RecorderWidget()
        self.outputs.append(rec)
        return rec

    def HTML(self, value=""):
        self.last_html = value
        return SimpleNamespace(value=value)

    def Text(self, **kwargs):
        self.last_text = kwargs
        return SimpleNamespace(
            value="",
            layout=kwargs.get("layout"),
            observe=lambda *a, **kw: None,
            _click_handlers=[],
        )

    def Button(self, **kwargs):
        click_handlers: list = []

        def on_click(func, **kw):
            click_handlers.append(func)

        return SimpleNamespace(
            description=kwargs.get("description", ""),
            button_style=kwargs.get("button_style", ""),
            disabled=False,
            on_click=on_click,
            _click_handlers=click_handlers,
        )

    def HBox(self, children):
        return SimpleNamespace(children=tuple(children))

    def VBox(self, children):
        self.last_panel = SimpleNamespace(children=tuple(children))
        return self.last_panel

    def observe(self, *args, **kwargs):
        return None

    def on_click(self, *args, **kwargs):
        return None


def _install_ipywidgets_stub(monkeypatch, recorder: _Recorder) -> None:
    """Pretend ``ipywidgets`` is importable and route every factory to the recorder."""
    def fake_layout(**kwargs):
        return SimpleNamespace(**kwargs)

    fake_widgets = SimpleNamespace(
        HTML=recorder.HTML,
        Text=recorder.Text,
        Button=recorder.Button,
        HBox=recorder.HBox,
        VBox=recorder.VBox,
        Output=recorder.Output,
        Layout=fake_layout,
    )
    monkeypatch.setitem(__import__("sys").modules, "ipywidgets", fake_widgets)


def _install_ipython_display_stub(monkeypatch, transcript_widget: _RecorderWidget) -> None:
    """Stub IPython.display.display so it records calls when transcript is targeted."""

    calls: list[str] = []

    def fake_display(obj=None, **kwargs):
        if obj is None:
            return
        rendered = getattr(obj, "data", None) or getattr(obj, "value", None) or str(obj)
        if "autoetf" in str(type(obj)).lower() or "<img" in str(rendered) or "<pre" in str(rendered):
            transcript_widget.displayed.append(str(rendered))
        calls.append(str(rendered))

    fake_display_module = SimpleNamespace(
        display=fake_display,
        HTML=lambda data: SimpleNamespace(data=data, _tag="html"),
    )
    monkeypatch.setitem(__import__("sys").modules, "IPython.display", fake_display_module)


def _fake_factor_chat_result(tmp_path: Path) -> dict:
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    ic_series = pd.Series([0.01] * 19, index=dates[1:])
    quantile_return = pd.Series([0.001, 0.002, 0.003, 0.004, 0.005], index=[1, 2, 3, 4, 5])
    daily_returns = pd.Series([0.001] * 19, index=dates[1:])
    factor_result = {
        "rsi_14d": {
            "future_return_5d": {
                "ic_series": ic_series,
                "quantile_return": quantile_return,
            }
        }
    }
    backtest_result = {
        "daily_returns": pd.DataFrame({"return": daily_returns}),
        "cumulative_return": 0.05,
        "annual_return": 0.06,
        "max_drawdown": -0.05,
        "sharpe": 0.5,
        "win_rate": 0.55,
        "annualized_turnover": 1.0,
        "total_cost": 0.001,
    }
    return {
        "user_input": "研究 RSI",
        "decision": {"tool_name": "run_factor_research_pipeline"},
        "tool_name": "run_factor_research_pipeline",
        "tool_result": {
            "factor_result": factor_result,
            "backtest_result": backtest_result,
            "diagnosis": {"summary": "ok", "research_decision": "继续研究"},
            "report": "# AutoETF\n\n## 5. 单因子分析结果\n## 6. Top 10% 简化回测结果\n",
        },
        "reply": "基于工具结果关于 RSI 的研究结论…",
        "state": {},
    }


def test_chat_ui_injects_report_html_after_factor_research(monkeypatch, tmp_path: Path) -> None:
    recorder = _Recorder()
    _install_ipywidgets_stub(monkeypatch, recorder)

    # The transcript_output is created during launch_notebook_chat; we install
    # the IPython.display stub *after* the launch but the renderer runs lazily
    # at handle_send time, so by then the stub is in place.
    cfg = DEFAULT_CONFIG  # outputs_dir is not part of the config; the chat UI
    # falls back to the default "outputs" path. We monkeypatch
    # src.pipeline._resolve_outputs_dir to redirect everything to tmp_path.

    # Patch run_chat_turn_with_captured_logs to return our fake chat result.
    import src.notebook_chat_ui as chat_ui_module

    def fake_turn(runner, *, user_input, config, use_llm, api_key, model, state):
        return {
            "ok": True,
            "result": _fake_factor_chat_result(tmp_path),
            "logs": "",
            "error": None,
            "traceback": None,
        }

    monkeypatch.setattr(chat_ui_module, "run_chat_turn_with_captured_logs", fake_turn)

    import src.pipeline as pipeline_module

    def fake_resolve(outputs_dir):
        from pathlib import Path
        p = Path(str(outputs_dir))
        if not p.is_absolute():
            p = (Path(__file__).resolve().parent.parent / p).resolve()
        p = tmp_path if str(p).endswith("/outputs") or p.name == "outputs" else p
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(pipeline_module, "_resolve_outputs_dir", fake_resolve)

    # Launch the chat panel (this will wire up the button click handler internally).
    chat_ui_obj = launch_notebook_chat(cfg, chat_runner=lambda **kwargs: {})
    assert chat_ui_obj is not None

    transcript_widget = recorder.outputs[0]  # captured for compatibility; unused after refactor

    # Stub _render_report_html_into_output so we can capture the HTML the chat
    # UI would have injected into the transcript widget.
    rendered_html: list[str] = []

    def fake_render(output, html_block):
        if html_block:
            rendered_html.append(html_block)

    monkeypatch.setattr(chat_ui_module, "_render_report_html_into_output", fake_render)

    # Populate the input box so handle_send does not early-return.
    chat_ui_obj["input_box"].value = "研究 RSI"

    # Trigger the send button click handler registered on the actual button widget.
    send_button = chat_ui_obj["send_button"]
    assert send_button._click_handlers, "send button should have at least one click handler"
    send_button._click_handlers[0]()  # type: ignore[attr-defined]

    assert rendered_html, "chat UI did not invoke the report renderer"
    rendered = " ".join(rendered_html)
    assert "<img" in rendered, f"chat UI did not embed <img>: {rendered_html}"
    assert "<pre" in rendered, f"chat UI did not embed <pre>: {rendered_html}"
    assert "file://" in rendered, f"chat UI did not embed file:// URIs: {rendered_html}"
