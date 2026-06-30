"""Render the research Markdown report inside the notebook chat UI.

The chat agent already produces a full ``tool_result`` dict with the raw
factor / condition analysis, but the notebook chat panel only displayed
the LLM's free-form ``reply`` string. This module bridges that gap by
showing the full Markdown report directly in the transcript.

Design choice: charts are NOT inlined in the chat panel. Inline PNGs bloat
the transcript and make it slow to scroll. Instead, ``build_report_html``
returns a small banner pointing the user at the ``outputs/`` folder where
the PNGs already live on disk, followed by the Markdown report rendered
as a ``<pre>`` block for readability.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _markdown_to_pre(markdown: str) -> str:
    """Render Markdown as an indented ``<pre>`` block for chat transcripts."""
    return (
        "<pre style='margin:0 0 12px 0; padding:12px; "
        "background:#fafafa; border:1px solid #e5e5e5; "
        "border-radius:6px; max-height:480px; overflow:auto; "
        "white-space:pre-wrap; word-break:break-word; "
        "font-family:ui-monospace, SFMono-Regular, Menlo, monospace; "
        "font-size:12px; color:#222;'>"
        f"{escape(markdown or '')}"
        "</pre>"
    )


def build_report_html(
    tool_result: dict[str, Any],
    task_type: str,
    outputs_dir: str | Path = "outputs",
) -> str:
    """Generate the chat-panel HTML for one research turn.

    Returns an empty string when ``task_type`` is not a research pipeline.
    """
    if task_type not in ("factor_research", "condition_research"):
        return ""
    try:
        # Lazy import: avoid a circular pipeline -> chat_ui -> renderer -> pipeline.
        from src.pipeline import _build_report_artifacts

        artifacts = _build_report_artifacts(tool_result, task_type, outputs_dir)
    except Exception as exc:  # noqa: BLE001 — never crash the chat panel
        print(f"[notebook_report_render] failed to build artifacts: {exc!r}")
        return ""

    plot_paths = artifacts.get("plot_paths") or []
    report_md = artifacts.get("report_markdown") or ""
    if not plot_paths and not report_md:
        return ""

    charts_dir = Path(outputs_dir).expanduser()
    if not charts_dir.is_absolute():
        charts_dir = (Path(__file__).resolve().parent.parent / charts_dir).resolve()
    chart_count = len(plot_paths)
    banner = (
        "<div style='margin:0 0 12px 0; padding:8px 12px; "
        "background:#eef6ff; border:1px solid #b9d6ff; "
        "border-radius:6px; font-size:12px; color:#1a4a7a;'>"
        f"已生成 {chart_count} 张图表，存放于本地目录："
        f"<code style='background:#fff; padding:2px 6px; border-radius:4px; "
        f"margin-left:4px;'>{escape(str(charts_dir))}</code>"
        "</div>"
    )
    return banner + _markdown_to_pre(report_md)
