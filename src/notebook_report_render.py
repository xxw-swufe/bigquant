"""Render factor / condition research artifacts inside the notebook chat UI.

The chat agent already produces a full ``tool_result`` dict with the raw
factor-analysis / backtest statistics, but the notebook chat panel only
displays the LLM's free-form ``reply`` string. The actual Markdown report
and PNG charts produced by ``src/pipeline._build_report_artifacts`` were
never wired into the chat panel — this module bridges that gap.

The renderer is intentionally conservative:

* It renders Markdown as plain ``<pre>`` text. The chat panel lives inside
  an ``ipywidgets.Output`` widget which has no Markdown interpreter, but
  preserving newlines and indentation keeps the report readable.
* PNGs are inlined as ``<img src="file://...">`` tags. The browser fetches
  them lazily; we deliberately avoid ``base64`` to keep the transcript
  cheap to render even with multi-megabyte charts.
* Any failure here is logged to stderr and the renderer falls back to a
  one-line placeholder so the chat panel still looks tidy.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

# NOTE: _build_report_artifacts lives in src.pipeline and we import it
# lazily inside build_report_html to avoid a circular import:
# pipeline -> notebook_chat_ui -> notebook_report_render -> pipeline.


_PLACEHOLDER = (
    "<div style='margin:0 0 12px 0; padding:8px 12px; "
    "background:#fff7e6; border:1px solid #f0c36d; border-radius:6px; "
    "color:#7a5a00; font-size:12px;'>"
    "（未生成研究报告和图表，请查看 outputs 目录）"
    "</div>"
)


def _safe_file_uri(absolute_path: str) -> str:
    """Return a ``file://`` URI safe to drop into an ``<img src>`` tag."""
    return "file://" + escape(str(Path(absolute_path).resolve()))


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


def _image_grid_html(plot_paths: list[str]) -> str:
    """Render a responsive grid of ``<img>`` tags for the report's PNGs."""
    if not plot_paths:
        return ""
    cells: list[str] = []
    for p in plot_paths:
        uri = _safe_file_uri(p)
        stem = escape(Path(p).stem)
        cells.append(
            "<div style='margin:0 0 8px 0; padding:8px; "
            "background:#fff; border:1px solid #eee; border-radius:6px;'>"
            f"<div style='font-size:11px; color:#666; margin-bottom:4px;'>{stem}</div>"
            f"<img src='{uri}' style='max-width:100%; height:auto; display:block; "
            "border-radius:4px;'/>"
            "</div>"
        )
    return (
        "<div style='margin:0 0 12px 0;'>"
        + "".join(cells)
        + "</div>"
    )


def build_report_html(
    tool_result: dict[str, Any],
    task_type: str,
    outputs_dir: str | Path = "outputs",
) -> str:
    """Generate and render the full chat-panel HTML for one research turn.

    Returns an empty string when ``task_type`` is not a research pipeline,
    so the chat UI can call this unconditionally without branching.
    """
    if task_type not in ("factor_research", "condition_research"):
        return ""
    try:
        from src.pipeline import _build_report_artifacts  # lazy: avoid circular import

        artifacts = _build_report_artifacts(tool_result, task_type, outputs_dir)
    except Exception as exc:  # noqa: BLE001 — never crash the chat panel
        print(f"[notebook_report_render] failed to build artifacts: {exc!r}")
        return _PLACEHOLDER

    plot_paths = artifacts.get("plot_paths") or []
    report_md = artifacts.get("report_markdown") or ""
    if not plot_paths and not report_md:
        return _PLACEHOLDER

    header = (
        "<div style='margin:0 0 8px 0; padding:6px 10px; "
        "background:#eef6ff; border:1px solid #b9d6ff; "
        "border-radius:6px; font-size:12px; color:#1a4a7a;'>"
        f"已生成 {len(plot_paths)} 张图表和研究报告。"
        "</div>"
    )
    return header + _image_grid_html(plot_paths) + _markdown_to_pre(report_md)
