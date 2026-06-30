"""Unified single-turn research pipeline for AutoETF.

This module wraps the LLM-routed `run_research_chat` runner, persists the
produced markdown report to disk, generates the standard set of research
charts into the same directory, embeds image references into the markdown
report, and returns a structured result that is friendly for notebooks,
scripts, and tests.

Design notes:
- Pure orchestration: no new research logic is added here. The actual
  research (factor / condition) is delegated to the existing
  `tool_run_factor_research_pipeline` / `tool_run_condition_research_pipeline`
  inside `agent_tools.py`, which are invoked transitively through
  `run_research_chat`.
- The chat runner is injected via the `chat_runner` parameter to keep this
  module testable without spinning up LLM calls.
- Reports are written with a timestamped filename under `outputs_dir` so
  historical runs are never overwritten. PNGs share the same timestamp so
  the markdown file references them via basename-only relative paths.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import time
from typing import Any, Callable, Optional

from src.chat_agent import DEFAULT_DEEPSEEK_MODEL, run_research_chat
from src.chat_state import create_chat_state
from src.config import DEFAULT_CONFIG, ResearchConfig
from src.condition_report import save_condition_report
from src.notebook_chat_ui import run_chat_turn_with_captured_logs
from src import plotting
from src.report import save_report


FACTOR_TOOL = "run_factor_research_pipeline"
CONDITION_TOOL = "run_condition_research_pipeline"
DIRECT_REPLY_TOOL = "direct_reply"

TASK_TYPE_FACTOR = "factor_research"
TASK_TYPE_CONDITION = "condition_research"
TASK_TYPE_DIRECT = "direct_reply"
TASK_TYPE_ERROR = "error"


def _now_timestamp() -> str:
    """Return a filesystem-safe timestamp like 20260628_153045."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_outputs_dir(outputs_dir: str | Path) -> Path:
    """Resolve `outputs_dir` into an absolute, existing path in a portable way.

    The project is meant to run both on the user's laptop and inside the
    BigQuant AI Studio cloud notebook, where the current working directory
    can differ from the project root. We therefore:

    * Treat the supplied value as already-portable when it is an absolute path
      (we still normalise and ensure it exists).
    * Treat relative inputs as project-relative by anchoring them at the
      project root, which is the parent of the ``src/`` package this module
      lives in. This avoids hard-coding ``/Users/yangpeng/...`` and keeps the
      pipeline working when the notebook is launched from any cwd.
    * Expand ``~`` so users can still pass ``~/some/dir`` if they want.

    Parameters
    ----------
    outputs_dir
        Either an absolute or a relative path. Relative paths are anchored at
        the project root (parent of ``src/``), **not** at the current working
        directory, so the behaviour is stable across environments.
    """
    path = Path(outputs_dir).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent.parent / path).resolve()
    else:
        path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Markdown image injection
# ---------------------------------------------------------------------------


def _inject_images(markdown: str, anchor: str, captions: list[tuple[str, str]]) -> str:
    """Insert `![caption](relative_path)` lines after the first `## <anchor>` heading.

    ``captions`` is a list of ``(caption, relative_path)`` tuples; each is
    rendered as ``![caption](relative_path)``. If the anchor heading is not
    found, the markdown is returned unchanged (silent skip — never raise).
    """
    if not captions:
        return markdown
    pattern = re.compile(rf"^(##\s+{re.escape(anchor)}\s*$)", re.MULTILINE)
    match = pattern.search(markdown)
    if match is None:
        return markdown
    insert_at = match.end()
    # Skip the line break so images appear on their own lines after the heading.
    if insert_at < len(markdown) and markdown[insert_at] == "\n":
        insert_at += 1
    block_lines = [f"![{caption}]({rel_path})" for caption, rel_path in captions]
    block = "\n" + "\n".join(block_lines) + "\n\n"
    return markdown[:insert_at] + block + markdown[insert_at:]


def _to_relative_paths(abs_paths: list[str], report_path: Path) -> list[str]:
    """Convert absolute PNG paths to paths relative to the markdown file's directory."""
    report_dir = report_path.parent
    rels: list[str] = []
    for p in abs_paths:
        try:
            rels.append(str(Path(p).resolve().relative_to(report_dir)))
        except ValueError:
            # Different drive or outside report_dir — fall back to basename.
            rels.append(Path(p).name)
    return rels


# ---------------------------------------------------------------------------
# Chart generation per task type
# ---------------------------------------------------------------------------


def _generate_factor_plots(
    tool_result: dict[str, Any],
    output_dir: Path,
    ts: str,
) -> list[str]:
    """Generate factor-research PNGs and return absolute paths of files written."""
    written: list[str] = []
    basename = f"research_report_{ts}"
    factor_result = tool_result.get("factor_result") or {}
    backtest_result = tool_result.get("backtest_result") or {}

    written.extend(plotting.plot_factor_analysis_panel(
        factor_result, output_dir, basename=basename,
    ))
    written.extend(plotting.plot_backtest_panel(
        backtest_result, output_dir, basename=basename,
    ))
    written.extend(plotting.plot_factor_correlation_heatmap(
        factor_result, output_dir, basename=f"{basename}_factor_correlation",
    ))
    written.extend(plotting.plot_ic_distribution(
        factor_result, output_dir, basename=f"{basename}_ic_distribution",
    ))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for p in written:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _generate_condition_plots(
    tool_result: dict[str, Any],
    output_dir: Path,
    ts: str,
) -> list[str]:
    """Generate condition-research PNGs and return absolute paths of files written."""
    written: list[str] = []
    basename = f"condition_research_report_{ts}"
    result = tool_result.get("result") or {}

    written.extend(plotting.plot_condition_panel(
        result,
        output_dir,
        basename=basename,
        event_returns=None,
        baseline_returns=None,
    ))
    seen: set[str] = set()
    unique: list[str] = []
    for p in written:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _classify_tool_name(tool_name: Optional[str]) -> str:
    if tool_name == FACTOR_TOOL:
        return TASK_TYPE_FACTOR
    if tool_name == CONDITION_TOOL:
        return TASK_TYPE_CONDITION
    if tool_name == DIRECT_REPLY_TOOL:
        return TASK_TYPE_DIRECT
    return TASK_TYPE_ERROR


def _resolve_section_pairs(
    plot_paths: list[str],
    task_type: str,
) -> dict[str, list[tuple[str, str]]]:
    """Bucket generated plot paths by report section based on filename suffix.

    The Markdown report uses fixed section headers ("5. 单因子分析结果",
    "6. Top 10% 简化回测结果", "4. 次日上涨概率检验", etc.); we map each
    plot to the right anchor so that ``_inject_images`` can drop them in
    the right place.

    Caption text is rendered first by ``Path(p).stem`` so the chat UI gets
    the same filenames the Markdown uses.
    """
    sections: dict[str, list[tuple[str, str]]] = {
        "factor.ic_quantile": [],
        "factor.backtest": [],
        "condition.probability": [],
        "condition.return_distribution": [],
        "condition.yearly_stability": [],
        "condition.instrument_stability": [],
    }
    for p in plot_paths:
        stem = Path(p).stem
        rel = Path(p).name  # use basename so chat UI can co-locate with report
        if task_type == TASK_TYPE_FACTOR:
            if "_ic_" in stem or "_quantile_" in stem or "_factor_correlation" in stem or "_ic_distribution" in stem:
                sections["factor.ic_quantile"].append((stem, rel))
            elif "_cumulative_return" in stem or "_drawdown" in stem:
                sections["factor.backtest"].append((stem, rel))
        else:
            if "_probability_bar" in stem:
                sections["condition.probability"].append((stem, rel))
            elif "_return_distribution" in stem:
                sections["condition.return_distribution"].append((stem, rel))
            elif "_yearly_stability" in stem:
                sections["condition.yearly_stability"].append((stem, rel))
            elif "_instrument_stability" in stem:
                sections["condition.instrument_stability"].append((stem, rel))
    return sections


def _build_report_artifacts(
    tool_result: dict[str, Any],
    task_type: str,
    outputs_dir: str | Path,
    *,
    ts: str | None = None,
) -> dict[str, Any]:
    """Generate PNGs + Markdown for one research tool result.

    Shared by ``run_research_pipeline`` (which writes the report to disk and
    threads ``paths`` through its return value) and the notebook chat UI
    (which wants the same artifacts rendered inline). Operating on the
    raw ``tool_result`` keeps the chart generation in a single place.

    Returns ``{plot_paths, report_markdown, report_path, basename}``.
    """
    resolved_dir = _resolve_outputs_dir(outputs_dir)
    ts = ts or _now_timestamp()

    if task_type not in (TASK_TYPE_FACTOR, TASK_TYPE_CONDITION):
        return {
            "plot_paths": [],
            "report_markdown": "",
            "report_path": None,
            "basename": "",
        }

    if task_type == TASK_TYPE_FACTOR:
        filename = f"research_report_{ts}.md"
        plot_paths = _generate_factor_plots(tool_result, resolved_dir, ts)
        report_markdown = tool_result.get("report", "") if isinstance(tool_result, dict) else ""
        rels = _to_relative_paths(plot_paths, resolved_dir / filename)
        sections = _resolve_section_pairs(rels, task_type)
        if sections["factor.ic_quantile"]:
            report_markdown = _inject_images(report_markdown, "5. 单因子分析结果", sections["factor.ic_quantile"])
        if sections["factor.backtest"]:
            report_markdown = _inject_images(report_markdown, "6. Top 10% 简化回测结果", sections["factor.backtest"])
        save_report(report_markdown, str(resolved_dir / filename))
    else:
        filename = f"condition_research_report_{ts}.md"
        plot_paths = _generate_condition_plots(tool_result, resolved_dir, ts)
        report_markdown = tool_result.get("report", "") if isinstance(tool_result, dict) else ""
        rels = _to_relative_paths(plot_paths, resolved_dir / filename)
        sections = _resolve_section_pairs(rels, task_type)
        if sections["condition.probability"]:
            report_markdown = _inject_images(report_markdown, "4. 次日上涨概率检验", sections["condition.probability"])
        if sections["condition.return_distribution"]:
            report_markdown = _inject_images(report_markdown, "5. 次日平均收益检验", sections["condition.return_distribution"])
        if sections["condition.yearly_stability"]:
            report_markdown = _inject_images(report_markdown, "6. 分年份稳定性", sections["condition.yearly_stability"])
        if sections["condition.instrument_stability"]:
            report_markdown = _inject_images(report_markdown, "7. 诊断结论", sections["condition.instrument_stability"])
        save_condition_report(report_markdown, str(resolved_dir / filename))

    return {
        "plot_paths": plot_paths,
        "report_markdown": report_markdown,
        "report_path": str((resolved_dir / filename).resolve()),
        "basename": filename.rsplit(".", 1)[0],
    }


def run_research_pipeline(
    user_idea: str,
    config: Optional[ResearchConfig] = None,
    *,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model: str = DEFAULT_DEEPSEEK_MODEL,
    outputs_dir: str | Path = "outputs",
    state: Optional[dict[str, Any]] = None,
    chat_runner: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run one research turn end-to-end and persist the markdown report.

    Parameters
    ----------
    user_idea:
        Natural-language research request from the user.
    config:
        Optional ``ResearchConfig``. Defaults to ``DEFAULT_CONFIG``.
    use_llm, api_key, model:
        Forwarded to the chat runner for LLM-driven routing and summarisation.
    outputs_dir:
        Directory where the markdown report is written. Created if missing.
    state:
        Optional chat state. A fresh state is created when omitted.
    chat_runner:
        Optional callable used in place of ``run_research_chat``. Primarily
        intended for tests; receives keyword arguments ``user_input``,
        ``config``, ``use_llm``, ``api_key``, ``model``, ``state``.

    Returns
    -------
    dict with the following keys:

    - ``user_idea`` (str)
    - ``task_type`` (str): one of ``factor_research`` / ``condition_research``
      / ``direct_reply`` / ``error``
    - ``decision`` (dict | None): raw LLM/rule decision from the chat runner
    - ``tool_name`` (str | None): tool that was dispatched
    - ``tool_result`` (dict | None): raw return from the research tool
    - ``report_markdown`` (str): the full markdown report; empty when no
      research ran
    - ``report_path`` (str | None): absolute path of the persisted report;
      ``None`` for ``direct_reply`` / ``error``. Image references inside
      the report use paths relative to the report's parent directory.
    - ``plot_paths`` (list[str]): absolute paths of every PNG chart produced
      for this run (empty for ``direct_reply`` / ``error``). The same
      files are referenced inside ``report_markdown`` via relative paths.
    - ``reply`` (str | None): the chat-runner reply (LLM summary)
    - ``state`` (dict | None): the chat state after the turn
    - ``duration_sec`` (float)
    - ``error`` (str | None): error message when ``task_type == "error"``
    """
    effective_config = config or DEFAULT_CONFIG
    effective_state = state if state is not None else create_chat_state(effective_config)
    runner = chat_runner if chat_runner is not None else run_research_chat

    started = time.monotonic()
    turn = run_chat_turn_with_captured_logs(
        runner,
        user_input=user_idea,
        config=effective_config,
        use_llm=use_llm,
        api_key=api_key,
        model=model,
        state=effective_state,
    )
    duration_sec = time.monotonic() - started

    base_result: dict[str, Any] = {
        "user_idea": user_idea,
        "duration_sec": duration_sec,
    }

    if not turn.get("ok"):
        return {
            **base_result,
            "task_type": TASK_TYPE_ERROR,
            "decision": None,
            "tool_name": None,
            "tool_result": None,
            "report_markdown": "",
            "report_path": None,
            "plot_paths": [],
            "reply": None,
            "state": effective_state,
            "error": turn.get("error") or "chat runner failed without raising",
        }

    chat_result = turn.get("result") or {}
    tool_name = chat_result.get("tool_name")
    tool_result = chat_result.get("tool_result") or {}
    task_type = _classify_tool_name(tool_name)

    if task_type in (TASK_TYPE_DIRECT, TASK_TYPE_ERROR):
        # Persist research-plan-style replies so outputs/ always has at least one
        # markdown file — pure chitchat ("你好") is skipped.
        reply_text = chat_result.get("reply") or ""
        report_path = None
        report_md = ""
        if task_type == TASK_TYPE_DIRECT and reply_text.lstrip().startswith("# AutoETF Research Plan"):
            try:
                resolved_dir = _resolve_outputs_dir(outputs_dir)
                # Use a higher-resolution timestamp to avoid collisions when
                # multiple failures are emitted in the same second.
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"unmet_research_plan_{ts}.md"
                target = resolved_dir / filename
                target.write_text(reply_text, encoding="utf-8")
                report_path = str(target.resolve())
                report_md = reply_text
            except Exception:
                # Persisting the plan is best-effort; never fail the run for it.
                report_path = None
                report_md = ""
        return {
            **base_result,
            "task_type": task_type,
            "decision": chat_result.get("decision"),
            "tool_name": tool_name,
            "tool_result": tool_result or None,
            "report_markdown": report_md,
            "report_path": report_path,
            "plot_paths": [],
            "reply": reply_text,
            "state": chat_result.get("state"),
            "error": None if task_type == TASK_TYPE_DIRECT else f"unknown tool_name: {tool_name}",
        }

    report_markdown = tool_result.get("report", "") if isinstance(tool_result, dict) else ""
    if not report_markdown:
        return {
            **base_result,
            "task_type": TASK_TYPE_ERROR,
            "decision": chat_result.get("decision"),
            "tool_name": tool_name,
            "tool_result": tool_result or None,
            "report_markdown": "",
            "report_path": None,
            "plot_paths": [],
            "reply": chat_result.get("reply"),
            "state": chat_result.get("state"),
            "error": "pipeline did not produce a markdown report",
        }

    resolved_dir = _resolve_outputs_dir(outputs_dir)
    ts = _now_timestamp()
    artifacts = _build_report_artifacts(tool_result, task_type, outputs_dir, ts=ts)
    plot_paths = artifacts["plot_paths"]
    report_markdown = artifacts["report_markdown"]
    report_path = artifacts["report_path"]


    return {
        **base_result,
        "task_type": task_type,
        "decision": chat_result.get("decision"),
        "tool_name": tool_name,
        "tool_result": tool_result,
        "report_markdown": report_markdown,
        "report_path": report_path,
        "plot_paths": plot_paths,
        "reply": chat_result.get("reply"),
        "state": chat_result.get("state"),
        "error": None,
    }