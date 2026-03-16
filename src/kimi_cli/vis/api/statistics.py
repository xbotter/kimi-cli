"""Vis API for aggregate statistics across all sessions."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from kimi_cli.share import get_share_dir
from kimi_cli.vis.api.sessions import collect_events, get_work_dir_for_hash
from kimi_cli.wire.file import WireFileMetadata, parse_wire_file_line

router = APIRouter(prefix="/api/vis", tags=["vis"])

# Cache configuration
_CACHE_TTL = 120  # 2 minutes
_last_result: dict[str, Any] | None = None
_last_update: float = 0
_cache_lock = threading.Lock()
_background_task: threading.Thread | None = None


def _ensure_background_refresh():
    """Start background refresh thread if not running."""
    global _background_task
    with _cache_lock:
        if _background_task is None or not _background_task.is_alive():
            _background_task = threading.Thread(target=_background_refresh_loop, daemon=True)
            _background_task.start()


def _background_refresh_loop():
    """Background thread that periodically refreshes statistics."""
    global _last_result, _last_update
    while True:
        try:
            result = _compute_statistics()
            with _cache_lock:
                _last_result = result
                _last_update = time.time()
        except Exception:
            pass
        time.sleep(_CACHE_TTL)


def _process_session_file(args: tuple[str, str]) -> dict[str, Any] | None:
    """Process a single session file and return aggregated stats."""
    wire_path_str, work_dir = args
    wire_path = Path(wire_path_str)
    
    if not wire_path.exists():
        return None

    session_turns = 0
    session_input_tokens = 0
    session_output_tokens = 0
    first_ts = 0.0
    last_ts = 0.0
    session_date: str | None = None
    pending_tools: dict[str, str] = {}
    tool_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "error_count": 0})

    try:
        with wire_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = parse_wire_file_line(line)
                except Exception:
                    continue
                if isinstance(parsed, WireFileMetadata):
                    continue

                ts = parsed.timestamp
                msg_type = parsed.message.type
                payload = parsed.message.payload

                if first_ts == 0:
                    first_ts = ts
                    try:
                        dt = datetime.fromtimestamp(ts, tz=UTC)
                        session_date = dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                last_ts = ts

                events_to_process: list[tuple[str, dict[str, Any]]] = []
                collect_events(msg_type, payload, events_to_process)

                for ev_type, ev_payload in events_to_process:
                    if ev_type == "TurnBegin":
                        session_turns += 1
                    elif ev_type == "ToolCall":
                        fn: dict[str, Any] | None = ev_payload.get("function")
                        tool_id: str = ev_payload.get("id", "")
                        if isinstance(fn, dict):
                            name: str = fn.get("name", "unknown")
                            tool_stats[name]["count"] += 1
                            if tool_id:
                                pending_tools[tool_id] = name
                    elif ev_type == "ToolResult":
                        tool_call_id: str = ev_payload.get("tool_call_id", "")
                        rv: dict[str, Any] | None = ev_payload.get("return_value")
                        if isinstance(rv, dict) and rv.get("is_error"):
                            tool_name = pending_tools.get(tool_call_id)
                            if tool_name:
                                tool_stats[tool_name]["error_count"] += 1
                        pending_tools.pop(tool_call_id, None)
                    elif ev_type == "StatusUpdate":
                        tu: dict[str, Any] | None = ev_payload.get("token_usage")
                        if isinstance(tu, dict):
                            session_input_tokens += (
                                int(tu.get("input_other", 0))
                                + int(tu.get("input_cache_read", 0))
                                + int(tu.get("input_cache_creation", 0))
                            )
                            session_output_tokens += int(tu.get("output", 0))
    except Exception:
        return None

    duration = last_ts - first_ts if last_ts > first_ts else 0

    return {
        "work_dir": work_dir,
        "date": session_date,
        "turns": session_turns,
        "input_tokens": session_input_tokens,
        "output_tokens": session_output_tokens,
        "duration": duration,
        "tool_stats": dict(tool_stats),
    }


def _compute_statistics() -> dict[str, Any]:
    """Compute statistics by processing all session files in parallel."""
    sessions_root = get_share_dir() / "sessions"
    if not sessions_root.exists():
        # Build empty daily_usage with 30 days
        today = datetime.now(tz=UTC)
        daily_usage: list[dict[str, Any]] = []
        for i in range(29, -1, -1):
            d = today - timedelta(days=i)
            daily_usage.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "sessions": 0,
                    "turns": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )
        return {
            "total_sessions": 0,
            "total_turns": 0,
            "total_tokens": {"input": 0, "output": 0},
            "total_duration_sec": 0,
            "tool_usage": [],
            "daily_usage": daily_usage,
            "per_project": [],
        }

    # Collect all session files
    file_args: list[tuple[str, str]] = []
    for work_dir_hash_dir in sessions_root.iterdir():
        if not work_dir_hash_dir.is_dir():
            continue
        work_dir = get_work_dir_for_hash(work_dir_hash_dir.name) or work_dir_hash_dir.name
        for session_dir in work_dir_hash_dir.iterdir():
            if not session_dir.is_dir():
                continue
            wire_path = session_dir / "wire.jsonl"
            if wire_path.exists():
                file_args.append((str(wire_path), work_dir))

    # Process files in parallel using thread pool
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_process_session_file, file_args))

    # Aggregate results
    total_sessions = 0
    total_turns = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_duration_sec = 0.0

    tool_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "error_count": 0})
    daily_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sessions": 0, "turns": 0, "input_tokens": 0, "output_tokens": 0}
    )
    project_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sessions": 0, "turns": 0, "input_tokens": 0, "output_tokens": 0}
    )

    for result in results:
        if isinstance(result, Exception) or result is None:
            continue

        total_sessions += 1
        total_turns += result["turns"]
        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_duration_sec += result["duration"]

        for name, stats in result["tool_stats"].items():
            tool_stats[name]["count"] += stats["count"]
            tool_stats[name]["error_count"] += stats["error_count"]

        session_date = result["date"]
        if session_date:
            daily_stats[session_date]["sessions"] += 1
            daily_stats[session_date]["turns"] += result["turns"]
            daily_stats[session_date]["input_tokens"] += result["input_tokens"]
            daily_stats[session_date]["output_tokens"] += result["output_tokens"]

        work_dir = result["work_dir"]
        project_stats[work_dir]["sessions"] += 1
        project_stats[work_dir]["turns"] += result["turns"]
        project_stats[work_dir]["input_tokens"] += result["input_tokens"]
        project_stats[work_dir]["output_tokens"] += result["output_tokens"]

    # Build final result
    tool_usage = sorted(
        [
            {"name": name, "count": stats["count"], "error_count": stats["error_count"]}
            for name, stats in tool_stats.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )[:20]

    today = datetime.now(tz=UTC)
    daily_usage: list[dict[str, Any]] = []
    for i in range(29, -1, -1):
        d = today - timedelta(days=i)
        date_str = d.strftime("%Y-%m-%d")
        entry = daily_stats.get(date_str, {"sessions": 0, "turns": 0, "input_tokens": 0, "output_tokens": 0})
        daily_usage.append(
            {
                "date": date_str,
                "sessions": entry["sessions"],
                "turns": entry["turns"],
                "input_tokens": entry["input_tokens"],
                "output_tokens": entry["output_tokens"],
            }
        )

    per_project = sorted(
        [
            {
                "work_dir": wd,
                "sessions": stats["sessions"],
                "turns": stats["turns"],
                "input_tokens": stats["input_tokens"],
                "output_tokens": stats["output_tokens"],
            }
            for wd, stats in project_stats.items()
        ],
        key=lambda x: x["turns"],
        reverse=True,
    )[:10]

    return {
        "total_sessions": total_sessions,
        "total_turns": total_turns,
        "total_tokens": {"input": total_input_tokens, "output": total_output_tokens},
        "total_duration_sec": total_duration_sec,
        "tool_usage": tool_usage,
        "daily_usage": daily_usage,
        "per_project": per_project,
    }


@router.get("/statistics")
def get_statistics() -> dict[str, Any]:
    """Aggregate statistics across all sessions.
    
    Uses background pre-computation with 2-minute cache.
    First request may be slow (~3s), subsequent requests are fast (<50ms).
    """
    global _last_result, _last_update
    
    now = time.time()
    
    # Ensure background refresh is running
    _ensure_background_refresh()
    
    with _cache_lock:
        # Return cached result if available and fresh
        if _last_result is not None and (now - _last_update) < _CACHE_TTL:
            return _last_result
    
    # Compute synchronously if no cache available
    result = _compute_statistics()
    
    with _cache_lock:
        _last_result = result
        _last_update = now
    
    return result
