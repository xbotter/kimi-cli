"""Tests for Vis statistics API."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kimi_cli.vis.app import create_app
from kimi_cli.vis.api import statistics as stats_module


@pytest.fixture(autouse=True)
def reset_cache(monkeypatch, tmp_path):
    """Reset statistics cache before each test and use tmp share dir."""
    stats_module._last_result = None
    stats_module._last_update = 0
    monkeypatch.setenv("KIMI_SHARE_DIR", str(tmp_path))
    yield


def create_wire_event(event_type: str, payload: dict, timestamp: float = 0.0) -> dict:
    """Create a wire event record."""
    return {
        "timestamp": timestamp,
        "message": {
            "type": event_type,
            "payload": payload,
        },
    }


def create_mock_session(
    session_dir: Path,
    turns: int = 1,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    tools: list[str] | None = None,
) -> None:
    """Create a mock wire.jsonl file with session events."""
    session_dir.mkdir(parents=True, exist_ok=True)
    
    events = [
        # Metadata header
        json.dumps({"type": "metadata", "protocol_version": "2.0"}),
    ]
    
    # Use current time (within last 30 days)
    ts = time.time() - 86400  # Yesterday
    
    for i in range(turns):
        # TurnBegin
        events.append(json.dumps(create_wire_event(
            "TurnBegin",
            {"user_input": f"Test message {i}"},
            timestamp=ts + i * 10,
        )))
        
        # StatusUpdate with token usage
        events.append(json.dumps(create_wire_event(
            "StatusUpdate",
            {
                "token_usage": {
                    "input_other": input_tokens // turns,
                    "input_cache_read": 0,
                    "input_cache_creation": 0,
                    "output": output_tokens // turns,
                },
                "context_usage": 0.5,
            },
            timestamp=ts + i * 10 + 1,
        )))
        
        # Tool calls
        if tools:
            for tool_name in tools:
                events.append(json.dumps(create_wire_event(
                    "ToolCall",
                    {
                        "id": f"call_{i}_{tool_name}",
                        "function": {"name": tool_name, "arguments": "{}"},
                    },
                    timestamp=ts + i * 10 + 2,
                )))
                
                events.append(json.dumps(create_wire_event(
                    "ToolResult",
                    {
                        "tool_call_id": f"call_{i}_{tool_name}",
                        "return_value": {"result": "ok"},
                    },
                    timestamp=ts + i * 10 + 3,
                )))
    
    wire_path = session_dir / "wire.jsonl"
    wire_path.write_text("\n".join(events) + "\n", encoding="utf-8")


class TestStatisticsAPI:
    """Test suite for statistics API."""
    
    def test_statistics_empty_sessions(self) -> None:
        """Test statistics with no sessions."""
        # KIMI_SHARE_DIR is set by reset_cache fixture
        
        with TestClient(create_app()) as client:
            response = client.get("/api/vis/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_sessions"] == 0
        assert data["total_turns"] == 0
        assert data["total_tokens"]["input"] == 0
        assert data["total_tokens"]["output"] == 0
        assert data["tool_usage"] == []
        assert len(data["daily_usage"]) == 30
        assert data["per_project"] == []
    
    def test_statistics_single_session(
        self,
        tmp_path: Path,
    ) -> None:
        """Test statistics with a single session."""
        # KIMI_SHARE_DIR is set by reset_cache fixture
        sessions_dir = tmp_path / "sessions"
        work_dir_hash = "abc123"
        session_dir = sessions_dir / work_dir_hash / "session1"
        
        create_mock_session(
            session_dir,
            turns=2,
            input_tokens=2000,
            output_tokens=1000,
            tools=["ReadFile", "Shell"],
        )
        
        with TestClient(create_app()) as client:
            response = client.get("/api/vis/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_sessions"] == 1
        assert data["total_turns"] == 2
        assert data["total_tokens"]["input"] == 2000
        assert data["total_tokens"]["output"] == 1000
        
        # Check tool usage
        tool_names = {t["name"] for t in data["tool_usage"]}
        assert "ReadFile" in tool_names
        assert "Shell" in tool_names
        
        # Check daily usage has 30 days
        assert len(data["daily_usage"]) == 30
        
        # At least one day should have data
        days_with_sessions = [d for d in data["daily_usage"] if d["sessions"] > 0]
        assert len(days_with_sessions) >= 1
        
        # Check the day has correct token data
        day = days_with_sessions[0]
        assert day["input_tokens"] == 2000
        assert day["output_tokens"] == 1000
    
    def test_statistics_multiple_sessions(
        self,
        tmp_path: Path,
    ) -> None:
        """Test statistics aggregation across multiple sessions."""
        # KIMI_SHARE_DIR is set by reset_cache fixture
        sessions_dir = tmp_path / "sessions"
        
        # Create two sessions
        for i, (work_dir, turns, input_t, output_t) in enumerate([
            ("abc123", 1, 1000, 500),
            ("def456", 2, 2000, 1000),
        ]):
            session_dir = sessions_dir / work_dir / f"session{i}"
            create_mock_session(
                session_dir,
                turns=turns,
                input_tokens=input_t,
                output_tokens=output_t,
            )
        
        with TestClient(create_app()) as client:
            response = client.get("/api/vis/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_sessions"] == 2
        assert data["total_turns"] == 3  # 1 + 2
        assert data["total_tokens"]["input"] == 3000  # 1000 + 2000
        assert data["total_tokens"]["output"] == 1500  # 500 + 1000
    
    def test_statistics_caching(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that statistics are cached and subsequent requests are faster."""
        # KIMI_SHARE_DIR is set by reset_cache fixture
        sessions_dir = tmp_path / "sessions"
        session_dir = sessions_dir / "abc123" / "session1"
        create_mock_session(session_dir, turns=1)
        
        with TestClient(create_app()) as client:
            # First request (cold cache)
            start = time.time()
            response1 = client.get("/api/vis/statistics")
            cold_duration = time.time() - start
            
            assert response1.status_code == 200
            
            # Second request (should be cached)
            start = time.time()
            response2 = client.get("/api/vis/statistics")
            cached_duration = time.time() - start
            
            assert response2.status_code == 200
            assert response1.json() == response2.json()
            
            # Cached request should be significantly faster
            # (allowing for some variance in test environment)
            assert cached_duration < cold_duration * 0.5
    
    def test_statistics_token_calculation(
        self,
        tmp_path: Path,
    ) -> None:
        """Test accurate token calculation including cache tokens."""
        # KIMI_SHARE_DIR is set by reset_cache fixture
        session_dir = tmp_path / "sessions" / "abc123" / "session1"
        session_dir.mkdir(parents=True)
        
        # Create event with all token types
        events = [
            json.dumps({"type": "metadata", "protocol_version": "2.0"}),
            json.dumps(create_wire_event(
                "TurnBegin",
                {"user_input": "test"},
                timestamp=time.time() - 86400,
            )),
            json.dumps(create_wire_event(
                "StatusUpdate",
                {
                    "token_usage": {
                        "input_other": 1000,
                        "input_cache_read": 500,
                        "input_cache_creation": 200,
                        "output": 300,
                    },
                },
                timestamp=time.time() - 86400 + 1,
            )),
        ]
        
        wire_path = session_dir / "wire.jsonl"
        wire_path.write_text("\n".join(events) + "\n", encoding="utf-8")
        
        with TestClient(create_app()) as client:
            response = client.get("/api/vis/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        # input = input_other + input_cache_read + input_cache_creation
        assert data["total_tokens"]["input"] == 1700  # 1000 + 500 + 200
        assert data["total_tokens"]["output"] == 300
    
    def test_statistics_tool_error_counting(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that tool errors are correctly counted."""
        # KIMI_SHARE_DIR is set by reset_cache fixture
        session_dir = tmp_path / "sessions" / "abc123" / "session1"
        session_dir.mkdir(parents=True)
        
        events = [
            json.dumps({"type": "metadata", "protocol_version": "2.0"}),
            json.dumps(create_wire_event(
                "ToolCall",
                {
                    "id": "call_1",
                    "function": {"name": "Shell", "arguments": "{}"},
                },
                timestamp=time.time() - 86400,
            )),
            json.dumps(create_wire_event(
                "ToolResult",
                {
                    "tool_call_id": "call_1",
                    "return_value": {"is_error": True, "error": "Command failed"},
                },
                timestamp=time.time() - 86400 + 1,
            )),
            json.dumps(create_wire_event(
                "ToolCall",
                {
                    "id": "call_2",
                    "function": {"name": "Shell", "arguments": "{}"},
                },
                timestamp=time.time() - 86400 + 2,
            )),
            json.dumps(create_wire_event(
                "ToolResult",
                {
                    "tool_call_id": "call_2",
                    "return_value": {"result": "ok"},
                },
                timestamp=time.time() - 86400 + 3,
            )),
        ]
        
        wire_path = session_dir / "wire.jsonl"
        wire_path.write_text("\n".join(events) + "\n", encoding="utf-8")
        
        with TestClient(create_app()) as client:
            response = client.get("/api/vis/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Find Shell tool in tool_usage
        shell_tool = next((t for t in data["tool_usage"] if t["name"] == "Shell"), None)
        assert shell_tool is not None
        assert shell_tool["count"] == 2
        assert shell_tool["error_count"] == 1
