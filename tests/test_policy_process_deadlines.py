"""Ready/ack separates bounded startup from task time; actual workers are reaped."""
import os
import subprocess
import time
from types import SimpleNamespace

import pytest
from test_policy_process import _runner

from rca_agent.models import RequestStopAction, StopReason
from rca_agent.runner import (
    RecordedReplay,
    ScriptedPolicyProcessHost,
    ScriptedPolicySpec,
    policy_process,
)
from rca_agent.runner.transcript import EventType


@pytest.mark.parametrize("behavior", ["complete", "crash", "hang", "startup_timeout"])
def test_actual_worker_deadlines_and_reaping(cases_root, monkeypatch, behavior):
    children, waits = [], []
    original_popen = subprocess.Popen
    original_read = policy_process._read_message

    def capture(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        children.append(process)
        return process

    def read(process, timeout):
        started = time.monotonic()
        kind = None
        try:
            result = original_read(process, timeout)
            kind = result["kind"]
            return result
        finally:
            waits.append((timeout, time.monotonic() - started, kind))

    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(policy_process, "_read_message", read)
    spec = {"actions": (RequestStopAction(summary="bounded worker lifecycle").model_dump(
        mode="json"),), "startup_delay_seconds": .4}
    if behavior in {"crash", "hang"}:
        spec["crash_after_actions" if behavior == "crash" else "hang_after_actions"] = 0
    runner = _runner(cases_root)
    started = time.monotonic()
    report = ScriptedPolicyProcessHost().run(runner, ScriptedPolicySpec(**spec),
        timeout_seconds=.1, startup_timeout_seconds=.1 if behavior == "startup_timeout" else 5)
    elapsed = time.monotonic() - started
    assert elapsed < 7  # Startup + task + bounded cleanup, independent of a .3s import guess.
    assert len(children) == 1 and children[0].poll() is not None
    assert children[0].stdin.closed and children[0].stdout.closed and children[0].stderr.closed
    assert not any(e.event_type == EventType.TOOL_REQUEST for e in runner.transcript.events)
    assert RecordedReplay.verify(runner.transcript.events).valid
    expected = {"complete": "completed", "crash": "crash", "hang": "timeout",
                "startup_timeout": "timeout"}[behavior]
    assert report.exit_kind == expected
    print({"behavior": behavior, "waits": waits, "elapsed": elapsed,
           "returncode": children[0].returncode, "exit_kind": report.exit_kind})
    if behavior == "startup_timeout":
        assert len(waits) == 1 and waits[0][0] == .1 and waits[0][2] is None
        assert report.visible_context == {} and report.actions_received == 0
    else:
        assert waits[0][0] == 5 and waits[0][1] >= .4 and waits[0][2] == "ready"
        assert all(w[0] == .1 for w in waits[1:])
        if behavior == "hang":
            assert .1 <= waits[1][1] < 2
            assert report.answer.termination_reason == StopReason.POLICY_TIMEOUT
        elif behavior == "crash":
            assert report.answer.termination_reason == StopReason.POLICY_CRASH
        else:
            assert report.actions_received == 1


def test_cancel_during_worker_execution_reaps_process(cases_root, monkeypatch):
    children = []
    original_popen = subprocess.Popen
    original_read = policy_process._read_message

    def capture(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    def read(child, timeout):
        if timeout == .1:
            raise KeyboardInterrupt("synthetic host cancellation after ready")
        return original_read(child, timeout)

    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(policy_process, "_read_message", read)
    runner = _runner(cases_root)
    with pytest.raises(KeyboardInterrupt):
        ScriptedPolicyProcessHost().run(runner, ScriptedPolicySpec(actions=(),
            startup_delay_seconds=.1), timeout_seconds=.1)
    assert len(children) == 1 and children[0].poll() is not None
    assert all(stream.closed for stream in (children[0].stdin, children[0].stdout, children[0].stderr))
    assert not any(e.event_type in {EventType.TOOL_REQUEST, EventType.QUERY_COST}
                   for e in runner.transcript.events)


def test_partial_worker_frame_cannot_escape_response_deadline():
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "r")
    process = SimpleNamespace(stdout=stream)
    try:
        os.write(write_fd, b'{"kind":')
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            policy_process._read_message(process, .05)
        assert .05 <= time.monotonic() - started < 2
    finally:
        os.close(write_fd)
        stream.close()
