from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rca_agent.models import AnswerArtifact, EpisodeStatus, StopReason, parse_policy_action
from rca_agent.runner.episode import ActionRejectedError, EpisodeRunner


class ScriptedPolicySpec(BaseModel):
    """Host test fixture for phase 2 process isolation, not an investigation strategy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions: tuple[dict[str, Any], ...]
    crash_after_actions: int | None = Field(default=None, ge=0)
    hang_after_actions: int | None = Field(default=None, ge=0)
    startup_delay_seconds: float = Field(default=0, ge=0, le=2)


class PolicyProcessReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exit_kind: str
    visible_context: dict[str, Any]
    actions_received: int = Field(ge=0)
    answer: AnswerArtifact


class ScriptedPolicyProcessHost:
    """Runs only a scripted/fake policy in a sanitized local child process."""

    def run(
        self,
        runner: EpisodeRunner,
        policy: ScriptedPolicySpec,
        *,
        timeout_seconds: float = 5.0,
        startup_timeout_seconds: float = 5.0,
    ) -> PolicyProcessReport:
        if timeout_seconds <= 0 or startup_timeout_seconds <= 0:
            raise ValueError("policy deadlines must be positive")
        if runner.status == EpisodeStatus.CREATED:
            bootstrap = runner.start()
        elif runner.status == EpisodeStatus.ACTIVE and runner.bootstrap is not None:
            bootstrap = runner.bootstrap
        else:
            raise RuntimeError("policy process requires a created or active episode")
        initialization = {
            "bootstrap": bootstrap.model_dump(mode="json"),
            "actions": list(policy.actions),
            "crash_after_actions": policy.crash_after_actions,
            "hang_after_actions": policy.hang_after_actions,
            "startup_delay_seconds": policy.startup_delay_seconds,
        }
        visible_context: dict[str, Any] = {}
        actions_received = 0
        exit_kind = "completed"
        with tempfile.TemporaryDirectory(prefix="rca-policy-") as child_directory:
            process = subprocess.Popen(
                [sys.executable, "-I", "-m", "rca_agent.runner.policy_worker"],
                cwd=child_directory,
                env=_sanitized_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            assert process.stdin is not None
            try:
                process.stdin.write(json.dumps(initialization, ensure_ascii=False) + "\n")
                process.stdin.flush()
                # Execution starts only after the worker's ready/ack handshake.
                # Interpreter/import/isolation startup has a separate bounded deadline.
                ready = _read_message(process, startup_timeout_seconds)
                if ready.get("kind") != "ready":
                    raise RuntimeError("policy worker did not send ready message")
                visible_context = ready["visible_context"]
                runner.record_policy_process_started(visible_context)
                _acknowledge(process, terminal=False, result={"status": "ready"})
                while runner.status == EpisodeStatus.ACTIVE:
                    message = _read_message(process, timeout_seconds)
                    if message.get("kind") == "complete":
                        runner.finish_script()
                        break
                    if message.get("kind") != "action":
                        raise RuntimeError("policy worker sent an unsupported message")
                    actions_received += 1
                    try:
                        action = parse_policy_action(message["payload"])
                    except (ValidationError, TypeError, ValueError):
                        runner.reject_raw_action(message.get("payload"))
                        runner.terminate_policy_failure(
                            StopReason.POLICY_INVALID_ACTION,
                            "invalid_policy_action",
                        )
                        _acknowledge(process, terminal=True, result=None)
                        exit_kind = "invalid_action"
                        break
                    try:
                        result = runner.handle(action)
                    except ActionRejectedError:
                        _acknowledge(
                            process,
                            terminal=False,
                            result={"status": "action_rejected"},
                        )
                        continue
                    terminal = runner.status != EpisodeStatus.ACTIVE
                    serialized = (
                        result.model_dump(mode="json") if isinstance(result, BaseModel) else None
                    )
                    _acknowledge(process, terminal=terminal, result=serialized)
                if process.poll() is None:
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
            except TimeoutError:
                exit_kind = "timeout"
                process.kill()
                process.wait(timeout=1)
                if runner.status == EpisodeStatus.ACTIVE:
                    runner.terminate_policy_failure(StopReason.POLICY_TIMEOUT, "policy_timeout")
            except (EOFError, RuntimeError, BrokenPipeError, json.JSONDecodeError):
                exit_kind = "crash"
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=1)
                if runner.status == EpisodeStatus.ACTIVE:
                    runner.terminate_policy_failure(StopReason.POLICY_CRASH, "policy_process_exit")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=1)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
        runner.record_policy_process_exited(exit_kind)
        if runner.answer is None:
            raise RuntimeError("policy process ended without a frozen answer")
        return PolicyProcessReport(
            exit_kind=exit_kind,
            visible_context=visible_context,
            actions_received=actions_received,
            answer=runner.answer,
        )


def _sanitized_environment() -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "TMPDIR"}
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _read_message(process: subprocess.Popen[str], timeout_seconds: float) -> dict[str, Any]:
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    buffered = getattr(process, "_rca_pending_stdout", b"")
    try:
        while True:
            if b"\n" in buffered:
                line, _, tail = buffered.partition(b"\n")
                process._rca_pending_stdout = tail
                return json.loads(line)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("policy process response timed out")
            if selector.select(remaining):
                # readline() could wait indefinitely for a newline after select
                # reports a partial frame. Read available bytes under one deadline.
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    raise EOFError("policy process closed its output")
                buffered += chunk
    finally:
        selector.close()


def _acknowledge(
    process: subprocess.Popen[str], *, terminal: bool, result: dict[str, Any] | None
) -> None:
    assert process.stdin is not None
    process.stdin.write(
        json.dumps({"terminal": terminal, "result": result}, ensure_ascii=False) + "\n"
    )
    process.stdin.flush()
