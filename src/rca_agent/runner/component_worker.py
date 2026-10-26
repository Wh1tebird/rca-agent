"""A small, typed, cancellable worker protocol for future component adapters.

This module is intentionally independent from :mod:`episode`.  It is a Phase 0
building block: the parent process sends JSON only, and the worker can be
terminated and reaped when a deadline expires.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import struct
import subprocess
import sys
import tempfile
import time
from enum import StrEnum
from typing import Any, BinaryIO

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL_VERSION = 1
DEFAULT_MAX_FRAME_SIZE = 64 * 1024
FAKE_FACTORY_ID = "test.fake"
_HEADER = struct.Struct("!I")


class ComponentWorkerSpec(BaseModel):
    """Serializable description of a worker, with no live Python objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_type: str = Field(min_length=1, max_length=80)
    factory_id: str = Field(min_length=1, max_length=120)
    config: dict[str, Any] = Field(default_factory=dict)
    protocol_version: int = Field(default=PROTOCOL_VERSION, ge=1)

    @field_validator("config")
    @classmethod
    def _json_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("worker config must be JSON serializable") from exc
        return value


class WorkerMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: int = Field(ge=1)
    message_type: str = Field(min_length=1, max_length=40)
    request_id: str | None = Field(default=None, max_length=120)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _json_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("worker payload must be JSON serializable") from exc
        return value


class WorkerLifecycle(StrEnum):
    STARTING = "starting"
    READY = "ready"
    REQUESTED = "requested"
    RESPONDED = "responded"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    TERMINATING = "terminating"
    EXITED = "exited"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class WorkerProtocolError(RuntimeError):
    """The worker sent a malformed or unexpected protocol message."""


class WorkerProtocolVersionError(WorkerProtocolError):
    pass


class WorkerRequestMismatchError(WorkerProtocolError):
    pass


class WorkerTimeoutError(TimeoutError):
    pass


class WorkerCrashedError(RuntimeError):
    pass


class WorkerExecutionError(RuntimeError):
    pass


class FrameReader:
    """Read bounded length-prefixed JSON frames, including partial reads."""

    def __init__(self, stream: BinaryIO, *, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> None:
        if max_frame_size <= 0:
            raise ValueError("max_frame_size must be positive")
        self._stream = stream
        self._max_frame_size = max_frame_size
        self._buffer = bytearray()

    def read(self, timeout_seconds: float) -> dict[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        header = self._read_exact(_HEADER.size, deadline)
        (size,) = _HEADER.unpack(header)
        if size == 0 or size > self._max_frame_size:
            raise WorkerProtocolError(f"invalid frame size: {size}")
        body = self._read_exact(size, deadline)
        try:
            decoded = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerProtocolError("malformed JSON frame") from exc
        if not isinstance(decoded, dict):
            raise WorkerProtocolError("worker frame must contain a JSON object")
        return decoded

    def _read_exact(self, size: int, deadline: float) -> bytes:
        while len(self._buffer) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeoutError("worker frame timed out")
            selector = selectors.DefaultSelector()
            try:
                selector.register(self._stream.fileno(), selectors.EVENT_READ)
                if not selector.select(remaining):
                    raise WorkerTimeoutError("worker frame timed out")
            finally:
                selector.close()
            chunk = os.read(self._stream.fileno(), max(4096, size - len(self._buffer)))
            if not chunk:
                raise EOFError("worker closed its output")
            self._buffer.extend(chunk)
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result


def encode_frame(message: dict[str, Any], *, max_frame_size: int = DEFAULT_MAX_FRAME_SIZE) -> bytes:
    """Encode one bounded JSON frame without accepting arbitrary objects."""

    try:
        body = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError("frame payload must be JSON serializable") from exc
    if not body or len(body) > max_frame_size:
        raise WorkerProtocolError(f"frame exceeds maximum size: {len(body)}")
    return _HEADER.pack(len(body)) + body


def write_frame(
    stream: BinaryIO,
    message: dict[str, Any],
    *,
    max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
    chunk_size: int | None = None,
    chunk_delay_seconds: float = 0,
) -> None:
    data = encode_frame(message, max_frame_size=max_frame_size)
    offset = 0
    while offset < len(data):
        step = len(data) - offset if chunk_size is None else min(chunk_size, len(data) - offset)
        written = os.write(stream.fileno(), data[offset : offset + step])
        if written <= 0:
            raise BrokenPipeError("worker output closed")
        offset += written
        if chunk_delay_seconds and offset < len(data):
            time.sleep(chunk_delay_seconds)


def _message(
    message_type: str,
    *,
    request_id: str | None = None,
    payload: dict[str, Any] | None = None,
    protocol_version: int = PROTOCOL_VERSION,
) -> dict[str, Any]:
    return WorkerMessage(
        protocol_version=protocol_version,
        message_type=message_type,
        request_id=request_id,
        payload=payload or {},
    ).model_dump(mode="json")


class ComponentWorkerHost:
    """Owns one worker process and always reaps it on timeout or cancellation."""

    def __init__(
        self,
        spec: ComponentWorkerSpec,
        *,
        ready_timeout_seconds: float = 1.0,
        request_timeout_seconds: float = 1.0,
        terminate_grace_seconds: float = 0.2,
        max_frame_size: int = DEFAULT_MAX_FRAME_SIZE,
    ) -> None:
        if ready_timeout_seconds <= 0 or request_timeout_seconds <= 0:
            raise ValueError("worker deadlines must be positive")
        if terminate_grace_seconds <= 0:
            raise ValueError("terminate grace must be positive")
        self.spec = spec
        self.ready_timeout_seconds = ready_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.terminate_grace_seconds = terminate_grace_seconds
        self.max_frame_size = max_frame_size
        self.state = WorkerLifecycle.STARTING
        self._process: subprocess.Popen[bytes] | None = None
        self._reader: FrameReader | None = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("worker already started")
        self.state = WorkerLifecycle.STARTING
        if self.spec.protocol_version != PROTOCOL_VERSION:
            self.state = WorkerLifecycle.FAILED
            raise WorkerProtocolVersionError("worker protocol version mismatch")
        with tempfile.TemporaryDirectory(prefix="rca-component-") as child_directory:
            self._process = subprocess.Popen(
                [sys.executable, "-I", "-m", "rca_agent.runner.component_worker"],
                cwd=child_directory,
                env=_sanitized_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._reader = FrameReader(self._process.stdout, max_frame_size=self.max_frame_size)
        try:
            self._send(
                _message(
                    "start",
                    payload={"spec": self.spec.model_dump(mode="json")},
                )
            )
            ready = self._read(self.ready_timeout_seconds)
            self._expect_version(ready)
            if ready.get("message_type") != "ready":
                raise WorkerProtocolError("worker did not send ready")
            self.state = WorkerLifecycle.READY
        except WorkerTimeoutError:
            self.state = WorkerLifecycle.TIMED_OUT
            self._terminate_and_reap()
            raise
        except WorkerProtocolVersionError:
            self.state = WorkerLifecycle.FAILED
            self._terminate_and_reap()
            raise
        except WorkerProtocolError:
            self.state = WorkerLifecycle.FAILED
            self._terminate_and_reap()
            raise
        except (EOFError, BrokenPipeError) as exc:
            self.state = WorkerLifecycle.FAILED
            self._terminate_and_reap()
            raise WorkerCrashedError("worker failed during startup") from exc

    def request(self, payload: dict[str, Any], *, request_id: str = "request-1") -> dict[str, Any]:
        if self.state not in {WorkerLifecycle.READY, WorkerLifecycle.RESPONDED}:
            raise RuntimeError(f"worker is not ready: {self.state}")
        _ensure_json_object(payload)
        self.state = WorkerLifecycle.REQUESTED
        try:
            self._send(_message("request", request_id=request_id, payload=payload))
            response = self._read(self.request_timeout_seconds)
            self._expect_version(response)
            if response.get("request_id") != request_id:
                raise WorkerRequestMismatchError("worker response request id mismatch")
            message_type = response.get("message_type")
            if message_type == "error":
                self.state = WorkerLifecycle.FAILED
                self._terminate_and_reap()
                raise WorkerExecutionError(response.get("payload", {}).get("message", "worker error"))
            if message_type != "response":
                raise WorkerProtocolError("worker sent an unsupported response")
            self.state = WorkerLifecycle.RESPONDED
            return response["payload"]
        except WorkerTimeoutError:
            self.state = WorkerLifecycle.TIMED_OUT
            self._terminate_and_reap()
            raise
        except (EOFError, BrokenPipeError) as exc:
            self.state = WorkerLifecycle.FAILED
            self._terminate_and_reap()
            raise WorkerCrashedError("worker exited before response") from exc
        except WorkerProtocolError:
            self.state = WorkerLifecycle.FAILED
            self._terminate_and_reap()
            raise

    def cancel(self, *, request_id: str = "request-1") -> None:
        if self._process is None or self._process.poll() is not None:
            self.state = WorkerLifecycle.CANCELLED
            self._close_streams()
            return
        self.state = WorkerLifecycle.CANCELLING
        try:
            try:
                self._send(_message("cancel", request_id=request_id))
                response = self._read(min(self.request_timeout_seconds, self.terminate_grace_seconds))
                self._expect_version(response)
                if response.get("message_type") != "cancel_ack":
                    raise WorkerProtocolError("worker did not acknowledge cancellation")
            except (WorkerTimeoutError, EOFError, WorkerProtocolError, BrokenPipeError):
                pass
        finally:
            self._terminate_and_reap()
            self.state = WorkerLifecycle.CANCELLED

    def terminate(self) -> None:
        self.state = WorkerLifecycle.TERMINATING
        process = self._process
        if process is not None and process.poll() is None:
            try:
                self._send(_message("exit"))
                exited = process.wait(timeout=self.terminate_grace_seconds)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                exited = None
            if exited is None:
                self._terminate_and_reap()
            else:
                self._close_streams()
        else:
            self._close_streams()
        self.state = WorkerLifecycle.EXITED

    def wait(self, timeout_seconds: float | None = None) -> int | None:
        if self._process is None:
            return None
        try:
            return self._process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return None

    def _read(self, timeout_seconds: float) -> dict[str, Any]:
        if self._reader is None:
            raise RuntimeError("worker has not started")
        raw = self._reader.read(timeout_seconds)
        try:
            return WorkerMessage.model_validate(raw).model_dump(mode="json")
        except ValueError as exc:
            raise WorkerProtocolError("worker message is malformed") from exc

    def _expect_version(self, message: dict[str, Any]) -> None:
        if message.get("protocol_version") != PROTOCOL_VERSION:
            raise WorkerProtocolVersionError("worker protocol version mismatch")

    def _send(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("worker has not started")
        write_frame(self._process.stdin, message, max_frame_size=self.max_frame_size)

    def _terminate_and_reap(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.terminate_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=self.terminate_grace_seconds)
        self._close_streams()

    def _close_streams(self) -> None:
        process = self._process
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def _ensure_json_object(value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise TypeError("worker payload must be a JSON object")
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("worker payload must be JSON serializable") from exc


def _sanitized_environment() -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "TMPDIR", "PYTHONPATH"}
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _worker_main() -> int:
    reader = FrameReader(sys.stdin.buffer)
    output = sys.stdout.buffer
    try:
        start = WorkerMessage.model_validate(reader.read(30.0))
        if start.protocol_version != PROTOCOL_VERSION or start.message_type != "start":
            return 2
        spec = ComponentWorkerSpec.model_validate(start.payload.get("spec"))
        if spec.worker_type != "fake" or spec.factory_id != FAKE_FACTORY_ID:
            return 6
        config = spec.config
        if config.get("ignore_sigterm"):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        startup_delay = float(config.get("startup_delay_ms", 0)) / 1000
        if startup_delay:
            time.sleep(startup_delay)
        ready_version = int(config.get("ready_protocol_version", PROTOCOL_VERSION))
        write_frame(
            output,
            _message(
                "ready",
                payload={"worker_type": spec.worker_type, "factory_id": spec.factory_id},
                protocol_version=ready_version,
            ),
        )
        while True:
            message = WorkerMessage.model_validate(reader.read(30.0))
            if message.protocol_version != PROTOCOL_VERSION:
                return 3
            if message.message_type == "cancel":
                write_frame(output, _message("cancel_ack", request_id=message.request_id))
                continue
            if message.message_type == "exit":
                write_frame(output, _message("exit", payload={"status": "exited"}))
                return 0
            if message.message_type != "request" or message.request_id is None:
                return 4
            if config.get("crash_on_request"):
                os._exit(17)
            sleep_ms = float(config.get("sleep_ms", 0))
            if sleep_ms:
                time.sleep(sleep_ms / 1000)
            if config.get("controlled_exception"):
                write_frame(
                    output,
                    _message(
                        "error",
                        request_id=message.request_id,
                        payload={"error_type": "RuntimeError", "message": "controlled exception"},
                    ),
                )
                continue
            if config.get("malformed_response"):
                _write_raw_frame(output, b"not-json")
                continue
            if config.get("oversized_response"):
                os.write(output.fileno(), _HEADER.pack(DEFAULT_MAX_FRAME_SIZE + 1))
                continue
            response_id = "wrong-request-id" if config.get("wrong_request_id") else message.request_id
            write_frame(
                output,
                _message("response", request_id=response_id, payload={"echo": message.payload}),
                chunk_size=1 if config.get("partial_response") else None,
                chunk_delay_seconds=float(config.get("partial_delay_ms", 0)) / 1000,
            )
            if config.get("exit_after_response"):
                return 0
    except (EOFError, BrokenPipeError, WorkerProtocolError, ValueError, TypeError):
        return 5


def _write_raw_frame(stream: BinaryIO, body: bytes) -> None:
    os.write(stream.fileno(), _HEADER.pack(len(body)) + body)


if __name__ == "__main__":  # pragma: no cover - exercised in subprocesses
    raise SystemExit(_worker_main())
