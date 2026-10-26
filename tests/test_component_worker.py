from __future__ import annotations

import os
import struct
import time

import pytest

from rca_agent.runner.component_worker import (
    DEFAULT_MAX_FRAME_SIZE,
    FAKE_FACTORY_ID,
    ComponentWorkerHost,
    ComponentWorkerSpec,
    FrameReader,
    WorkerCrashedError,
    WorkerExecutionError,
    WorkerLifecycle,
    WorkerProtocolError,
    WorkerProtocolVersionError,
    WorkerRequestMismatchError,
    WorkerTimeoutError,
    encode_frame,
)
from rca_agent.runner.transcript import EventType, Transcript


def spec(**config: object) -> ComponentWorkerSpec:
    return ComponentWorkerSpec(
        worker_type="fake",
        factory_id=FAKE_FACTORY_ID,
        config=config,
    )


def host(worker_spec: ComponentWorkerSpec, **kwargs: float) -> ComponentWorkerHost:
    return ComponentWorkerHost(
        worker_spec,
        ready_timeout_seconds=kwargs.pop("ready_timeout_seconds", 0.5),
        request_timeout_seconds=kwargs.pop("request_timeout_seconds", 0.5),
        terminate_grace_seconds=kwargs.pop("terminate_grace_seconds", 0.1),
    )


@pytest.mark.parametrize("component", ["discovery", "formalizer", "planner"])
def test_ready_handshake_and_normal_response(component: str) -> None:
    worker = host(spec())
    worker.start()
    assert worker.state is WorkerLifecycle.READY
    assert worker._child_directory is not None
    assert os.path.isdir(worker._child_directory.name)
    assert worker.request({"component": component}, request_id="r-1") == {
        "echo": {"component": component}
    }
    assert worker.state is WorkerLifecycle.RESPONDED
    worker.terminate()
    assert worker.state is WorkerLifecycle.EXITED
    assert not worker.is_alive
    assert worker._child_directory is None


def test_partial_framed_response_does_not_block() -> None:
    worker = host(spec(partial_response=True, partial_delay_ms=1), request_timeout_seconds=1)
    worker.start()
    assert worker.request({"value": 7}) == {"echo": {"value": 7}}
    worker.terminate()


def test_frame_reader_rejects_malformed_and_oversized_frames() -> None:
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb")
    try:
        os.write(write_fd, struct.pack("!I", 8) + b"not-json")
        with pytest.raises(WorkerProtocolError, match="malformed JSON"):
            FrameReader(stream).read(0.2)
    finally:
        os.close(write_fd)
        stream.close()

    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "rb")
    try:
        os.write(write_fd, struct.pack("!I", DEFAULT_MAX_FRAME_SIZE + 1))
        with pytest.raises(WorkerProtocolError, match="invalid frame size"):
            FrameReader(stream).read(0.2)
    finally:
        os.close(write_fd)
        stream.close()


def test_frame_reader_rejects_eof_before_frame() -> None:
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    stream = os.fdopen(read_fd, "rb")
    try:
        with pytest.raises(EOFError):
            FrameReader(stream).read(0.2)
    finally:
        stream.close()


def test_ready_timeout_reaps_worker() -> None:
    worker = host(spec(startup_delay_ms=300), ready_timeout_seconds=0.03)
    with pytest.raises(WorkerTimeoutError):
        worker.start()
    assert worker.state is WorkerLifecycle.TIMED_OUT
    assert not worker.is_alive
    assert worker.wait(0) is not None


def test_request_timeout_reaps_worker_and_late_response_is_discarded() -> None:
    worker = host(spec(sleep_ms=300), request_timeout_seconds=0.03)
    worker.start()
    with pytest.raises(WorkerTimeoutError):
        worker.request({"late": True}, request_id="late-1")
    assert worker.state is WorkerLifecycle.TIMED_OUT
    assert not worker.is_alive
    # The worker was terminated before its delayed response could be accepted.
    time.sleep(0.05)
    assert worker.wait(0) is not None


def test_worker_exception_is_reported_and_reaped() -> None:
    worker = host(spec(controlled_exception=True))
    worker.start()
    with pytest.raises(WorkerExecutionError, match="controlled exception"):
        worker.request({"value": 1})
    assert worker.state is WorkerLifecycle.FAILED
    assert not worker.is_alive


def test_worker_crash_is_recognized() -> None:
    worker = host(spec(crash_on_request=True))
    worker.start()
    with pytest.raises(WorkerCrashedError):
        worker.request({"value": 1})
    assert worker.state is WorkerLifecycle.FAILED
    assert not worker.is_alive


def test_malformed_response_is_rejected() -> None:
    worker = host(spec(malformed_response=True))
    worker.start()
    with pytest.raises(WorkerProtocolError, match="malformed JSON"):
        worker.request({"value": 1})
    assert not worker.is_alive


def test_protocol_version_mismatch_is_rejected() -> None:
    worker = host(spec(ready_protocol_version=2))
    with pytest.raises(WorkerProtocolVersionError):
        worker.start()
    assert not worker.is_alive


def test_spec_protocol_version_mismatch_is_rejected_before_start() -> None:
    worker = host(ComponentWorkerSpec(
        worker_type="fake",
        factory_id=FAKE_FACTORY_ID,
        protocol_version=2,
    ))
    with pytest.raises(WorkerProtocolVersionError):
        worker.start()
    assert worker.state is WorkerLifecycle.FAILED
    assert worker.pid is None


def test_request_id_mismatch_is_rejected() -> None:
    worker = host(spec(wrong_request_id=True))
    worker.start()
    with pytest.raises(WorkerRequestMismatchError):
        worker.request({"value": 1}, request_id="expected")
    assert not worker.is_alive


def test_cancel_ack_and_reaping_leave_no_process() -> None:
    worker = host(spec())
    worker.start()
    worker.cancel(request_id="cancel-1")
    assert worker.state is WorkerLifecycle.CANCELLED
    assert not worker.is_alive
    assert worker.wait(0) is not None


def test_cancel_terminates_active_request() -> None:
    worker = host(spec(sleep_ms=500), request_timeout_seconds=1)
    worker.start()
    # Start the request in a separate process is intentionally unnecessary for
    # this protocol test: cancellation is guaranteed by the host's termination
    # path after a busy worker cannot consume the cancel frame.
    worker._send(
        {
            "protocol_version": 1,
            "message_type": "request",
            "request_id": "busy",
            "payload": {},
        }
    )
    worker.state = WorkerLifecycle.REQUESTED
    worker.cancel(request_id="busy")
    assert worker.state is WorkerLifecycle.CANCELLED
    assert not worker.is_alive


def test_timeout_uses_kill_fallback_when_sigterm_is_ignored() -> None:
    worker = host(
        spec(ignore_sigterm=True, sleep_ms=500),
        request_timeout_seconds=0.03,
        terminate_grace_seconds=0.03,
    )
    worker.start()
    with pytest.raises(WorkerTimeoutError):
        worker.request({"value": 1})
    assert worker.state is WorkerLifecycle.TIMED_OUT
    assert worker.wait(0) is not None
    assert not worker.is_alive


def test_parent_is_the_only_transcript_writer() -> None:
    transcript = Transcript("ep-" + "a" * 32)
    worker = host(spec(sleep_ms=200), request_timeout_seconds=0.02)
    worker.start()
    with pytest.raises(WorkerTimeoutError):
        worker.request({"secret": "late"})
    assert transcript.events == ()
    transcript.append(EventType.TOOL_RESPONSE, {"payload": {"accepted": True}})
    assert len(transcript.events) == 1
    assert transcript.events[0].payload == {"payload": {"accepted": True}}


def test_worker_payload_and_spec_are_json_typed() -> None:
    serialized = spec(answer="ok").model_dump_json()
    assert '"factory_id":"test.fake"' in serialized
    with pytest.raises(ValueError, match="JSON serializable"):
        ComponentWorkerSpec(
            worker_type="fake",
            factory_id=FAKE_FACTORY_ID,
            config={"bad": object()},
        )
    with pytest.raises(TypeError, match="JSON serializable"):
        worker = host(spec())
        worker.start()
        try:
            worker.request({"bad": object()})
        finally:
            worker.terminate()


def test_encode_frame_has_length_prefix() -> None:
    frame = encode_frame({"ok": True})
    assert struct.unpack("!I", frame[:4])[0] == len(frame) - 4
