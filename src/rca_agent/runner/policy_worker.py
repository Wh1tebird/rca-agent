from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def _send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    initialization = json.loads(sys.stdin.readline())
    time.sleep(initialization.get("startup_delay_seconds", 0))
    bootstrap = initialization["bootstrap"]
    actions = initialization["actions"]
    crash_after = initialization.get("crash_after_actions")
    hang_after = initialization.get("hang_after_actions")
    _send(
        {
            "kind": "ready",
            "visible_context": {
                "bootstrap": bootstrap,
                "environment_keys": sorted(os.environ),
                "working_directory": os.getcwd(),
            },
        }
    )
    json.loads(sys.stdin.readline())
    for index, action in enumerate(actions):
        if crash_after == index:
            raise RuntimeError("scripted policy crash")
        if hang_after == index:
            time.sleep(3600)
        _send({"kind": "action", "payload": action})
        acknowledgement = json.loads(sys.stdin.readline())
        if acknowledgement.get("terminal"):
            break
    _send({"kind": "complete"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
