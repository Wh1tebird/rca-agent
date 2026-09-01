import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def cases_root() -> Path:
    configured = os.environ.get("RCA100_CASES_ROOT")
    path = Path(configured) if configured else Path.home() / "study/AgenticOpsEval/RCA100/cases"
    if not path.is_dir():
        pytest.skip("local RCA100 observation directory is unavailable")
    return path
