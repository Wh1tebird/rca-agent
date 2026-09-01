from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def query_fingerprint(tool_name: str, request: BaseModel | dict[str, Any]) -> str:
    payload = request.model_dump(mode="json") if isinstance(request, BaseModel) else request
    digest = hashlib.sha256(f"{tool_name}:{canonical_json(payload)}".encode()).hexdigest()
    return f"qry-{digest[:16]}"
