#!/usr/bin/python3
from __future__ import annotations

import base64
import contextlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _decode(value: Any, objects: dict[int, object]) -> Any:
    if isinstance(value, list):
        return [_decode(item, objects) for item in value]
    if not isinstance(value, dict):
        return value

    kind = value.get("__hermesbench_type__")

    if kind == "bytes":
        return base64.b64decode(value["value"], validate=True)
    if kind == "path":
        return Path(value["value"])
    if kind == "tuple":
        return tuple(_decode(item, objects) for item in value["items"])
    if kind == "remote":
        return objects[value["id"]]

    return {key: _decode(item, objects) for key, item in value.items()}


def _encode(value: Any, objects: dict[int, object]) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {
            "__hermesbench_type__": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Path):
        return {"__hermesbench_type__": "path", "value": str(value)}
    if isinstance(value, tuple):
        return {
            "__hermesbench_type__": "tuple",
            "items": [_encode(item, objects) for item in value],
        }
    if isinstance(value, list):
        return [_encode(item, objects) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _encode(item, objects) for key, item in value.items()}

    identifier = max(objects, default=0) + 1
    objects[identifier] = value
    return {
        "__hermesbench_type__": "remote",
        "id": identifier,
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
    }


def _reply(payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
    os.write(1, data.encode("utf-8"))


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: candidate_rpc.py MODULE [MODULE ...]")

    modules: dict[str, object] = {}
    objects: dict[int, object] = {}
    candidate = os.environ.get("HERMES_BENCH_CANDIDATE")

    if not candidate:
        raise SystemExit("candidate path is unavailable")

    sys.path.insert(0, candidate)

    with contextlib.redirect_stdout(sys.stderr):
        for name in sys.argv[1:]:
            modules[name] = importlib.import_module(name)

    for raw in sys.stdin.buffer:
        request: Any = None
        try:
            request = json.loads(raw)
            request_id = request["id"]
            operation = request["operation"]
            module = modules[request["module"]]
            name = request["name"]

            with contextlib.redirect_stdout(sys.stderr):
                target = getattr(module, name)

                if operation == "get":
                    value = target
                elif operation == "call":
                    args = _decode(request.get("args", []), objects)
                    kwargs = _decode(request.get("kwargs", {}), objects)
                    value = target(*args, **kwargs)
                else:
                    raise ValueError(f"unsupported RPC operation: {operation}")

            _reply({"id": request_id, "ok": True, "value": _encode(value, objects)})
        except BaseException as exc:
            _reply(
                {
                    "id": request.get("id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": {
                        "module": type(exc).__module__,
                        "name": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
