"""Serialization-safe receipt writer (ops lane, additive, adopt-if-you-like).

Two receipts aborted on 2026-09-01 22:37-22:46 purely at the JSON writer
(rung 443: numpy Boolean predicate scalars; rung 444: a CheckpointReceipt
object) after all computation had finished.  `dump(result, path)` ends
that class: it recursively sanitizes numpy scalars/arrays, torch tensors,
dataclasses, Paths, and sets, then writes atomically (tmp + rename) so a
crash can never leave a truncated receipt.  Semantics of values are never
changed -- only their JSON encoding.
"""
import dataclasses
import json
import os
from pathlib import Path


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sanitize(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()  # numpy/torch 0-d scalars
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()  # numpy arrays / torch tensors
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)  # last resort: never crash the writer


def dump(result: dict, path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(sanitize(result), indent=1) + "\n")
    os.replace(tmp, path)
