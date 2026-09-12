#!/usr/bin/env python3
"""Write a rung receipt so a full disk cannot destroy the GPU work that produced it.

WHY THIS EXISTS. On 2026-09-12 at 06:52 run_unit_verbprep_registered_objective_v647 completed its fit and died on
the final line: OSError [Errno 28] No space left on device, writing its receipt into circuits/followups/. About four
minutes of GPU was spent and the result was lost entirely, because the receipt is written once, at the end, with
pathlib.write_text. The same disk took out dozens of Codex runs in the same minutes. The disk itself is not mine to
manage -- the standing instruction is to report it, not act on it -- but losing a completed result to it IS mine to
prevent, and the fix is small.

WHAT IT DOES. `write_receipt(path, payload)` tries the intended path first. If that raises OSError -- ENOSPC, a
read-only mount, a missing parent -- it retries into the session scratchpad, and only then re-raises if even that
fails. It returns the path actually written and whether a fallback was used, so the runner can print it and the
result can be recovered by hand instead of being recomputed.

WHAT IT DELIBERATELY DOES NOT DO. It does not delete anything to make room, it does not retry the primary path, and
it does not silently swallow the error: a fallback write prints a loud line naming both paths, because a receipt
sitting in the scratchpad is NOT a booked receipt and must not be mistaken for one.
"""
from __future__ import annotations

import json
import os
import pathlib

FALLBACK_DIR = os.environ.get(
    "BQ_RECEIPT_FALLBACK",
    "/tmp/claude-0/-workspace/c538e58c-33f7-49ac-88b6-859af000fa92/scratchpad/rescued_receipts",
)


def write_receipt(path, payload, indent: int = 2) -> tuple[pathlib.Path, bool]:
    """Write `payload` as JSON to `path`; on OSError fall back to the scratchpad. Returns (path, used_fallback)."""
    target = pathlib.Path(path)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=indent, sort_keys=True)
    if not text.endswith("\n"):
        text += "\n"
    try:
        target.write_text(text)
        return target, False
    except OSError as err:
        fallback_dir = pathlib.Path(FALLBACK_DIR)
        fallback_dir.mkdir(parents=True, exist_ok=True)
        rescued = fallback_dir / target.name
        rescued.write_text(text)
        print(f"RECEIPT FALLBACK: {type(err).__name__}: {err}", flush=True)
        print(f"RECEIPT FALLBACK: wrote {rescued} instead of {target} -- "
              f"this is NOT a booked receipt; copy it back before releasing the claim", flush=True)
        return rescued, True
