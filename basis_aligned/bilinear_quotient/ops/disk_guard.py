#!/usr/bin/env python3
# BQGATE: LIBRARY -- shared disk-space guard (added 20 Sep, after the joint-tensor pipeline filled the instance's 32GB root disk to 0 bytes free by
# re-saving the same ~1.2GB composed matrix on every run). Import and call before ANY torch.save (or other large write) in a runner.
"""shutil.disk_usage-based guard: raises DiskBudgetError before a write would leave the filesystem below SAFETY_MARGIN_BYTES free, and separately
warns (prints, does not raise) once free space drops below WARN_MARGIN_BYTES so an approaching problem is visible in the runlog before it bites.
Also exposes `estimate_state_dict_bytes(d)` so a caller can check a planned torch.save's size BEFORE building/writing it, not after."""
from __future__ import annotations
import shutil

SAFETY_MARGIN_BYTES = 3 * 1024**3      # refuse a write that would leave less than this free
WARN_MARGIN_BYTES = 6 * 1024**3        # print a warning once free space is below this, even if the write is still allowed


class DiskBudgetError(RuntimeError):
    pass


def free_bytes(path: str = "/") -> int:
    return shutil.disk_usage(path).free


def estimate_tensor_bytes(obj) -> int:
    """Best-effort byte estimate for a torch tensor / nested dict-of-tensors, for a PRE-write check (no serialization overhead included)."""
    import torch
    if torch.is_tensor(obj):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, dict):
        return sum(estimate_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(estimate_tensor_bytes(v) for v in obj)
    return 0   # ints/strings/etc: negligible


EMERGENCY_FLOOR_BYTES = 512 * 1024**2   # refuse ANY write once free space is this low, regardless of the write's own size
NEGLIGIBLE_WRITE_BYTES = 20 * 1024**2   # a write this small (e.g. a JSON receipt) is never the thing that tips the disk over; only block it in a genuine emergency


def guard_write(planned_bytes: int, path: str = "/", label: str = "write") -> None:
    """Call BEFORE writing `planned_bytes` worth of data. Raises DiskBudgetError if (a) free space is already at the emergency floor regardless of
    size, or (b) this write is non-negligible AND would leave < SAFETY_MARGIN_BYTES free. A tiny write (a JSON receipt) is never blocked just
    because the margin is already tight from EARLIER large writes -- only the emergency floor stops it, so results are never lost to this guard."""
    free = free_bytes(path)
    if free < WARN_MARGIN_BYTES:
        print(f"[disk_guard] WARNING: only {free / 1e9:.2f} GB free on {path} before this {label} ({planned_bytes / 1e6:.1f} MB)")
    if free < EMERGENCY_FLOOR_BYTES:
        raise DiskBudgetError(f"[disk_guard] REFUSED: {label} -- only {free / 1e6:.1f} MB free on {path}, below the emergency floor ({EMERGENCY_FLOOR_BYTES / 1e6:.0f} MB). Free space before writing anything.")
    if planned_bytes > NEGLIGIBLE_WRITE_BYTES and free - planned_bytes < SAFETY_MARGIN_BYTES:
        raise DiskBudgetError(
            f"[disk_guard] REFUSED: {label} of {planned_bytes / 1e6:.1f} MB would leave {(free - planned_bytes) / 1e9:.2f} GB free "
            f"on {path}, below the {SAFETY_MARGIN_BYTES / 1e9:.0f} GB safety margin. Save a smaller object (factors, not composed matrices), "
            f"or delete old followups/*.pt files first."
        )


def guard_torch_save(obj, path: str, label: str | None = None) -> None:
    """Drop-in replacement for torch.save(obj, path): checks the budget first, then saves."""
    import torch
    planned = estimate_tensor_bytes(obj)
    guard_write(planned, path="/", label=label or f"torch.save -> {path}")
    torch.save(obj, path)
