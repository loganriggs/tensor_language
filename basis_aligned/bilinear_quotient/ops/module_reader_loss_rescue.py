"""Exact module-reader loss and complete-output rescue hooks.

The caller owns the upstream source intervention. This module changes only registered
sequence positions at named downstream module boundaries and always removes its hooks.
"""

# BQGATE: LIBRARY
from __future__ import annotations


class ReaderLossRescueError(RuntimeError):
    pass


def _validate_tensor_pair(current, replacement, position_rows):
    if current.ndim != 3 or replacement.shape != current.shape:
        raise ReaderLossRescueError("reader/output tensors must share [row, position, width]")
    if len(position_rows) != current.shape[0]:
        raise ReaderLossRescueError("position-row coverage changed")
    checked = []
    for positions in position_rows:
        row = tuple(int(position) for position in positions)
        if len(row) != len(set(row)) or any(
            position < 0 or position >= current.shape[1] for position in row
        ):
            raise ReaderLossRescueError("registered position is invalid")
        checked.append(row)
    return tuple(checked)


def replace_positions(current, replacement, position_rows):
    """Return ``current`` with only registered rows/positions taken from replacement."""
    checked = _validate_tensor_pair(current, replacement, position_rows)
    changed = current.clone()
    for row, positions in enumerate(checked):
        if positions:
            changed[row, list(positions)] = replacement[row, list(positions)].to(changed)
    return changed


def capture_reader_outputs(forward, modules):
    """Run once and capture argument-0 reader inputs plus complete tensor outputs."""
    modules = dict(modules)
    if not modules or len(modules) != len(set(modules)):
        raise ReaderLossRescueError("module labels must be unique and nonempty")
    readers, outputs = {}, {}
    calls = {label: {"reader": 0, "output": 0} for label in modules}
    handles = []

    def reader_hook(label):
        def hook(_module, arguments):
            calls[label]["reader"] += 1
            if not arguments or not hasattr(arguments[0], "detach"):
                raise ReaderLossRescueError(f"{label} reader argument changed")
            readers[label] = arguments[0].detach().clone()
        return hook

    def output_hook(label):
        def hook(_module, _arguments, output):
            calls[label]["output"] += 1
            if not hasattr(output, "detach"):
                raise ReaderLossRescueError(f"{label} output is not a tensor")
            outputs[label] = output.detach().clone()
        return hook

    for label, module in modules.items():
        handles.append(module.register_forward_pre_hook(reader_hook(label)))
        handles.append(module.register_forward_hook(output_hook(label)))
    try:
        result = forward()
    finally:
        for handle in handles:
            handle.remove()
    if any(record != {"reader": 1, "output": 1} for record in calls.values()):
        raise ReaderLossRescueError(f"capture hook coverage changed: {calls}")
    return result, readers, outputs, calls


def run_reader_loss_rescue(
    forward,
    modules,
    source_absent_readers,
    source_present_outputs,
    position_rows,
    *,
    losses,
    rescues=(),
):
    """Run with selected reader inputs removed and selected complete outputs rescued.

    ``rescues`` must be a subset of ``losses``: this keeps rescue diagnostic rather
    than turning it into an unconstrained output patch.
    """
    modules = dict(modules)
    losses, rescues = tuple(losses), tuple(rescues)
    if len(losses) != len(set(losses)) or len(rescues) != len(set(rescues)):
        raise ReaderLossRescueError("loss/rescue labels must be unique")
    if not set(rescues).issubset(losses):
        raise ReaderLossRescueError("output rescue requires the same reader loss")
    if not set(losses).issubset(modules):
        raise ReaderLossRescueError("unknown module label")
    if set(losses) - set(source_absent_readers) or set(rescues) - set(source_present_outputs):
        raise ReaderLossRescueError("missing captured reader or output authority")

    calls = {label: {"reader_loss": 0, "output_rescue": 0} for label in losses}
    handles = []

    def loss_hook(label):
        def hook(_module, arguments):
            calls[label]["reader_loss"] += 1
            if not arguments:
                raise ReaderLossRescueError(f"{label} reader argument changed")
            changed = replace_positions(
                arguments[0], source_absent_readers[label], position_rows
            )
            return (changed,) + tuple(arguments[1:])
        return hook

    def rescue_hook(label):
        def hook(_module, _arguments, output):
            calls[label]["output_rescue"] += 1
            return replace_positions(output, source_present_outputs[label], position_rows)
        return hook

    for label in losses:
        module = modules[label]
        handles.append(module.register_forward_pre_hook(loss_hook(label)))
        if label in rescues:
            handles.append(module.register_forward_hook(rescue_hook(label)))
    try:
        result = forward()
    finally:
        for handle in handles:
            handle.remove()
    expected = {
        label: {"reader_loss": 1, "output_rescue": int(label in rescues)}
        for label in losses
    }
    if calls != expected:
        raise ReaderLossRescueError(f"intervention hook coverage changed: {calls}")
    return result, calls
