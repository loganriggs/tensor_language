"""Physical replay primitives for the C512 -> MLP2 compensation assay.

This module deliberately contains no row selection, thresholds, inference, or
authority mutation.  Exact/candidate MLP1 captures are produced by the preceding
interchange primitive; this module constructs the O/C upstream paths, captures the
physical MLP2 state and write, crosses them, and replays the unchanged layer-3 suffix.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


UPSTREAM_PATHS = ("O", "C")
PHYSICAL_ARMS = tuple(p + q for p in UPSTREAM_PATHS for q in UPSTREAM_PATHS)


def _require_same_shape(values: Mapping[str, torch.Tensor], name: str) -> None:
    shapes = {tuple(value.shape) for value in values.values()}
    if len(shapes) != 1:
        raise ValueError(f"{name} tensors have different shapes")


def post_mlp1_paths(
    exact: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor], *,
    state_identity_tolerance: float = 1e-6,
) -> dict[str, dict[str, torch.Tensor]]:
    """Construct exact O and observational C post-MLP1 paths."""
    required = ("s", "m", "post", "v1", "x0")
    if any(key not in exact or key not in candidate for key in required):
        raise ValueError("MLP1 captures are incomplete")
    _require_same_shape({"exact_s": exact["s"], "candidate_s": candidate["s"]}, "MLP1 state")
    _require_same_shape({"exact_m": exact["m"], "candidate_m": candidate["m"]}, "MLP1 write")
    if not (torch.isfinite(torch.tensor(state_identity_tolerance))
            and state_identity_tolerance >= 0):
        raise ValueError("state identity tolerance is invalid")
    for key in ("x0", "v1"):
        _require_same_shape({"exact": exact[key], "candidate": candidate[key]}, key)
        if not torch.isfinite(exact[key]).all() or not torch.isfinite(candidate[key]).all():
            raise ValueError(f"{key} contains a nonfinite value")
        difference = float((exact[key] - candidate[key]).abs().max())
        if difference > state_identity_tolerance:
            raise ValueError(f"{key} exact/candidate identity exceeds tolerance")
    return {
        "O": {"post": exact["post"], "v1": exact["v1"], "x0": exact["x0"]},
        "C": {"post": candidate["post"], "v1": candidate["v1"], "x0": candidate["x0"]},
    }


@torch.no_grad()
def capture_mlp2_interface(
    block2: Any,
    post_mlp1: torch.Tensor,
    v1: torch.Tensor,
    x0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run block-2 mixing/attention and capture its pre-write state and write."""
    if post_mlp1.shape != x0.shape or post_mlp1.ndim != 3:
        raise ValueError("post-MLP1 state and x0 must be aligned rank-3 tensors")
    d_model = int(post_mlp1.shape[-1])
    mixed = block2.lambdas[0] * post_mlp1 + block2.lambdas[1] * x0
    attn, carried = block2.attn(F.rms_norm(mixed, (d_model,)), v1)
    state = mixed + attn
    write = block2.mlp(F.rms_norm(state, (d_model,)))
    return {
        "mixed": mixed,
        "attn": attn,
        "s": state,
        "m": write,
        "post": state + write,
        "v1": carried,
        "x0": x0,
    }


@torch.no_grad()
def capture_mlp2_paths(
    blocks: Sequence[Any], paths: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, dict[str, torch.Tensor]]:
    """Capture the MLP2 interface for the registered O/C upstream paths."""
    if len(blocks) < 4 or set(paths) != set(UPSTREAM_PATHS):
        raise ValueError("the MLP2 assay requires blocks 0--3 and exactly O/C paths")
    return {
        path: capture_mlp2_interface(
            blocks[2], values["post"], values["v1"], values["x0"]
        )
        for path, values in paths.items()
    }


def physical_mlp2_matrix(
    interfaces: Mapping[str, Mapping[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Construct X_pq = state from path p plus MLP2 write from path q."""
    if set(interfaces) != set(UPSTREAM_PATHS):
        raise ValueError("MLP2 interfaces must contain exactly O/C")
    _require_same_shape({path: values["s"] for path, values in interfaces.items()}, "MLP2 state")
    _require_same_shape({path: values["m"] for path, values in interfaces.items()}, "MLP2 write")
    return {
        p + q: interfaces[p]["s"] + interfaces[q]["m"]
        for p in UPSTREAM_PATHS
        for q in UPSTREAM_PATHS
    }


@torch.no_grad()
def suffix_from_mlp2(
    model: Any,
    blocks: Sequence[Any],
    post_mlp2: torch.Tensor,
    v1: torch.Tensor,
    x0: torch.Tensor,
    *,
    return_raw: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Replay blocks 3..end and return capped, optionally raw, logits."""
    if len(blocks) < 4 or post_mlp2.shape != x0.shape:
        raise ValueError("the suffix requires blocks 3..end and aligned residuals")
    d_model = int(post_mlp2.shape[-1])
    x = post_mlp2
    carried = v1
    for block in blocks[3:]:
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attn, carried = block.attn(F.rms_norm(x, (d_model,)), carried)
        x = x + attn
        write = block.mlp(F.rms_norm(x, (d_model,)))
        x = x + write
    raw = model.lm_head(F.rms_norm(x, (d_model,))).float()
    capped = 30.0 * torch.tanh(raw / 30.0)
    values: list[torch.Tensor] = []
    if return_raw:
        values.append(raw)
    values.append(capped)
    return tuple(values) if len(values) > 1 else values[0]


def centered_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def additive_factorial_prediction(logits: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Gauge-fix the no-interaction CC prediction CO + OC - OO."""
    required = ("OO", "CO", "OC")
    if any(key not in logits for key in required):
        raise ValueError("factorial logits are incomplete")
    return centered_logits(logits["CO"] + logits["OC"] - logits["OO"])


def norm_matched_native_write(
    delta_write: torch.Tensor, native_write: torch.Tensor, *,
    atol: float = 1e-6, rtol: float = 1e-5,
) -> torch.Tensor:
    """Scale the exact MLP2 write per position to the delta-write norm."""
    if delta_write.shape != native_write.shape:
        raise ValueError("write tensors have different shapes")
    if (not torch.isfinite(delta_write).all() or not torch.isfinite(native_write).all()
            or atol < 0 or rtol < 0):
        raise ValueError("write control inputs or tolerances are invalid")
    target = delta_write.float().norm(dim=-1, keepdim=True)
    source = native_write.float().norm(dim=-1, keepdim=True)
    if bool(((source == 0) & (target > 0)).any()):
        raise ValueError("cannot norm-match a nonzero delta to a zero native write")
    scale = torch.where(source > 0, target / source, torch.zeros_like(source))
    if not torch.isfinite(scale).all():
        raise ValueError("native-write norm scale is nonfinite")
    control = (native_write.float() * scale).to(native_write.dtype)
    realized = control.float().norm(dim=-1, keepdim=True)
    if bool((torch.abs(realized - target) > atol + rtol * target).any()):
        raise RuntimeError("native-write control failed its realized norm invariant")
    return control
