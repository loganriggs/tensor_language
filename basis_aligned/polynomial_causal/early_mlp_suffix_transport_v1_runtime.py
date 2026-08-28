"""Pure differentiable runtime primitives for suffix-transport v1.

This module deliberately has no row, model-loader, artifact, or scoring access.  It
owns the one affine implementation shared by the L/R/S/T routes, the physical
projected replacement, deterministic fit permutations, and the two registered
training losses.  Model-specific orchestration must call these primitives rather
than defining route-specific forward paths.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import early_mlp_suffix_transport_v1 as contract


D_MODEL = contract.D_MODEL
CODE_DIM = contract.CODE_DIM
LEARNING_RATES = (1e-5, 3e-5, 1e-4)
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_CLIP_NORM = 1.0
SEQUENCE_LENGTH = 256
SCORE_START = 64
SCORE_STOP = 256


def _finite_tensor(name: str, value: Any, shape: tuple[int, ...]) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.detach().clone().float()


class AffineCodeProgram(nn.Module):
    """The common float32 ``((z-mean)/scale) W + bias`` code predictor."""

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        scale: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        mean_value = _finite_tensor("mean", mean, (D_MODEL,))
        scale_value = _finite_tensor("scale", scale, (D_MODEL,))
        if bool(torch.any(scale_value <= 0)):
            raise ValueError("scale must be strictly positive")
        self.register_buffer("mean", mean_value)
        self.register_buffer("scale", scale_value)
        self.weight = nn.Parameter(_finite_tensor("weight", weight, (D_MODEL, CODE_DIM)))
        self.bias = nn.Parameter(_finite_tensor("bias", bias, (CODE_DIM,)))

    @classmethod
    def from_v21_state(cls, state: Mapping[str, Any]) -> "AffineCodeProgram":
        required = {"grammar", "interface", "mean", "scale", "left", "right", "bias"}
        if not isinstance(state, Mapping) or not required.issubset(state):
            raise ValueError("v2.1 affine initialization is incomplete")
        if state["grammar"] != "affine" or state["interface"] != "state_complete_p":
            raise ValueError("suffix transport requires a state-complete affine initialization")
        left = _finite_tensor("left", state["left"], (D_MODEL, CODE_DIM))
        right = _finite_tensor("right", state["right"], (CODE_DIM, CODE_DIM))
        return cls(
            mean=state["mean"], scale=state["scale"],
            weight=left @ right, bias=state["bias"],
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(z) or z.shape[-1] != D_MODEL or not torch.isfinite(z).all():
            raise ValueError("z must be finite and end in d_model")
        flat = z.float().reshape(-1, D_MODEL)
        code = ((flat - self.mean) / self.scale) @ self.weight + self.bias
        return code.view(*z.shape[:-1], CODE_DIM)

    def frozen_normalization(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu().clone(), "scale": self.scale.detach().cpu().clone()}


class JointAffineProgram(nn.Module):
    """One shared executable implementation for local, suffix, singleton, and T arms."""

    def __init__(
        self,
        site0: AffineCodeProgram,
        site1: AffineCodeProgram,
        *,
        route: str,
    ) -> None:
        super().__init__()
        if route not in {"L", "R", "S0", "S1", "T"}:
            raise ValueError("unknown suffix-transport route")
        self._topology_sealed = False
        self._route = route
        self.site0 = site0
        self.site1 = site1
        self.cross = nn.Parameter(torch.zeros(CODE_DIM, CODE_DIM, dtype=torch.float32)) \
            if route == "T" else None
        self._topology_sealed = True
        self.set_route_trainability()

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_topology_sealed", False) and name in {"route", "_route", "cross"}:
            raise AttributeError("suffix-transport route/cross topology is immutable")
        super().__setattr__(name, value)

    @property
    def route(self) -> str:
        return self._route

    def _validate_topology(self) -> None:
        route = self._route
        if route not in {"L", "R", "S0", "S1", "T"}:
            raise RuntimeError("suffix-transport route identity changed")
        cross = self._parameters.get("cross")
        if route == "T":
            if not isinstance(cross, nn.Parameter) or tuple(cross.shape) != (
                CODE_DIM, CODE_DIM,
            ) or not torch.isfinite(cross).all():
                raise RuntimeError("T cross topology changed")
        elif cross is not None:
            raise RuntimeError("non-T route acquired a cross parameter")

    @classmethod
    def from_v21_states(
        cls, states: Mapping[int, Mapping[str, Any]], *, route: str,
    ) -> "JointAffineProgram":
        if set(states) != {0, 1}:
            raise ValueError("joint initialization requires exactly sites 0 and 1")
        return cls(
            AffineCodeProgram.from_v21_state(states[0]),
            AffineCodeProgram.from_v21_state(states[1]),
            route=route,
        )

    def independent_clone(self, *, route: str | None = None) -> "JointAffineProgram":
        requested = self.route if route is None else route
        clone = JointAffineProgram(
            copy.deepcopy(self.site0), copy.deepcopy(self.site1), route=requested,
        )
        if requested == "T" and self.cross is not None:
            with torch.no_grad():
                clone.cross.copy_(self.cross)
        return clone

    def site0_code(self, z0: torch.Tensor) -> torch.Tensor:
        self._validate_topology()
        return self.site0(z0)

    def set_route_trainability(self) -> tuple[nn.Parameter, ...]:
        """Freeze the exact registered parameter set and return its stable order."""

        self._validate_topology()
        if self.cross is not None and not torch.count_nonzero(self.cross.detach()) == 0:
            raise RuntimeError("T must enter training from exactly zero A")
        site0_on = self.route in {"L", "R", "S0"}
        site1_on = self.route in {"L", "R", "S1"}
        for parameter in self.site0.parameters():
            parameter.requires_grad_(site0_on)
        for parameter in self.site1.parameters():
            parameter.requires_grad_(site1_on)
        if self.cross is not None:
            self.cross.requires_grad_(self.route == "T")
        ordered = (
            self.site0.weight, self.site0.bias, self.site1.weight, self.site1.bias,
            *((self.cross,) if self.cross is not None else ()),
        )
        return tuple(parameter for parameter in ordered if parameter.requires_grad)

    def site1_code(self, z1: torch.Tensor, parent_code: torch.Tensor | None = None) -> torch.Tensor:
        self._validate_topology()
        local = self.site1(z1)
        if self.route != "T":
            if parent_code is not None:
                raise ValueError("L/R/S programs cannot consume a parent code")
            return local
        if self.cross is None:
            raise RuntimeError("T route lacks its dense cross map")
        if parent_code is None or parent_code.shape != local.shape:
            raise ValueError("transport requires same-shaped executable parent code")
        if not torch.isfinite(parent_code).all():
            raise ValueError("parent code must be finite")
        return local + parent_code.float() @ self.cross

    @staticmethod
    def projected_replacement(
        deployed_output: torch.Tensor, predicted_code: torch.Tensor, basis: torch.Tensor,
    ) -> torch.Tensor:
        if deployed_output.shape[:-1] != predicted_code.shape[:-1] or (
            deployed_output.shape[-1] != D_MODEL or predicted_code.shape[-1] != CODE_DIM
        ):
            raise ValueError("deployed output and code shapes are incompatible")
        contract.validate_orthonormal_basis("basis", basis)
        basis_value = basis.to(device=deployed_output.device, dtype=torch.float32)
        deployed = deployed_output.float()
        live_code = deployed.reshape(-1, D_MODEL) @ basis_value
        replacement = (predicted_code.float().reshape(-1, CODE_DIM) - live_code) @ basis_value.T
        return deployed_output + replacement.view_as(deployed_output).to(deployed_output.dtype)


class StudentCorrectionHook:
    """Original-free N/P student execution with one-use same-forward transport."""

    def __init__(self, bases: Mapping[int, torch.Tensor]) -> None:
        if set(bases) != {0, 1}:
            raise ValueError("runtime requires exactly the MLP0/1 bases")
        self.bases = {
            site: _finite_tensor(f"basis{site}", value, (D_MODEL, CODE_DIM))
            for site, value in bases.items()
        }
        for site, value in self.bases.items():
            contract.validate_orthonormal_basis(f"basis{site}", value)
        self.program: JointAffineProgram | None = None
        self.states: dict[int, str] = {}
        self.capture_sites: frozenset[int] = frozenset()
        self.site0_edit: torch.Tensor | None = None
        self.parent_code: torch.Tensor | None = None
        self.captured_z: dict[int, list[torch.Tensor]] = {0: [], 1: []}
        self.calls = {0: 0, 1: 0}
        self.scope_calls = {0: 0, 1: 0}
        self.forward_nonce: str | None = None
        self.parent_nonce: str | None = None
        self.parent_consumed = False
        self.active = False

    def configure(
        self,
        *,
        program: JointAffineProgram | None,
        states: Mapping[int, str],
        capture_sites: Iterable[int] = (),
        site0_edit: torch.Tensor | None = None,
    ) -> None:
        allowed = {0: {"N", "P"}, 1: {"N", "P"}}
        if any(site not in allowed or state not in allowed[site] for site, state in states.items()):
            raise ValueError("student runtime permits only MLP0/1 N/P states")
        captures = frozenset(capture_sites)
        if not captures.issubset({0, 1}):
            raise ValueError("only MLP0/1 student inputs may be captured")
        if any(state == "P" for state in states.values()) and program is None:
            raise ValueError("predicted states require an executable program")
        if program is not None and not isinstance(program, JointAffineProgram):
            raise ValueError("program has the wrong runtime type")
        if site0_edit is not None and (
            not torch.is_tensor(site0_edit) or site0_edit.shape[-1] != CODE_DIM
            or not torch.isfinite(site0_edit).all()
        ):
            raise ValueError("site0 edit must be finite and end in code_dim")
        if site0_edit is not None and (states.get(0, "N") != "P" or program is None):
            raise ValueError("site0 edit requires an executable predicted MLP0 state")
        self.program = program
        self.states = dict(states)
        self.capture_sites = captures
        self.site0_edit = site0_edit
        self.parent_code = None
        self.captured_z = {0: [], 1: []}
        self.calls = {0: 0, 1: 0}
        self.scope_calls = {0: 0, 1: 0}
        self.forward_nonce = None
        self.parent_nonce = None
        self.parent_consumed = False

    @contextmanager
    def forward_scope(self, nonce: str):
        if self.active:
            raise RuntimeError("runtime forward scope is already active")
        if not isinstance(nonce, str) or not nonce:
            raise ValueError("forward nonce must be a nonempty string")
        self.active = True
        self.scope_calls = {0: 0, 1: 0}
        self.forward_nonce = nonce
        self.parent_code = None
        self.parent_nonce = None
        self.parent_consumed = False
        try:
            yield self
        finally:
            self.parent_code = None
            self.parent_nonce = None
            self.forward_nonce = None
            self.active = False

    def captured_inputs(self) -> dict[int, torch.Tensor]:
        if any(not self.captured_z[site] for site in self.capture_sites):
            raise RuntimeError("not every requested current-state input was captured")
        return {
            site: torch.cat(self.captured_z[site], dim=0)
            for site in sorted(self.capture_sites)
        }

    def __call__(
        self, site: int, z: torch.Tensor, mo: torch.Tensor, *, forward_nonce: str,
    ) -> torch.Tensor:
        if not self.active or forward_nonce != self.forward_nonce:
            raise RuntimeError("runtime call occurred outside a forward scope")
        if site not in (0, 1):
            raise ValueError("student runtime is restricted to MLP0/1")
        if self.scope_calls[site] != 0:
            raise RuntimeError(f"student MLP{site} was called more than once in one forward")
        self.scope_calls[site] += 1
        self.calls[site] += 1
        if site in self.capture_sites:
            self.captured_z[site].append(z.detach().cpu().contiguous())
        state = self.states.get(site, "N")
        if state == "N":
            if site == 0 and self.program is not None and self.program.route == "T":
                raise RuntimeError("T cannot source its parent from native/deployed MLP0")
            return mo
        if self.program is None:
            raise RuntimeError("predicted student runtime lacks a program")
        if site == 0:
            predicted = self.program.site0_code(z)
        else:
            if self.program.route == "T":
                if self.parent_code is None or self.parent_nonce != forward_nonce or (
                    self.parent_consumed
                ):
                    raise RuntimeError("T lacks one unused same-forward executable parent")
                predicted = self.program.site1_code(z, self.parent_code)
                self.parent_consumed = True
                self.parent_code = None
            else:
                predicted = self.program.site1_code(z)

        if site == 0 and self.site0_edit is not None:
            if self.site0_edit.shape != predicted.shape:
                raise RuntimeError("site0 edit shape differs from executable code")
            predicted = predicted + self.site0_edit.to(predicted.device, torch.float32)
        if site == 0:
            if self.parent_code is not None:
                raise RuntimeError("student parent code would be overwritten")
            self.parent_code = predicted
            self.parent_nonce = forward_nonce
        return JointAffineProgram.projected_replacement(mo, predicted, self.bases[site])


def scored_positions(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim < 2 or value.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("scored tensor must contain all 256 model positions")
    return value[:, SCORE_START:SCORE_STOP]


def _canonical_code_support(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] != CODE_DIM:
        raise ValueError("code support must be [batch, token, code_dim]")
    if value.shape[1] == SEQUENCE_LENGTH:
        return scored_positions(value)
    if value.shape[1] != SCORE_STOP - SCORE_START:
        raise ValueError("code support must be full 256 or exact scored 192 positions")
    return value


@dataclass(frozen=True)
class MomentSufficientStatistics:
    """Mergeable float64 raw statistics for one site's frozen denominator."""

    count: int
    coordinate_sum: torch.Tensor
    coordinate_square_sum: torch.Tensor
    mean: torch.Tensor
    centered_m2: torch.Tensor

    @classmethod
    def from_labels(cls, labels: torch.Tensor) -> "MomentSufficientStatistics":
        labels = _canonical_code_support(labels)
        if labels.shape[0] <= 0 or not torch.isfinite(labels).all():
            raise ValueError("labels must contain finite scored code vectors")
        values = labels.detach().cpu().double().reshape(-1, CODE_DIM)
        mean = values.mean(dim=0)
        return cls(
            count=int(values.shape[0]),
            coordinate_sum=values.sum(dim=0),
            coordinate_square_sum=values.square().sum(dim=0),
            mean=mean,
            centered_m2=torch.sum((values - mean).square(), dim=0),
        )

    def merge(self, other: "MomentSufficientStatistics") -> "MomentSufficientStatistics":
        if not isinstance(other, MomentSufficientStatistics):
            raise TypeError("moment statistics can merge only with their own type")
        count = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * (other.count / count)
        centered_m2 = self.centered_m2 + other.centered_m2 + delta.square() * (
            self.count * other.count / count
        )
        return MomentSufficientStatistics(
            count=count,
            coordinate_sum=self.coordinate_sum + other.coordinate_sum,
            coordinate_square_sum=self.coordinate_square_sum + other.coordinate_square_sum,
            mean=mean,
            centered_m2=centered_m2,
        )

    def finalize(
        self, *, expected_count: int, ordered_support_sha256: str,
    ) -> Mapping[str, Any]:
        if self.count != expected_count:
            raise ValueError(f"moment support count changed: {self.count}!={expected_count}")
        if not isinstance(ordered_support_sha256, str) or len(ordered_support_sha256) != 64 \
                or any(character not in "0123456789abcdef" for character in ordered_support_sha256):
            raise ValueError("ordered support hash is malformed")
        if self.count < 2 or tuple(self.coordinate_sum.shape) != (CODE_DIM,) or (
            tuple(self.coordinate_square_sum.shape) != (CODE_DIM,)
        ) or not torch.isfinite(self.coordinate_sum).all() or not torch.isfinite(
            self.coordinate_square_sum
        ).all() or tuple(self.mean.shape) != (CODE_DIM,) or tuple(
            self.centered_m2.shape
        ) != (CODE_DIM,) or not torch.isfinite(self.mean).all() or not torch.isfinite(
            self.centered_m2
        ).all():
            raise ValueError("moment sufficient statistics are malformed")
        raw_centered_sum = torch.sum(
            self.coordinate_square_sum - self.coordinate_sum.square() / self.count
        )
        centered_sum = self.centered_m2.sum()
        denominator = centered_sum / (self.count * CODE_DIM)
        if not torch.isfinite(denominator) or float(denominator) <= 0:
            raise ValueError("label second moment must be positive and finite")
        return {
            "count": self.count,
            "coordinate_sum": self.coordinate_sum.clone(),
            "coordinate_square_sum": self.coordinate_square_sum.clone(),
            "mean": self.mean.clone(),
            "centered_sum_of_squares": centered_sum,
            "raw_sum_square_replay": raw_centered_sum,
            "denominator": denominator,
            "ordered_support_sha256": ordered_support_sha256,
        }


def centered_second_moment(
    labels: torch.Tensor, *, ordered_support_sha256: str,
) -> torch.Tensor:
    """Return the frozen scalar float64 centered second moment for one site."""

    return MomentSufficientStatistics.from_labels(labels).finalize(
        expected_count=384 * (SCORE_STOP - SCORE_START),
        ordered_support_sha256=ordered_support_sha256,
    )["denominator"]


def normalized_local_loss(
    predictions: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    denominators: Sequence[torch.Tensor | float],
) -> torch.Tensor:
    """Registered L loss; labels and frozen denominators never receive gradients."""

    if len(predictions) != 2 or len(labels) != 2 or len(denominators) != 2:
        raise ValueError("local loss requires exactly the two MLP0/1 sites")
    terms = []
    for prediction, label, denominator in zip(predictions, labels, denominators, strict=True):
        prediction = _canonical_code_support(prediction)
        label = _canonical_code_support(label)
        if prediction.shape != label.shape:
            raise ValueError("local prediction/label shape changed")
        target = label.detach().to(device=prediction.device, dtype=torch.float32)
        scale = torch.as_tensor(denominator, dtype=torch.float64).detach()
        if scale.numel() != 1 or not torch.isfinite(scale) or float(scale) <= 0:
            raise ValueError("local loss denominator must be positive and finite")
        terms.append(F.mse_loss(prediction.float(), target) / scale.float().to(prediction.device))
    return terms[0] + terms[1]


def teacher_student_kl(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Token-weighted ``KL(teacher || student)`` with the teacher detached."""

    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 3 or (
        teacher_logits.shape[-1] <= 1
    ):
        raise ValueError("teacher/student logits must be same-shaped [batch, token, vocab]")
    if not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all():
        raise ValueError("teacher/student logits must be finite")
    if teacher_logits.shape[1] == SEQUENCE_LENGTH:
        teacher_logits = scored_positions(teacher_logits)
        student_logits = scored_positions(student_logits)
    elif teacher_logits.shape[1] != SCORE_STOP - SCORE_START:
        raise ValueError("suffix KL support must be positions 64 through 255")
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    return torch.mean(torch.sum(teacher_logp.exp() * (teacher_logp - student_logp), dim=-1))


def fit_permutations(row_count: int, trial: int) -> tuple[torch.Tensor, ...]:
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("row_count must be a positive integer")
    if isinstance(trial, bool) or not isinstance(trial, int) or trial not in range(3):
        raise ValueError("trial must index the three registered learning rates")
    permutations = []
    for epoch in range(EPOCHS):
        generator = torch.Generator(device="cpu").manual_seed(2026083000 + 100 * trial + epoch)
        permutations.append(torch.randperm(row_count, generator=generator))
    return tuple(permutations)


def batch_indices(row_count: int, trial: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        permutation[start:start + BATCH_SIZE]
        for permutation in fit_permutations(row_count, trial)
        for start in range(0, row_count, BATCH_SIZE)
    )


def make_optimizer(parameters: Iterable[nn.Parameter], learning_rate: float) -> torch.optim.AdamW:
    if learning_rate not in LEARNING_RATES:
        raise ValueError("learning rate is outside the frozen grid")
    values = list(parameters)
    if not values or any(not isinstance(value, nn.Parameter) for value in values):
        raise ValueError("optimizer requires explicit trainable parameters")
    return torch.optim.AdamW(
        values, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )


def optimizer_step(loss: torch.Tensor, optimizer: torch.optim.AdamW) -> float:
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("optimizer loss must be a finite scalar")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [value for group in optimizer.param_groups for value in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP_NORM)
    if not torch.isfinite(norm):
        raise RuntimeError("gradient norm is not finite")
    optimizer.step()
    return float(norm.detach().cpu())
