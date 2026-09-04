"""Narrow tensor backend primitives for Task 14 projector Program A.

The production model forward and the fit objective are injected.  This keeps
the DISCOVERY shard, differentiable head hook, and orthogonal optimizer
testable on CPU without importing a checkpoint or choosing scientific loss
constants in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils import parametrizations

import run_task14_head11_3_projector_discovery as program


OPS = Path(__file__).resolve().parent
SHARD_PATH = OPS / "task14_projector_discovery_endpoint_shard_v1.json"
SHARD_SHA256 = "1e3b9a204c08a9c6af4ea7f5668abba719fd1943a8a7e7df0dc488f3183f4e1b"
MODEL_WIDTH = 1152
HEAD_WIDTH = 128
HEAD_START = 3 * HEAD_WIDTH
HEAD_STOP = 4 * HEAD_WIDTH


class Task14BackendError(ValueError):
    """The frozen shard or differentiable model boundary is invalid."""


@dataclass(frozen=True)
class Endpoint:
    endpoint_id: str
    token_ids: tuple[int, ...]
    final_position: int
    answer_id: int
    foil_id: int
    cell_metadata: tuple[object, ...]


@dataclass(frozen=True)
class OrthogonalFitResult:
    frame: torch.Tensor
    losses: tuple[float, ...]
    maximum_orthonormality_error: float
    gradients_finite: bool


TensorForward = Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]
FrameObjective = Callable[[torch.Tensor], torch.Tensor]
OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_discovery_endpoints(
    path: Path = SHARD_PATH, *, expected_sha256: str = SHARD_SHA256
) -> dict[str, Endpoint]:
    """Load only the committed DISCOVERY shard and validate its narrow schema."""

    if _sha256(path) != expected_sha256:
        raise Task14BackendError("DISCOVERY endpoint shard hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("endpoints") if isinstance(payload, dict) else None
    if (
        payload.get("partition") != "DISCOVERY"
        or payload.get("endpoint_count") != 128
        or payload.get("group_count") != 16
        or not isinstance(rows, list)
        or len(rows) != 128
    ):
        raise Task14BackendError("DISCOVERY endpoint shard identity changed")
    output: dict[str, Endpoint] = {}
    groups: set[object] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("partition") != "DISCOVERY":
            raise Task14BackendError("non-DISCOVERY endpoint entered Program A")
        endpoint_id = row.get("endpoint_id")
        ids = row.get("ids")
        position = row.get("final_position")
        answer_id, foil_id = row.get("answer_id"), row.get("foil_id")
        if (
            not isinstance(endpoint_id, str)
            or endpoint_id in output
            or not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or token < 0 for token in ids)
            or type(position) is not int
            or position != len(ids) - 1
            or type(answer_id) is not int
            or type(foil_id) is not int
            or answer_id == foil_id
        ):
            raise Task14BackendError("DISCOVERY endpoint row is malformed")
        groups.add(row.get("group_id"))
        output[endpoint_id] = Endpoint(
            endpoint_id=endpoint_id,
            token_ids=tuple(ids),
            final_position=position,
            answer_id=answer_id,
            foil_id=foil_id,
            cell_metadata=(
                row.get("group_id"),
                row.get("group_number"),
                row.get("family"),
                row.get("subject_state"),
            ),
        )
    if len(output) != 128 or len(groups) != 16 or None in groups:
        raise Task14BackendError("DISCOVERY endpoint/group census changed")
    return output


def fit_householder_frame(
    initial_frame: torch.Tensor,
    *,
    objective: FrameObjective,
    updates: int,
    optimizer_factory: OptimizerFactory,
) -> OrthogonalFitResult:
    """Optimize an injected objective through PyTorch's Householder map.

    No learning rate, loss, regularizer, or update count is selected here.
    """

    if (
        not isinstance(initial_frame, torch.Tensor)
        or initial_frame.ndim != 2
        or initial_frame.shape[1] <= 0
        or initial_frame.shape[0] < initial_frame.shape[1]
        or not initial_frame.is_floating_point()
        or not bool(torch.isfinite(initial_frame).all())
        or type(updates) is not int
        or updates <= 0
        or not callable(objective)
        or not callable(optimizer_factory)
    ):
        raise Task14BackendError("orthogonal-fit inputs are malformed")
    ambient, rank = initial_frame.shape
    identity = torch.eye(rank, device=initial_frame.device, dtype=initial_frame.dtype)
    if not torch.allclose(initial_frame.T @ initial_frame, identity, atol=1e-6, rtol=0):
        raise Task14BackendError("initial frame must have orthonormal columns")

    holder = nn.Linear(rank, ambient, bias=False, device=initial_frame.device,
                       dtype=initial_frame.dtype)
    with torch.no_grad():
        holder.weight.copy_(initial_frame)
    parametrizations.orthogonal(
        holder, "weight", orthogonal_map="householder", use_trivialization=True
    )
    optimizer = optimizer_factory(holder.parametrizations.weight.parameters())
    losses: list[float] = []
    gradients_finite = True
    maximum_error = 0.0
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        frame = holder.weight
        loss = objective(frame)
        if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
            raise Task14BackendError("injected frame objective returned invalid loss")
        loss.backward()
        parameters = tuple(holder.parametrizations.weight.parameters())
        if not parameters or any(
            parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            gradients_finite = False
            raise Task14BackendError("Householder parameter gradient is absent or nonfinite")
        optimizer.step()
        with torch.no_grad():
            frame = holder.weight
            error = float(torch.max(torch.abs(frame.T @ frame - identity)))
            maximum_error = max(maximum_error, error)
            losses.append(float(loss.detach()))
    return OrthogonalFitResult(
        frame=holder.weight.detach().clone(),
        losses=tuple(losses),
        maximum_orthonormality_error=maximum_error,
        gradients_finite=gradients_finite,
    )


class Task14ProgramATorchBackend:
    """DISCOVERY-only differentiable collector for Program A.

    ``forward_logits`` must run the frozen model and call ``c_proj_module`` at
    attention layer 11.  Its returned logits stay as tensors.
    """

    def __init__(
        self,
        *,
        forward_logits: TensorForward,
        c_proj_module: nn.Module,
        device: str | torch.device,
        frozen_parameters: Sequence[nn.Parameter] = (),
        shard_path: Path = SHARD_PATH,
        shard_sha256: str = SHARD_SHA256,
        batch_size: int = 32,
        denominator_floor: float = 1e-12,
    ) -> None:
        if not callable(forward_logits) or not isinstance(c_proj_module, nn.Module):
            raise Task14BackendError("forward/c_proj boundary is malformed")
        if type(batch_size) is not int or batch_size <= 0:
            raise Task14BackendError("batch_size must be positive")
        if not math.isfinite(denominator_floor) or denominator_floor <= 0:
            raise Task14BackendError("denominator_floor must be positive and finite")
        if any(parameter.requires_grad for parameter in frozen_parameters):
            raise Task14BackendError("model parameters must be frozen")
        self.forward_logits = forward_logits
        self.c_proj_module = c_proj_module
        self.device = torch.device(device)
        self.endpoints = load_discovery_endpoints(
            shard_path, expected_sha256=shard_sha256
        )
        self.batch_size = batch_size
        self.denominator_floor = denominator_floor
        self.frozen_parameters = tuple(frozen_parameters)
        self.native_heads: dict[str, torch.Tensor] = {}
        self.native_logits: dict[str, torch.Tensor] = {}
        self.full_head_effects: dict[int, float] = {}

    def _chunks(self, values: Sequence[object]):
        for start in range(0, len(values), self.batch_size):
            yield values[start:start + self.batch_size]

    def _tensorize(self, endpoints: Sequence[Endpoint]):
        lengths = tuple(len(endpoint.token_ids) for endpoint in endpoints)
        maximum = max(lengths)
        tokens = torch.zeros(
            len(endpoints), maximum, dtype=torch.long, device=self.device
        )
        for index, endpoint in enumerate(endpoints):
            tokens[index, :lengths[index]] = torch.tensor(
                endpoint.token_ids, dtype=torch.long, device=self.device
            )
        return tokens, lengths

    def _forward(
        self,
        endpoints: Sequence[Endpoint],
        *,
        donor_heads: torch.Tensor | None = None,
        frame: torch.Tensor | None = None,
        require_activation_gradient: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, lengths = self._tensorize(endpoints)
        captured: list[torch.Tensor] = []
        hook_calls = 0

        def hook(_module, arguments):
            nonlocal hook_calls
            hook_calls += 1
            value = arguments[0]
            if tuple(value.shape) != (*tokens.shape, MODEL_WIDTH):
                raise Task14BackendError("layer-11 c_proj input shape changed")
            if require_activation_gradient:
                value = value.detach().requires_grad_(True)
            captured.append(value)
            if donor_heads is None:
                return (value,) + tuple(arguments[1:])
            positions = torch.tensor(
                [endpoint.final_position for endpoint in endpoints],
                dtype=torch.long,
                device=value.device,
            )
            rows = torch.arange(len(endpoints), device=value.device)
            recipient = value[rows, positions, HEAD_START:HEAD_STOP]
            donor = donor_heads.to(device=value.device, dtype=value.dtype)
            if tuple(donor.shape) != tuple(recipient.shape):
                raise Task14BackendError("donor head batch shape changed")
            patched = donor if frame is None else recipient + (
                (donor - recipient) @ frame
            ) @ frame.T
            changed = value.clone()
            changed[rows, positions, HEAD_START:HEAD_STOP] = patched
            return (changed,) + tuple(arguments[1:])

        handle = self.c_proj_module.register_forward_pre_hook(hook)
        try:
            logits = self.forward_logits(tokens, lengths)
        finally:
            handle.remove()
        if hook_calls != 1 or len(captured) != 1:
            raise Task14BackendError("layer-11 c_proj hook did not fire exactly once")
        if logits.ndim != 3 or logits.shape[:2] != tokens.shape:
            raise Task14BackendError("model returned malformed unsliced logits")
        if not bool(torch.isfinite(logits).all()):
            raise Task14BackendError("model returned nonfinite logits")
        rows = torch.arange(len(endpoints), device=logits.device)
        positions = torch.tensor(
            [length - 1 for length in lengths], device=logits.device
        )
        return logits[rows, positions], captured[0]

    def collect_spectral_inputs(
        self, relations: Sequence[program.Relation]
    ) -> program.SpectralInputs:
        """Collect exact ``d_i``, local ``g_i``, and finite full-head ``E_h``."""

        relations = tuple(relations)
        if not relations or any(relation.role != "target" for relation in relations):
            raise Task14BackendError("spectral collection accepts target relations only")
        if len({relation.ordinal for relation in relations}) != len(relations):
            raise Task14BackendError("spectral relation ordinals are duplicated")
        needed_ids = sorted({
            endpoint_id for relation in relations for endpoint_id in (
                relation.target_endpoint_id, relation.donor_endpoint_id
            )
        })
        try:
            needed = [self.endpoints[endpoint_id] for endpoint_id in needed_ids]
        except KeyError as error:
            raise Task14BackendError("relation escaped the DISCOVERY shard") from error

        score_pairs: dict[str, tuple[int, int]] = {}
        for relation in relations:
            donor = self.endpoints[relation.donor_endpoint_id]
            pair = (donor.answer_id, donor.foil_id)
            previous = score_pairs.setdefault(relation.target_endpoint_id, pair)
            if previous != pair:
                raise Task14BackendError("one target endpoint has inconsistent donor scores")

        gradients: dict[str, torch.Tensor] = {}
        forward_calls = backward_calls = example_evaluations = 0
        for chunk_values in self._chunks(needed):
            chunk = tuple(chunk_values)
            logits, activation = self._forward(
                chunk, require_activation_gradient=True
            )
            forward_calls += 1
            example_evaluations += len(chunk)
            terms = [
                logits[index, score_pairs[endpoint.endpoint_id][0]]
                - logits[index, score_pairs[endpoint.endpoint_id][1]]
                for index, endpoint in enumerate(chunk)
                if endpoint.endpoint_id in score_pairs
            ]
            if terms:
                gradient = torch.autograd.grad(torch.stack(terms).sum(), activation)[0]
                backward_calls += 1
            else:
                gradient = torch.zeros_like(activation)
            for index, endpoint in enumerate(chunk):
                position = endpoint.final_position
                self.native_heads[endpoint.endpoint_id] = activation[
                    index, position, HEAD_START:HEAD_STOP
                ].detach().cpu().to(torch.float64)
                self.native_logits[endpoint.endpoint_id] = logits[index].detach().cpu()
                if endpoint.endpoint_id in score_pairs:
                    gradients[endpoint.endpoint_id] = gradient[
                        index, position, HEAD_START:HEAD_STOP
                    ].detach().cpu().to(torch.float64)

        effects: list[float] = []
        for chunk_values in self._chunks(relations):
            chunk = tuple(chunk_values)
            target_endpoints = tuple(
                self.endpoints[relation.target_endpoint_id] for relation in chunk
            )
            donor_heads = torch.stack([
                self.native_heads[relation.donor_endpoint_id] for relation in chunk
            ]).to(self.device)
            with torch.no_grad():
                patched_logits, _ = self._forward(
                    target_endpoints, donor_heads=donor_heads
                )
            forward_calls += 1
            example_evaluations += len(chunk)
            for index, relation in enumerate(chunk):
                donor = self.endpoints[relation.donor_endpoint_id]
                base_logits = self.native_logits[relation.target_endpoint_id]
                base_score = float(
                    base_logits[donor.answer_id] - base_logits[donor.foil_id]
                )
                patched_score = float(
                    patched_logits[index, donor.answer_id]
                    - patched_logits[index, donor.foil_id]
                )
                effect = patched_score - base_score
                if not math.isfinite(effect) or effect <= self.denominator_floor:
                    raise Task14BackendError("full-head effect is nonpositive or unsafe")
                effects.append(effect)
                self.full_head_effects[relation.ordinal] = effect

        deltas = torch.stack([
            self.native_heads[relation.donor_endpoint_id]
            - self.native_heads[relation.target_endpoint_id]
            for relation in relations
        ])
        downstream = torch.stack([
            gradients[relation.target_endpoint_id] for relation in relations
        ])
        if any(parameter.grad is not None for parameter in self.frozen_parameters):
            raise Task14BackendError("a frozen model parameter acquired a gradient")
        return program.SpectralInputs(
            ordinals=tuple(relation.ordinal for relation in relations),
            cell_keys=tuple(relation.cell_key for relation in relations),
            head_deltas=deltas,
            downstream_gradients=downstream,
            full_head_effects=torch.tensor(effects, dtype=torch.float64),
            source_partitions=("DISCOVERY",),
            validation_records_seen=0,
            validation_token_sequences_seen=0,
            model_counts={
                "forward_calls": forward_calls,
                "backward_calls": backward_calls,
                "example_evaluations": example_evaluations,
            },
        )
