"""Narrow tensor backend primitives for Task 14 projector Program A.

The production model forward and the fit objective are injected.  This keeps
the DISCOVERY shard, differentiable head hook, and orthogonal optimizer
testable on CPU without importing a checkpoint or choosing scientific loss
constants in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.nn.utils import parametrizations

import run_task14_head11_3_projector_discovery as program
import task14_head11_3_projector_adapter as adapter


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


def _objective_signature(objective: object) -> tuple[object, ...]:
    """Compare frozen objective values across enqueue-snapshot module identities."""

    return tuple(
        getattr(objective, name)
        for name in program.FIT_OBJECTIVE.__dataclass_fields__
    )


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
        denominator_floor: float = program.FIT_OBJECTIVE.denominator_floor,
        enforce_production_contract: bool = True,
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
        self.enforce_production_contract = enforce_production_contract
        self.frozen_parameters = tuple(frozen_parameters)
        self._parameter_versions = tuple(
            parameter._version for parameter in self.frozen_parameters
        )
        self._checkpoint_tensor_sha256 = self._live_parameter_sha256()
        self.native_heads: dict[str, torch.Tensor] = {}
        self.native_logits: dict[str, torch.Tensor] = {}
        self.full_head_effects: dict[int, float] = {}
        self.full_head_logits: dict[int, torch.Tensor] = {}
        self.fit_control_normalizer: float | None = None
        self.replay_rank0_exact: bool | None = None
        self.replay_rank128_exact: bool | None = None

    def _live_parameter_sha256(self) -> str:
        """Hash the live frozen tensors without materializing one giant buffer."""
        digest = hashlib.sha256()
        for index, parameter in enumerate(self.frozen_parameters):
            header = json.dumps({
                "index": index, "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
            }, sort_keys=True, separators=(",", ":")).encode("ascii")
            digest.update(header)
            digest.update(b"\0")
            # Reshape before the dtype reinterpretation: the model contains
            # scalar parameters, and PyTorch cannot view a rank-0 float as bytes.
            raw = parameter.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
            digest.update(memoryview(raw))
        return digest.hexdigest()

    @classmethod
    def load_production(
        cls, *, device: str | torch.device = "cuda"
    ) -> "Task14ProgramATorchBackend":
        """Load the pinned frozen model lazily and expose its exact tensor forward."""

        facade_path = OPS.parent.parent / "polynomial_causal/bilin18_observed_model_facade.py"
        spec = importlib.util.spec_from_file_location(
            "task14_bilin18_observed_model_facade", facade_path
        )
        if spec is None or spec.loader is None:
            raise Task14BackendError("cannot load the pinned bilin18 facade")
        facade = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = facade
        spec.loader.exec_module(facade)
        model, receipt = facade.load_bilin18(device=device, dtype=torch.float32)
        if receipt.weights_sha256 != facade.WEIGHTS_SHA256:
            raise Task14BackendError("production checkpoint hash changed")

        def forward_logits(tokens: torch.Tensor, _lengths: tuple[int, ...]):
            def attention(event):
                return event.block.attn(event.state, event.first_value)

            def mlp(event):
                return event.block.mlp(event.state)

            return facade.forward_with_dispatch(
                model, tokens, attention, mlp, require_production=False
            )

        return cls(
            forward_logits=forward_logits,
            c_proj_module=model.transformer.h[11].attn.c_proj,
            device=device,
            frozen_parameters=tuple(model.parameters()),
        )

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
            local_frame = None if frame is None else frame.to(
                device=value.device, dtype=value.dtype
            )
            patched = donor if local_frame is None else adapter.projected_head_interchange(
                recipient, donor, local_frame, validate=False
            )
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
        if any(
            relation.target_endpoint_id not in self.endpoints
            or relation.donor_endpoint_id not in self.endpoints
            for relation in relations
        ):
            raise Task14BackendError("relation escaped the DISCOVERY shard")
        # Load all 64 FIT endpoints in the same two calls.  This lets later
        # control fitting share the endpoint cache without reopening authority.
        needed_ids = sorted(
            endpoint_id for endpoint_id, endpoint in self.endpoints.items()
            if endpoint.cell_metadata[1] in program.FIT_GROUP_NUMBERS
        )
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
                self.full_head_logits[relation.ordinal] = patched_logits[index].detach().cpu()

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
        result = program.SpectralInputs(
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
        if self.enforce_production_contract:
            expected_counts = {
                "forward_calls": 6,
                "backward_calls": 2,
                "example_evaluations": 180,
            }
            self._check_counts(result.model_counts or {}, maximum=expected_counts)
            if result.model_counts != expected_counts:
                raise Task14BackendError("spectral collection count changed")
        return result

    def _relation_pair(self, relation: program.Relation) -> tuple[int, int]:
        endpoint_id = (
            relation.donor_endpoint_id if relation.role == "target"
            else relation.target_endpoint_id
        )
        endpoint = self.endpoints[endpoint_id]
        return endpoint.answer_id, endpoint.foil_id

    def _cache_native_missing(
        self, relations: Sequence[program.Relation]
    ) -> dict[str, int]:
        endpoint_ids = sorted({
            endpoint_id for relation in relations for endpoint_id in (
                relation.target_endpoint_id, relation.donor_endpoint_id
            ) if endpoint_id not in self.native_heads
        })
        try:
            endpoints = [self.endpoints[endpoint_id] for endpoint_id in endpoint_ids]
        except KeyError as error:
            raise Task14BackendError("relation escaped the DISCOVERY shard") from error
        counts = {"forward_calls": 0, "backward_calls": 0,
                  "example_evaluations": 0}
        for values in self._chunks(endpoints):
            chunk = tuple(values)
            with torch.no_grad():
                logits, activation = self._forward(chunk)
            counts["forward_calls"] += 1
            counts["example_evaluations"] += len(chunk)
            for index, endpoint in enumerate(chunk):
                position = endpoint.final_position
                self.native_heads[endpoint.endpoint_id] = activation[
                    index, position, HEAD_START:HEAD_STOP
                ].detach().cpu().to(torch.float64)
                self.native_logits[endpoint.endpoint_id] = logits[index].detach().cpu()
        return counts

    def _cache_full_head_missing(
        self, relations: Sequence[program.Relation]
    ) -> dict[str, int]:
        missing = tuple(
            relation for relation in relations
            if relation.ordinal not in self.full_head_effects
        )
        counts = {"forward_calls": 0, "backward_calls": 0,
                  "example_evaluations": 0}
        # The corrected price treats the 37 inference-only FIT controls as one
        # shared cache call.  The 32-row rule governs stochastic fit updates.
        chunks = (missing,) if len(missing) == 37 and all(
            relation.role == "control" for relation in missing
        ) else self._chunks(missing)
        for values in chunks:
            chunk = tuple(values)
            targets = tuple(
                self.endpoints[relation.target_endpoint_id] for relation in chunk
            )
            donors = torch.stack([
                self.native_heads[relation.donor_endpoint_id] for relation in chunk
            ]).to(self.device)
            with torch.no_grad():
                logits, _ = self._forward(targets, donor_heads=donors)
            counts["forward_calls"] += 1
            counts["example_evaluations"] += len(chunk)
            for index, relation in enumerate(chunk):
                answer, foil = self._relation_pair(relation)
                base = self.native_logits[relation.target_endpoint_id]
                effect = float(logits[index, answer] - logits[index, foil]) - float(
                    base[answer] - base[foil]
                )
                if not math.isfinite(effect) or (
                    relation.role == "target" and effect <= self.denominator_floor
                ):
                    raise Task14BackendError("cached full-head effect is invalid")
                self.full_head_effects[relation.ordinal] = effect
                self.full_head_logits[relation.ordinal] = logits[index].detach().cpu()
        return counts

    @staticmethod
    def _add_counts(left: dict[str, int], right: Mapping[str, int]) -> None:
        for key in left:
            left[key] += int(right.get(key, 0))

    @staticmethod
    def _check_counts(
        counts: Mapping[str, int], *, maximum: Mapping[str, int]
    ) -> None:
        for key in ("forward_calls", "backward_calls", "example_evaluations"):
            value = counts.get(key)
            if type(value) is not int or value < 0 or value > maximum[key]:
                raise Task14BackendError(f"model count {key} is invalid or exceeds ceiling")

    @staticmethod
    def _validate_relation_cells(
        relations: Sequence[program.Relation], *, split: str
    ) -> None:
        expected = {
            "FIT": (153, 116, 37),
            "SELECT": (145, 106, 39),
        }.get(split)
        if expected is None:
            raise Task14BackendError("unknown relation split")
        target_count = sum(row.role == "target" for row in relations)
        control_count = sum(row.role == "control" for row in relations)
        target_cells = {row.cell_key for row in relations if row.role == "target"}
        control_cells = {row.cell_key for row in relations if row.role == "control"}
        coordinated = {
            cell for cell in target_cells
            if cell.startswith("C_to_ordinary_singular|C|")
        }
        if (
            (len(relations), target_count, control_count) != expected
            or len(target_cells) != 24
            or len(control_cells) != 7
            or target_cells & control_cells
            or len(coordinated) != 2
            or len({row.ordinal for row in relations}) != len(relations)
            or len({row.record_id for row in relations}) != len(relations)
        ):
            raise Task14BackendError(
                f"{split} relation/cell coverage differs from the frozen contract"
            )

    def _ensure_baselines(
        self, relations: Sequence[program.Relation]
    ) -> dict[str, int]:
        counts = {"forward_calls": 0, "backward_calls": 0,
                  "example_evaluations": 0}
        self._add_counts(counts, self._cache_native_missing(relations))
        self._add_counts(counts, self._cache_full_head_missing(relations))
        return counts

    def _ensure_endpoint_replays(
        self, select_relations: Sequence[program.Relation]
    ) -> dict[str, int]:
        """Measure the registered exact rank-0 and rank-128 endpoints once."""
        counts = {"forward_calls": 0, "backward_calls": 0,
                  "example_evaluations": 0}
        if self.replay_rank0_exact is not None or self.replay_rank128_exact is not None:
            if self.replay_rank0_exact is None or self.replay_rank128_exact is None:
                raise Task14BackendError("endpoint replay cache is partially initialized")
            return counts
        select_relations = tuple(select_relations)
        if self.enforce_production_contract:
            self._validate_relation_cells(select_relations, split="SELECT")
        if any(
            relation.ordinal not in self.full_head_logits
            or relation.target_endpoint_id not in self.native_logits
            for relation in select_relations
        ):
            raise Task14BackendError("endpoint replay requires complete SELECT caches")

        select_endpoint_ids = sorted({
            endpoint_id for relation in select_relations for endpoint_id in (
                relation.target_endpoint_id, relation.donor_endpoint_id
            )
        })
        if self.enforce_production_contract and len(select_endpoint_ids) != 64:
            raise Task14BackendError("rank-0 replay endpoint census changed")
        rank0_exact = True
        empty = torch.empty(HEAD_WIDTH, 0, device=self.device, dtype=torch.float32)
        for values in self._chunks(select_endpoint_ids):
            endpoint_ids = tuple(values)
            endpoints = tuple(self.endpoints[endpoint_id] for endpoint_id in endpoint_ids)
            donors = torch.stack([
                self.native_heads[endpoint_id] for endpoint_id in endpoint_ids
            ]).to(self.device)
            with torch.no_grad():
                logits, _ = self._forward(endpoints, donor_heads=donors, frame=empty)
            counts["forward_calls"] += 1
            counts["example_evaluations"] += len(endpoints)
            rank0_exact = rank0_exact and all(
                torch.equal(logits[index].detach().cpu(), self.native_logits[endpoint_id])
                for index, endpoint_id in enumerate(endpoint_ids)
            )

        rank128_exact = True
        identity = torch.eye(HEAD_WIDTH, device=self.device, dtype=torch.float32)
        for values in self._chunks(select_relations):
            chunk = tuple(values)
            endpoints = tuple(
                self.endpoints[relation.target_endpoint_id] for relation in chunk
            )
            donors = torch.stack([
                self.native_heads[relation.donor_endpoint_id] for relation in chunk
            ]).to(self.device)
            with torch.no_grad():
                logits, _ = self._forward(endpoints, donor_heads=donors, frame=identity)
            counts["forward_calls"] += 1
            counts["example_evaluations"] += len(chunk)
            rank128_exact = rank128_exact and all(
                torch.equal(
                    logits[index].detach().cpu(),
                    self.full_head_logits[relation.ordinal],
                ) for index, relation in enumerate(chunk)
            )

        self.replay_rank0_exact = rank0_exact
        self.replay_rank128_exact = rank128_exact
        if self.enforce_production_contract and counts != {
            "forward_calls": 7, "backward_calls": 0,
            "example_evaluations": 209,
        }:
            raise Task14BackendError("endpoint replay price changed")
        return counts

    @staticmethod
    def _permutation_labels(
        relations: Sequence[program.Relation], permutation_id: int | None
    ) -> dict[int, float]:
        if permutation_id is None:
            return {relation.ordinal: 1.0 for relation in relations if relation.role == "target"}
        if permutation_id not in (0, 1):
            raise Task14BackendError("permutation ID must be 0, 1, or None")
        output: dict[int, float] = {}
        cells = sorted({r.cell_key for r in relations if r.role == "target"})
        for cell in cells:
            rows = [r for r in relations if r.role == "target" and r.cell_key == cell]
            rows.sort(key=lambda row: hashlib.sha256(
                f"task14-head11.3-permutation|{permutation_id}|{row.record_id}".encode()
            ).digest())
            if permutation_id == 1:
                rows.reverse()
            positive = (len(rows) + 1) // 2
            for index, row in enumerate(rows):
                output[row.ordinal] = 1.0 if index < positive else -1.0
        return output

    @staticmethod
    def _training_schedule(
        relations: Sequence[program.Relation], *, rank: int, start: int,
        permutation_id: int | None, updates: int,
        objective: program.FitObjectiveConfig,
    ) -> tuple[tuple[program.Relation, ...], ...]:
        targets: dict[str, list[program.Relation]] = {}
        controls: dict[str, list[program.Relation]] = {}
        for relation in relations:
            (targets if relation.role == "target" else controls).setdefault(
                relation.cell_key, []
            ).append(relation)
        if not targets or not controls:
            raise Task14BackendError("FIT relations lack target or control cells")
        p = 0 if permutation_id is None else 1 + permutation_id
        generator = torch.Generator(device="cpu").manual_seed(
            objective.schedule_seed_base + 100 * rank + start + 10000 * p
        )

        def draw(groups: Mapping[str, Sequence[program.Relation]], count: int):
            keys = sorted(groups)
            chosen = []
            for _ in range(count):
                key = keys[int(torch.randint(len(keys), (), generator=generator))]
                rows = groups[key]
                chosen.append(rows[int(torch.randint(len(rows), (), generator=generator))])
            return chosen

        return tuple(tuple(
            draw(targets, objective.target_draws_per_update)
            + draw(controls, objective.control_draws_per_update)
        ) for _ in range(updates))

    def _projected_logits(
        self, relations: Sequence[program.Relation], frame: torch.Tensor
    ) -> torch.Tensor:
        targets = tuple(self.endpoints[r.target_endpoint_id] for r in relations)
        donors = torch.stack([
            self.native_heads[r.donor_endpoint_id] for r in relations
        ]).to(self.device)
        logits, _ = self._forward(targets, donor_heads=donors, frame=frame)
        return logits

    def _score_frame(
        self, relations: Sequence[program.Relation], frame: torch.Tensor
    ) -> tuple[
        dict[str, program.TargetCellScore],
        dict[str, program.ControlCellScore],
        tuple[float, ...],
        dict[str, int],
    ]:
        rows: dict[int, tuple[float, float, float]] = {}
        vocab_rms: dict[int, float] = {}
        counts = {"forward_calls": 0, "backward_calls": 0,
                  "example_evaluations": 0}
        tau = self.fit_control_normalizer
        if tau is None or not math.isfinite(tau) or tau <= self.denominator_floor:
            raise Task14BackendError("FIT target control normalizer is invalid")
        for values in self._chunks(tuple(relations)):
            chunk = tuple(values)
            with torch.no_grad():
                logits = self._projected_logits(chunk, frame)
            counts["forward_calls"] += 1
            counts["example_evaluations"] += len(chunk)
            for index, relation in enumerate(chunk):
                answer, foil = self._relation_pair(relation)
                base = self.native_logits[relation.target_endpoint_id]
                effect = float(logits[index, answer] - logits[index, foil]) - float(
                    base[answer] - base[foil]
                )
                donor = self.native_logits[relation.donor_endpoint_id]
                native_gap = float(donor[answer] - donor[foil]) - float(
                    base[answer] - base[foil]
                )
                rows[relation.ordinal] = (
                    effect, self.full_head_effects[relation.ordinal], native_gap
                )
                if relation.role == "control":
                    difference = logits[index].detach().cpu().to(torch.float64) \
                        - base.to(torch.float64)
                    if difference.numel() != program.FIT_OBJECTIVE.full_vocabulary_size:
                        raise Task14BackendError("full-vocabulary output width changed")
                    vocab_rms[relation.ordinal] = float(
                        torch.sqrt(torch.mean(difference.square())) / tau
                    )

        target_scores: dict[str, program.TargetCellScore] = {}
        control_scores: dict[str, program.ControlCellScore] = {}
        for cell in sorted({relation.cell_key for relation in relations}):
            members = [relation for relation in relations if relation.cell_key == cell]
            values = [rows[relation.ordinal] for relation in members]
            if members[0].role == "target":
                effects = [value[0] for value in values]
                full = [value[1] for value in values]
                gaps = [value[2] for value in values]
                if sum(full) <= self.denominator_floor or any(
                    gap <= self.denominator_floor for gap in gaps
                ):
                    raise Task14BackendError("target cell denominator is invalid")
                target_scores[cell] = program.TargetCellScore(
                    full_head_fraction=sum(effects) / sum(full),
                    direction_fraction=sum(value > 0 for value in effects) / len(effects),
                    native_donor_recovery=sum(effects) / sum(gaps),
                    coordinated_subject_cell=cell.startswith("C_to_ordinary_singular|"),
                )
            else:
                control_scores[cell] = program.ControlCellScore(
                    normalized_margin_movement=sum(abs(value[0]) for value in values)
                    / (len(values) * tau),
                    full_head_normalized_movement=sum(abs(value[1]) for value in values)
                    / (len(values) * tau),
                    normalized_full_vocabulary_rms=sum(
                        vocab_rms[relation.ordinal] for relation in members
                    ) / len(members),
                )
        normalized = tuple(
            rows[relation.ordinal][0] / rows[relation.ordinal][1]
            for relation in relations if relation.role == "target"
        )
        if self.enforce_production_contract and len(normalized) != 106:
            raise Task14BackendError("SELECT target effect order/count changed")
        return target_scores, control_scores, normalized, counts

    def fit_and_score(
        self,
        *,
        fit_relations: Sequence[program.Relation],
        select_relations: Sequence[program.Relation],
        rank: int,
        start: int,
        initial_frame: torch.Tensor,
        updates: int,
        batch_size: int,
        objective: program.FitObjectiveConfig,
        permutation_id: int | None,
    ) -> program.FitResult:
        # The managed runner executes a byte snapshot as ``__main__``. Its
        # frozen dataclass therefore has a different Python class identity from
        # the otherwise identical module imported here; compare registered
        # values, not dataclass identity.
        if _objective_signature(objective) != _objective_signature(program.FIT_OBJECTIVE) \
                or updates != program.UPDATES \
                or batch_size != program.BATCH_SIZE or rank != initial_frame.shape[1] \
                or self.denominator_floor != objective.denominator_floor:
            raise Task14BackendError("fit configuration differs from the frozen addendum")
        fit_relations, select_relations = tuple(fit_relations), tuple(select_relations)
        if self.enforce_production_contract:
            self._validate_relation_cells(fit_relations, split="FIT")
            self._validate_relation_cells(select_relations, split="SELECT")
        counts = self._ensure_baselines(fit_relations)
        # Preserve the SELECT-only batch layout used by the exact rank-128
        # replay. Mixing the final FIT-control chunk into SELECT would make a
        # bitwise comparison depend on batch shape rather than intervention semantics.
        self._add_counts(counts, self._ensure_baselines(select_relations))
        self._add_counts(counts, self._ensure_endpoint_replays(select_relations))
        positive_fit_effects = [
            self.full_head_effects[r.ordinal] for r in fit_relations
            if r.role == "target"
            and self.full_head_effects[r.ordinal] > objective.denominator_floor
        ]
        if len(positive_fit_effects) != sum(r.role == "target" for r in fit_relations):
            raise Task14BackendError("FIT target full-head denominator is invalid")
        tau = float(torch.median(torch.tensor(positive_fit_effects, dtype=torch.float64)))
        self.fit_control_normalizer = tau
        labels = self._permutation_labels(fit_relations, permutation_id)
        schedule = self._training_schedule(
            fit_relations, rank=rank, start=start, permutation_id=permutation_id,
            updates=updates, objective=objective,
        )

        holder = nn.Linear(rank, HEAD_WIDTH, bias=False, device=self.device,
                           dtype=torch.float32)
        with torch.no_grad():
            holder.weight.copy_(initial_frame.to(self.device, dtype=torch.float32))
        parametrizations.orthogonal(
            holder, "weight", orthogonal_map="householder", use_trivialization=True
        )
        optimizer = torch.optim.Adam(
            holder.parametrizations.weight.parameters(), lr=objective.learning_rate,
            betas=(objective.adam_beta1, objective.adam_beta2), eps=objective.adam_epsilon,
            weight_decay=objective.weight_decay,
        )
        losses: list[float] = []
        gradients_finite = True
        for update, batch in enumerate(schedule):
            rate = objective.learning_rate * (
                1.0 + math.cos(math.pi * update / (updates - 1))
            ) / 2.0
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            logits = self._projected_logits(batch, holder.weight)
            target_losses, control_losses = [], []
            for index, relation in enumerate(batch):
                answer, foil = self._relation_pair(relation)
                base = self.native_logits[relation.target_endpoint_id]
                effect = logits[index, answer] - logits[index, foil] - float(
                    base[answer] - base[foil]
                )
                if relation.role == "target":
                    residual = effect / self.full_head_effects[relation.ordinal] \
                        - labels[relation.ordinal]
                    absolute = torch.abs(residual)
                    delta = objective.huber_transition
                    target_losses.append(torch.where(
                        absolute <= delta, 0.5 * residual.square(),
                        delta * (absolute - 0.5 * delta),
                    ))
                else:
                    control_losses.append((effect / tau).square())
            loss = objective.target_coefficient * torch.stack(target_losses).mean() \
                + objective.control_coefficient * torch.stack(control_losses).mean()
            if not bool(torch.isfinite(loss.detach())):
                raise Task14BackendError("FIT objective became nonfinite")
            loss.backward()
            parameters = tuple(holder.parametrizations.weight.parameters())
            if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                   for parameter in parameters):
                gradients_finite = False
                raise Task14BackendError("Householder fit gradient is invalid")
            optimizer.step()
            losses.append(float(loss.detach()))
            counts["forward_calls"] += 1
            counts["backward_calls"] += 1
            counts["example_evaluations"] += len(batch)

        frame = holder.weight.detach().cpu().to(torch.float64)
        scored = self._score_frame(select_relations, frame.to(self.device, torch.float32))
        target_cells, control_cells, normalized, score_counts = scored
        self._add_counts(counts, score_counts)
        self._check_counts(
            counts,
            maximum={
                "forward_calls": 120,
                "backward_calls": 100,
                "example_evaluations": 3800,
            },
        )
        identity = torch.eye(rank, dtype=torch.float64)
        orth_error = float(torch.max(torch.abs(frame.T @ frame - identity)))
        initial = initial_frame.detach().cpu().to(torch.float64)
        movement = float(torch.linalg.matrix_norm(
            frame @ frame.T - initial @ initial.T
        ) / math.sqrt(2 * rank))
        model_clean = all(parameter.grad is None for parameter in self.frozen_parameters)
        versions_clean = tuple(p._version for p in self.frozen_parameters) \
            == self._parameter_versions
        checkpoint_clean = self._live_parameter_sha256() == self._checkpoint_tensor_sha256
        return program.FitResult(
            rank=rank, start=start, frame=frame,
            health=program.FitHealth(
                finite=all(math.isfinite(value) for value in losses),
                model_parameter_gradients_absent=model_clean,
                checkpoint_hash_unchanged=versions_clean and checkpoint_clean,
                hook_exact=True,
                replay_rank0_exact=bool(self.replay_rank0_exact),
                replay_rank128_exact=bool(self.replay_rank128_exact),
                orthonormality_error=orth_error,
                normalized_projector_movement=movement,
                first20_objective=sum(losses[:20]) / len(losses[:20]),
                final20_objective=sum(losses[-20:]) / len(losses[-20:]),
                schedule_updates=len(losses),
            ),
            target_cells=target_cells, control_cells=control_cells,
            normalized_row_effects=normalized, model_counts=counts,
            scored_ordinals=tuple(r.ordinal for r in select_relations),
            normalized_row_effect_ordinals=tuple(
                r.ordinal for r in select_relations if r.role == "target"
            ),
        )

    def score_fixed_frame(
        self, *, select_relations: Sequence[program.Relation],
        frame: torch.Tensor, control_id: str,
    ) -> program.FitResult:
        if not isinstance(control_id, str) or not control_id:
            raise Task14BackendError("fixed-frame control ID is empty")
        select_relations = tuple(select_relations)
        if self.enforce_production_contract:
            self._validate_relation_cells(select_relations, split="SELECT")
        counts = self._ensure_baselines(select_relations)
        frame = frame.detach().cpu().to(torch.float64)
        rank = frame.shape[1]
        identity = torch.eye(rank, dtype=torch.float64)
        if tuple(frame.shape) != (HEAD_WIDTH, rank) or not torch.allclose(
            frame.T @ frame, identity, atol=1e-5, rtol=0
        ):
            raise Task14BackendError("fixed frame is malformed")
        target_cells, control_cells, normalized, score_counts = self._score_frame(
            select_relations, frame.to(self.device, torch.float32)
        )
        self._add_counts(counts, score_counts)
        self._check_counts(
            counts,
            maximum={
                "forward_calls": 5,
                "backward_calls": 0,
                "example_evaluations": 145,
            },
        )
        return program.FitResult(
            rank=rank, start=-1, frame=frame,
            health=program.FitHealth(
                True, True, True, True,
                bool(self.replay_rank0_exact), bool(self.replay_rank128_exact), 0.0, 0.0,
                math.nan, math.nan, 0,
            ),
            target_cells=target_cells, control_cells=control_cells,
            normalized_row_effects=normalized, model_counts=counts,
            scored_ordinals=tuple(r.ordinal for r in select_relations),
            normalized_row_effect_ordinals=tuple(
                r.ordinal for r in select_relations if r.role == "target"
            ),
        )
