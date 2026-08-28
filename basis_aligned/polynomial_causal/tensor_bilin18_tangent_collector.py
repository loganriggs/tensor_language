"""Sealed Fisher-VJP collection at bilin18 residual-write interfaces.

This module operates on the owned :class:`TensorBilin18Program`, never on native
checkpoint modules.  Zero-valued graph leaves are added to complete MLP writes at the
registered interfaces.  Because the exact writes themselves are not detached, an
MLP0 perturbation retains every indirect path through MLP1/2, attention, RMSNorm, and
the residual stream.  Only direction-projected CPU float64 responses leave the one-use
transaction; logits, leaves, targets, and raw VJPs are revoked.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from finite_horizon_tangent_response_bank import TangentResponsePlan
from tensor_bilin18_program import LAYERS, TensorBilin18Program


SOURCE_SITES = (0, 1, 2)
SCORE_START = 64
SCORE_STOP = 256
PSD_RTOL = 1e-10
SUPPORT_RTOL = 1e-12
PRODUCTION_BATCH = 4
PRODUCTION_WIDTH = 1152
PRODUCTION_VOCAB = 50_304
PRODUCTION_TOKEN_VOCAB = 50_257
RANK640_STORED_VALUES = 516_707_766


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class WriteCovarianceGeometry:
    site: int
    count: int
    mean: torch.Tensor
    covariance: torch.Tensor
    support_rank: int
    eigenvalues: torch.Tensor
    directions: torch.Tensor
    covariance_sha256: str
    directions_sha256: str
    psd_rtol: float = PSD_RTOL
    support_rtol: float = SUPPORT_RTOL


@dataclass(frozen=True)
class WriteGeometryBank:
    geometries: Mapping[int, WriteCovarianceGeometry]
    receipt: Mapping[str, Any]


def write_covariance_geometry(
    codes: torch.Tensor, *, site: int, direction_count: int = 32,
    seed: int = 2026082801,
) -> WriteCovarianceGeometry:
    """Freeze a natural-write covariance and its normalized Rademacher directions."""
    if type(site) is not int or site < 0 or type(direction_count) is not int or (
        direction_count <= 0
    ) or type(seed) is not int or seed < 0:
        raise ValueError("write geometry identifiers are malformed")
    if not torch.is_tensor(codes) or codes.ndim != 2 or codes.shape[0] < 2 or (
        codes.shape[1] <= 0
    ) or not codes.is_floating_point() or not bool(torch.isfinite(codes).all()):
        raise ValueError("write codes must be finite [sample>=2, width]")
    values = codes.detach().cpu().double().contiguous()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = ((centered.T @ centered) / (len(centered) - 1)).contiguous()
    covariance = ((covariance + covariance.T) / 2).contiguous()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = max(1.0, float(eigenvalues[-1]))
    if float(eigenvalues[0]) < -PSD_RTOL * scale:
        raise RuntimeError("write covariance is not PSD within the frozen tolerance")
    clipped = torch.clamp(eigenvalues, min=0)
    support_rank = int((clipped > SUPPORT_RTOL * scale).sum())
    if support_rank == 0:
        raise RuntimeError("write covariance has empty measured support")
    covariance = (
        eigenvectors @ torch.diag(clipped) @ eigenvectors.T
    ).contiguous()
    covariance_sqrt = (
        eigenvectors @ torch.diag(torch.sqrt(clipped)) @ eigenvectors.T
    ).contiguous()
    signs = []
    for direction in range(direction_count):
        generator = torch.Generator(device="cpu").manual_seed(
            seed + 1_000_003 * site + direction
        )
        draw = torch.randint(
            0, 2, (values.shape[1],), generator=generator, dtype=torch.int64,
        )
        signs.append((2 * draw - 1).double())
    directions = torch.stack(signs) @ covariance_sqrt
    rms = torch.sqrt(torch.mean(directions.square(), dim=1, keepdim=True))
    if bool((rms <= SUPPORT_RTOL).any()) or not bool(torch.isfinite(rms).all()):
        raise RuntimeError("a covariance-shaped direction is degenerate")
    directions = (directions / rms).contiguous()
    return WriteCovarianceGeometry(
        site=site, count=len(values), mean=mean.contiguous(), covariance=covariance,
        support_rank=support_rank, eigenvalues=clipped.flip(0).contiguous(),
        directions=directions, covariance_sha256=_tensor_sha256(covariance),
        directions_sha256=_tensor_sha256(directions),
    )


@torch.no_grad()
def collect_write_geometry_bank(
    program: TensorBilin18Program, rows: torch.Tensor, plan: TangentResponsePlan, *,
    batch_size: int = PRODUCTION_BATCH, score_start: int = SCORE_START,
    score_stop: int = SCORE_STOP, production: bool = True,
) -> WriteGeometryBank:
    """Reduce exact natural MLP writes to covariance geometry without code escape."""
    if not isinstance(program, TensorBilin18Program) or not isinstance(plan, TangentResponsePlan):
        raise TypeError("validated program and tangent plan are required")
    if type(batch_size) is not int or batch_size <= 0 or not (
        0 <= score_start < score_stop
    ):
        raise ValueError("geometry batch or score support is malformed")
    if not torch.is_tensor(rows) or rows.ndim != 2 or rows.dtype != torch.long or (
        len(rows) < 2 or rows.shape[1] < score_stop
    ):
        raise ValueError("geometry rows must be a complete integer token matrix")
    if production:
        cost = program.cost_receipt()
        if tuple(rows.shape) != (96, 513) or batch_size != PRODUCTION_BATCH or (
            score_start != SCORE_START or score_stop != SCORE_STOP
        ):
            raise ValueError("production geometry support changed")
        # The plan stores a raw-byte tensor hash without dtype/shape framing.
        raw_hash = hashlib.sha256(
            rows.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest()
        if raw_hash != plan.row_artifact_sha256:
            raise ValueError("production geometry rows differ from the frozen plan")
        if program.width != PRODUCTION_WIDTH or program.logit_vocab != PRODUCTION_VOCAB or (
            int(cost["total_stored_values"]) != RANK640_STORED_VALUES
        ) or int(cost["native_calls_per_forward"]) != 0:
            raise ValueError("production geometry requires the admitted rank640 program")
    chunks: dict[int, list[torch.Tensor]] = {site: [] for site in SOURCE_SITES}
    call_counts = {site: 0 for site in range(LAYERS)}
    device = program.token_embedding.device
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size, :score_stop].to(device).contiguous()
        program.validate_tokens(batch)
        state = F.embedding(batch, program.token_embedding)
        state = F.rms_norm(state, (program.width,))
        initial = state
        first_value = None
        for site in range(LAYERS):
            lambdas = program.residual_lambdas[site].to(state.dtype)
            state = lambdas[0] * state + lambdas[1] * initial
            attention_state = F.rms_norm(state, (program.width,))
            attention_write, first_value = program.attention_bank.programs[site](
                attention_state, first_value,
            )
            state = state + attention_write
            mlp_state = F.rms_norm(state, (program.width,))
            write = program.mlp_bank.programs[site](mlp_state)
            call_counts[site] += 1
            if site in SOURCE_SITES:
                chunks[site].append(
                    write[:, score_start:score_stop].detach().cpu().float().reshape(
                        -1, program.width,
                    ).contiguous()
                )
            state = state + write
    expected_calls = math.ceil(len(rows) / batch_size)
    if set(call_counts.values()) != {expected_calls}:
        raise RuntimeError("geometry collection did not execute every MLP site per batch")
    geometries = {}
    for site in SOURCE_SITES:
        codes = torch.cat(chunks[site], dim=0)
        expected_codes = len(rows) * (score_stop - score_start)
        if tuple(codes.shape) != (expected_codes, program.width):
            raise RuntimeError("geometry code support is incomplete")
        geometries[site] = write_covariance_geometry(
            codes, site=site, direction_count=dict(plan.input_dims)[site],
            seed=plan.direction_seed,
        )
        chunks[site].clear()
    manifest = {
        str(site): {
            "count": geometry.count,
            "support_rank": geometry.support_rank,
            "covariance_sha256": geometry.covariance_sha256,
            "directions_sha256": geometry.directions_sha256,
        }
        for site, geometry in geometries.items()
    }
    return WriteGeometryBank(
        geometries=geometries,
        receipt={
            "status": "complete",
            "plan_fingerprint": plan.fingerprint,
            "rows": len(rows),
            "score_support": [score_start, score_stop],
            "write_samples_per_site": len(rows) * (score_stop - score_start),
            "direction_rule": (
                "unit-RMS C^1/2 Rademacher; seed + 1000003*site + direction"
            ),
            "psd_rtol": PSD_RTOL,
            "support_rtol": SUPPORT_RTOL,
            "sites": manifest,
            "geometry_manifest_sha256": _json_sha256(manifest),
            "raw_write_codes_returned": False,
        },
    )


def _forward_with_additive_write_leaves(
    program: TensorBilin18Program, tokens: torch.Tensor,
    source_sites: tuple[int, ...] = SOURCE_SITES,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, Any]]:
    """Exact program forward with independent zero edits at registered MLP writes."""
    if not isinstance(program, TensorBilin18Program):
        raise TypeError("tangent collection requires an owned TensorBilin18Program")
    sites = tuple(source_sites)
    if not sites or len(set(sites)) != len(sites) or any(
        type(site) is not int or not 0 <= site < LAYERS for site in sites
    ):
        raise ValueError("source sites must be distinct valid MLP sites")
    program.validate_tokens(tokens)
    state = F.embedding(tokens, program.token_embedding)
    state = F.rms_norm(state, (program.width,))
    initial = state
    first_value = None
    leaves: dict[int, torch.Tensor] = {}
    calls = {"attention": [], "mlp": []}
    for site in range(LAYERS):
        lambdas = program.residual_lambdas[site].to(state.dtype)
        state = lambdas[0] * state + lambdas[1] * initial
        attention_state = F.rms_norm(state, (program.width,))
        attention_write, first_value = program.attention_bank.programs[site](
            attention_state, first_value,
        )
        calls["attention"].append(site)
        state = state + attention_write
        mlp_state = F.rms_norm(state, (program.width,))
        write = program.mlp_bank.programs[site](mlp_state)
        calls["mlp"].append(site)
        if site in sites:
            leaf = torch.zeros_like(write, requires_grad=True)
            leaves[site] = leaf
            state = state + write + leaf
        else:
            state = state + write
    final_state = F.rms_norm(state, (program.width,))
    logits = F.linear(final_state, program.unembedding.to(final_state.dtype))
    logits = (30.0 * torch.tanh(logits / 30.0)).float()
    if calls != {"attention": list(range(LAYERS)), "mlp": list(range(LAYERS))}:
        raise RuntimeError("tangent forward did not execute every tensor component once")
    if set(leaves) != set(sites) or any(
        tuple(leaf.shape) != (*tokens.shape, program.width) or not leaf.is_leaf
        or not leaf.requires_grad or leaf.grad_fn is not None
        for leaf in leaves.values()
    ):
        raise RuntimeError("tangent write leaves are malformed")
    return logits, leaves, {
        "attention_calls": tuple(calls["attention"]),
        "mlp_calls": tuple(calls["mlp"]),
        "source_sites": sites,
    }


def _uniform_from_identity(seed: int, row_id: str, position: int) -> float:
    digest = hashlib.sha256(f"{seed}:{row_id}:{position}".encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 0.5) / 2**64


def stateless_categorical_fisher_targets(
    logits: torch.Tensor, row_ids: Sequence[str], probe_seeds: Sequence[int], *,
    score_start: int = SCORE_START, score_stop: int = SCORE_STOP,
) -> torch.Tensor:
    """Sample full-vocabulary categorical probes independent of batch partitioning."""
    if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[2] <= 1 or (
        not bool(torch.isfinite(logits.detach()).all())
    ):
        raise ValueError("logits must be finite [batch, position, vocabulary]")
    rows, seeds = tuple(row_ids), tuple(probe_seeds)
    if len(rows) != logits.shape[0] or len(set(rows)) != len(rows) or any(
        not isinstance(row, str) or not row for row in rows
    ):
        raise ValueError("row identities must align uniquely with the batch")
    if not seeds or len(set(seeds)) != len(seeds) or any(
        type(seed) is not int or seed < 0 for seed in seeds
    ):
        raise ValueError("probe seeds must be distinct nonnegative integers")
    if not 0 <= score_start < score_stop <= logits.shape[1]:
        raise ValueError("Fisher score support is outside the logit trajectory")
    scored = logits.detach()[:, score_start:score_stop].float()
    batch, positions, vocabulary = scored.shape
    probabilities = torch.softmax(scored, dim=-1).reshape(-1, vocabulary)
    cdf = torch.cumsum(probabilities, dim=-1)
    cdf.clamp_(min=0.0, max=1.0)
    cdf[:, -1] = 1.0
    uniforms = torch.tensor([
        [
            _uniform_from_identity(seed, row, position)
            for row in rows for position in range(score_start, score_stop)
        ]
        for seed in seeds
    ], dtype=torch.float64).to(device=cdf.device, dtype=cdf.dtype)
    one = torch.tensor(1.0, device=cdf.device, dtype=cdf.dtype)
    zero = torch.tensor(0.0, device=cdf.device, dtype=cdf.dtype)
    uniforms = uniforms.clamp(min=torch.nextafter(zero, one), max=torch.nextafter(one, zero))
    targets = torch.searchsorted(cdf, uniforms.T.contiguous(), right=True).T
    expected = (len(seeds), batch * positions)
    if tuple(targets.shape) != expected or bool((targets < 0).any()) or bool(
        (targets >= vocabulary).any()
    ):
        raise RuntimeError("stateless categorical Fisher sampling failed")
    return targets.reshape(len(seeds), batch, positions).detach().cpu().long().contiguous()


@dataclass(frozen=True)
class TangentBatchResult:
    responses: Mapping[str, Mapping[int, torch.Tensor]]
    receipt: Mapping[str, Any]


class TensorBilin18TangentTransaction:
    """One-use graph transaction returning only projected response rows."""

    def __init__(
        self, *, program: TensorBilin18Program, plan: TangentResponsePlan,
        row_ids: Sequence[str], tokens: torch.Tensor,
        geometries: Mapping[int, WriteCovarianceGeometry], production: bool = True,
        injection_positions_for_test: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(program, TensorBilin18Program) or not isinstance(
            plan, TangentResponsePlan
        ):
            raise TypeError("validated program and tangent plan are required")
        rows = tuple(row_ids)
        if not rows or len(set(rows)) != len(rows) or any(row not in plan.row_ids for row in rows):
            raise ValueError("batch row identities are empty, duplicated, or unregistered")
        expected_shape = (PRODUCTION_BATCH, SCORE_STOP) if production else (
            len(rows), tokens.shape[1] if torch.is_tensor(tokens) and tokens.ndim == 2 else -1,
        )
        if not torch.is_tensor(tokens) or tuple(tokens.shape) != expected_shape or (
            tokens.dtype != torch.long or tokens.device != program.token_embedding.device
        ):
            raise ValueError("tangent tokens have the wrong shape, dtype, or device")
        if production:
            if injection_positions_for_test is not None:
                raise ValueError("production injection positions come only from the frozen plan")
            cost = program.cost_receipt()
            if program.width != PRODUCTION_WIDTH or program.logit_vocab != PRODUCTION_VOCAB or (
                program.vocab_size != PRODUCTION_VOCAB
            ) or int(cost["total_stored_values"]) != RANK640_STORED_VALUES or (
                int(cost["native_calls_per_forward"]) != 0
            ) or not bool(cost["total_input_support"]) or int(tokens.min()) < 0 or (
                int(tokens.max()) >= PRODUCTION_TOKEN_VOCAB
            ):
                raise ValueError("production tangent collection requires the admitted rank640 program")
        if not production:
            positions = tuple(injection_positions_for_test or (0,) * len(rows))
            if len(positions) != len(rows) or any(
                type(position) is not int or not 0 <= position < tokens.shape[1]
                for position in positions
            ):
                raise ValueError("test injection positions must align with tokens")
        else:
            positions = ()
        dimension_ledger = dict(plan.input_dims)
        if tuple(sorted(dimension_ledger)) != SOURCE_SITES or set(geometries) != set(SOURCE_SITES):
            raise ValueError("tangent source-site ledger changed")
        copied: dict[int, torch.Tensor] = {}
        geometry_receipt: dict[int, dict[str, Any]] = {}
        for site in SOURCE_SITES:
            geometry = geometries[site]
            if not isinstance(geometry, WriteCovarianceGeometry) or geometry.site != site:
                raise ValueError("tangent direction geometry is missing or site-mismatched")
            value = geometry.directions
            if _tensor_sha256(geometry.covariance) != geometry.covariance_sha256 or (
                _tensor_sha256(value) != geometry.directions_sha256
            ):
                raise ValueError("tangent geometry content differs from its frozen hash")
            if not torch.is_tensor(value) or tuple(value.shape) != (
                dimension_ledger[site], program.width,
            ) or value.device.type != "cpu" or value.dtype != torch.float64 or (
                value.requires_grad
            ) or not bool(torch.isfinite(value).all()):
                raise ValueError("tangent directions must be detached CPU float64 matrices")
            copied[site] = value.contiguous().clone()
            geometry_receipt[site] = {
                "write_count": geometry.count,
                "support_rank": geometry.support_rank,
                "covariance_sha256": geometry.covariance_sha256,
                "directions_sha256": geometry.directions_sha256,
                "psd_rtol": geometry.psd_rtol,
                "support_rtol": geometry.support_rtol,
            }
        self.__program: TensorBilin18Program | None = program
        self.__plan: TangentResponsePlan | None = plan
        self.__row_ids: tuple[str, ...] | None = rows
        self.__tokens: torch.Tensor | None = tokens.contiguous().clone()
        self.__tokens_sha256: str | None = _tensor_sha256(self.__tokens)
        self.__directions: dict[int, torch.Tensor] | None = copied
        self.__geometry_receipt: dict[int, dict[str, Any]] | None = geometry_receipt
        self.__production = production
        self.__test_injection_positions: tuple[int, ...] | None = (
            positions if not production else None
        )
        self.__closed = False

    @property
    def closed(self) -> bool:
        return self.__closed

    @property
    def aliases_revoked(self) -> bool:
        return self.__closed and all(getattr(self, name) is None for name in (
            "_TensorBilin18TangentTransaction__program",
            "_TensorBilin18TangentTransaction__plan",
            "_TensorBilin18TangentTransaction__row_ids",
            "_TensorBilin18TangentTransaction__tokens",
            "_TensorBilin18TangentTransaction__tokens_sha256",
            "_TensorBilin18TangentTransaction__directions",
            "_TensorBilin18TangentTransaction__geometry_receipt",
            "_TensorBilin18TangentTransaction__test_injection_positions",
        ))

    def _revoke(self) -> None:
        self.__program = None
        self.__plan = None
        self.__row_ids = None
        self.__tokens = None
        self.__tokens_sha256 = None
        self.__directions = None
        self.__geometry_receipt = None
        self.__test_injection_positions = None
        self.__closed = True

    def consume(self) -> TangentBatchResult:
        if self.__closed:
            raise RuntimeError("tangent graph transaction is spent")
        program, plan, row_ids = self.__program, self.__plan, self.__row_ids
        tokens, directions = self.__tokens, self.__directions
        tokens_sha256 = self.__tokens_sha256
        geometry_receipt = self.__geometry_receipt
        test_injection_positions = self.__test_injection_positions
        assert program is not None and plan is not None and row_ids is not None
        assert tokens is not None and tokens_sha256 is not None
        assert directions is not None and geometry_receipt is not None
        logits: torch.Tensor | None = None
        leaves: dict[int, torch.Tensor] | None = None
        targets: torch.Tensor | None = None
        try:
            if tuple(program.parameters()):
                raise RuntimeError("owned tensor program unexpectedly has trainable parameters")
            if _tensor_sha256(tokens) != tokens_sha256:
                raise RuntimeError("owned tangent tokens changed after transaction construction")
            if any(buffer.requires_grad or buffer.grad is not None for buffer in program.buffers()):
                raise RuntimeError("owned tensor program buffer gradient state changed")
            logits, leaves, forward_receipt = _forward_with_additive_write_leaves(
                program, tokens, SOURCE_SITES,
            )
            expected_logits = (*tokens.shape, program.logit_vocab)
            if tuple(logits.shape) != expected_logits or logits.dtype != torch.float32 or (
                not logits.requires_grad
            ):
                raise RuntimeError("tangent forward logits have the wrong graph contract")
            probe_seeds = tuple(plan.probe_seed + index for index in range(plan.probes_per_row))
            targets = stateless_categorical_fisher_targets(
                logits, row_ids, probe_seeds,
                score_start=SCORE_START if self.__production else 0,
                score_stop=SCORE_STOP if self.__production else tokens.shape[1],
            )
            row_ordinals = [plan.row_ids.index(row_id) for row_id in row_ids]
            injection_positions = torch.tensor([
                plan.scored_positions[index] if self.__production
                else test_injection_positions[row]
                for row, index in enumerate(row_ordinals)
            ], device=logits.device, dtype=torch.long)
            score_start = SCORE_START if self.__production else 0
            score_stop = SCORE_STOP if self.__production else tokens.shape[1]
            absolute = torch.arange(score_start, score_stop, device=logits.device)
            causal_mask = absolute.unsqueeze(0) >= injection_positions.unsqueeze(1)
            log_probabilities = F.log_softmax(
                logits[:, score_start:score_stop].float(), dim=-1,
            )
            target_device = targets.to(logits.device)
            projected = {
                site: torch.empty(
                    plan.probes_per_row, len(row_ids), dict(plan.input_dims)[site],
                    dtype=torch.float64,
                )
                for site in SOURCE_SITES
            }
            row_index = torch.arange(len(row_ids), device=logits.device)
            for probe in range(plan.probes_per_row):
                selected = torch.gather(
                    log_probabilities, -1, target_device[probe].unsqueeze(-1),
                ).squeeze(-1)
                objective = (selected * causal_mask).sum()
                gradients = torch.autograd.grad(
                    objective, tuple(leaves[site] for site in SOURCE_SITES),
                    retain_graph=probe + 1 < plan.probes_per_row,
                    create_graph=False, allow_unused=False,
                )
                for site, gradient in zip(SOURCE_SITES, gradients, strict=True):
                    chosen = gradient[row_index, injection_positions].detach().cpu().double()
                    if tuple(chosen.shape) != (len(row_ids), program.width) or not bool(
                        torch.isfinite(chosen).all()
                    ):
                        raise RuntimeError("tangent VJP row has a malformed shape or value")
                    projected[site][probe] = chosen @ directions[site].T
            response_rows = {
                row_id: {
                    site: projected[site][:, row].contiguous().clone()
                    for site in SOURCE_SITES
                }
                for row, row_id in enumerate(row_ids)
            }
            target_hash = _tensor_sha256(targets)
            response_hashes = {
                row_id: {str(site): _tensor_sha256(value) for site, value in site_rows.items()}
                for row_id, site_rows in response_rows.items()
            }
            receipt_arguments = {
                "status": "complete",
                "plan_fingerprint": plan.fingerprint,
                "row_ids": list(row_ids),
                "tokens_sha256": tokens_sha256,
                "source_sites": list(SOURCE_SITES),
                "probe_seeds": list(probe_seeds),
                "target_ids_sha256": target_hash,
                "response_sha256": _json_sha256(response_hashes),
                "direction_geometry": {
                    str(site): geometry_receipt[site] for site in SOURCE_SITES
                },
                "logit_shape": list(logits.shape),
                "logit_dtype": str(logits.dtype),
                "full_vocabulary": logits.shape[-1] == program.logit_vocab,
                "future_output_mask": True,
                "forward": forward_receipt,
                "program_parameters_untouched": not tuple(program.parameters()),
                "program_buffer_gradients_absent": all(
                    not buffer.requires_grad and buffer.grad is None
                    for buffer in program.buffers()
                ),
            }
        finally:
            logits = None
            leaves = None
            targets = None
            self._revoke()
        receipt_arguments["graph_aliases_revoked"] = self.aliases_revoked
        return TangentBatchResult(responses=response_rows, receipt=receipt_arguments)
