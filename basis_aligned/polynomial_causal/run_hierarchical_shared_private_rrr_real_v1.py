#!/usr/bin/env python3
"""Source-closed real discovery runner for hierarchical shared/private RRR.

This module is import-pure.  Production reuses the audited shared-output RRR v2
capture, observed-model facade, CE ledger, and receipt-last transaction after replacing
only the prospectively frozen arm bank, fitter, autonomous adapter, and semantic gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/workspace/tensor_language")
HERE = ROOT / "basis_aligned" / "polynomial_causal"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import hierarchical_shared_private_rrr as hybrid
import run_shared_output_rrr_real_v1 as base


RUNNER = HERE / "run_hierarchical_shared_private_rrr_real_v1.py"
TEST = HERE / "test_run_hierarchical_shared_private_rrr_real_v1.py"
ADDENDUM = HERE / "HIERARCHICAL_SHARED_PRIVATE_RRR_REAL_V1_EXECUTION_ADDENDUM.md"
MATH_PREREG = HERE / "HIERARCHICAL_SHARED_PRIVATE_RRR_V1_PREREGISTRATION.md"
MATH_CORE = HERE / "hierarchical_shared_private_rrr.py"
MATH_TEST = HERE / "test_hierarchical_shared_private_rrr.py"

AUTHORITY = HERE / "hierarchical_shared_private_rrr_real_v1_authority.json"
RESULTS = HERE / "hierarchical_shared_private_rrr_real_v1_results.json"
FAILURE = HERE / "hierarchical_shared_private_rrr_real_v1_failure.json"
RECEIPT = HERE / "hierarchical_shared_private_rrr_real_v1_receipt.json"
LOCK = Path("/workspace/runs/.hierarchical_shared_private_rrr_real_v1.lock")
PROTOCOL_VERSION = "hierarchical_shared_private_v1"

PARENT_AUTHORITY = HERE / "shared_output_rrr_real_v2_recovery_authority.json"
PARENT_RESULTS = HERE / "shared_output_rrr_real_v2_recovery_results.json"
PARENT_RECEIPT = HERE / "shared_output_rrr_real_v2_recovery_receipt.json"
PARENT_FAILURE = HERE / "shared_output_rrr_real_v2_recovery_failure.json"
PARENT_HASHES = {
    str(PARENT_AUTHORITY.relative_to(ROOT)):
        "32106d80f43bd73853ca25841b81ea1297d0e9c5d2bf6f404510a1c1b217c7db",
    str(PARENT_RESULTS.relative_to(ROOT)):
        "19d65e2c6d4a0cff19ddfb76ddbe62dcd26c462a695e006c457da85a89adc053",
    str(PARENT_RECEIPT.relative_to(ROOT)):
        "57f699d680a7ea010f6ec8b12c3c33d61f1b3f540ad2891517fe751074dbdd56",
}

D = base.D
N_SITES = base.N_SITES
COMMON_TABLE_FLOATS = base.COMMON_TABLE_FLOATS
BUDGETS = hybrid.canonical_bilin18_storage_points()["budgets"]
GRID = {
    "global_q512": (0, 128, 512),
    "typed_q512": (0, 128),
    "independent_q512": (0, 128),
}
N_ARMS = sum(len(values) for values in GRID.values())

SOURCE_PATHS = tuple(dict.fromkeys((
    *base.SOURCE_PATHS,
    *(str(path.relative_to(ROOT)) for path in (
        MATH_PREREG, MATH_CORE, MATH_TEST, ADDENDUM, RUNNER, TEST,
    )),
)))
FILE_PINS = {**base.FILE_PINS, **PARENT_HASHES}


def arm_name(budget: str, q0: int) -> str:
    return f"{budget}_shared_q{q0}"


def arm_descriptors() -> tuple[dict[str, Any], ...]:
    # q0-major ordering permits one residual eigensystem to serve every budget and
    # then be discarded before the next q0, bounding CPU memory.
    output = []
    for q0 in (0, 128, 512):
        for budget, ranks in GRID.items():
            if q0 in ranks:
                output.append({
                    "name": arm_name(budget, q0),
                    "family": "hierarchical_shared_private",
                    "budget_name": budget,
                    "map_float_budget": BUDGETS[budget],
                    "shared_rank": q0,
                })
    return tuple(output)


def expected_call_schedule() -> dict[str, Any]:
    fit = base.FIT_OUTER_CALLS
    evaluation = base.EVAL_CALLS_PER_ARM
    return {
        "fit_native_outer": fit,
        "native_reference_outer": evaluation,
        "compiled_outer_per_arm": evaluation,
        "compiled_arm_count": N_ARMS,
        "compiled_outer_total": N_ARMS * evaluation,
        "outer_total": fit + (N_ARMS + 1) * evaluation,
        "native_component_calls_per_kind": 18 * (fit + evaluation),
        "compiled_component_calls": N_ARMS * evaluation * N_SITES,
        "optimizer_calls": 0,
        "backward_calls": 0,
    }


def verify_parent() -> dict[str, Any]:
    observed = {relative: base.file_sha256(ROOT / relative) for relative in PARENT_HASHES}
    if observed != PARENT_HASHES or PARENT_FAILURE.exists():
        raise RuntimeError("hierarchical RRR parent byte closure changed")
    authority = json.loads(PARENT_AUTHORITY.read_text())
    result = json.loads(PARENT_RESULTS.read_text())
    receipt = json.loads(PARENT_RECEIPT.read_text())
    if receipt.get("authority_file_sha256") != PARENT_HASHES[str(
        PARENT_AUTHORITY.relative_to(ROOT)
    )] or receipt.get("results_file_sha256") != PARENT_HASHES[str(
        PARENT_RESULTS.relative_to(ROOT)
    )] or receipt.get("failure_absent") is not True or receipt.get(
        "authority_scope"
    ) != "discovery_only_no_validation_final_or_semantic_coordinates":
        raise RuntimeError("hierarchical RRR parent receipt join changed")
    if result.get("authority_sha256") != authority.get("authority_sha256") or result.get(
        "status"
    ) != "discovery_complete_no_validation_or_generalization_authority":
        raise RuntimeError("hierarchical RRR parent semantic join changed")
    required = {
        "global_q512", "price_global_q512", "typed_q512", "price_typed_q512",
        "independent_q512",
    }
    if not required.issubset(result.get("arms", {})):
        raise RuntimeError("hierarchical RRR parent comparator bank changed")
    return {
        "authority_file_sha256": observed[str(PARENT_AUTHORITY.relative_to(ROOT))],
        "authority_sha256": authority["authority_sha256"],
        "results_file_sha256": observed[str(PARENT_RESULTS.relative_to(ROOT))],
        "receipt_file_sha256": observed[str(PARENT_RECEIPT.relative_to(ROOT))],
        "receipt_sha256": receipt["receipt_sha256"],
        "authority_scope": receipt["authority_scope"],
    }


def authority_payload(
    source: Mapping[str, Any], inputs: Mapping[str, str], checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    parent = verify_parent()
    protocol = {
        "math_preregistration": str(MATH_PREREG),
        "execution_addendum": str(ADDENDUM),
        "parent": parent,
        "fit_path": str(base.FIT_PATH),
        "roles": {key: str(value) for key, value in base.ROLE_PATHS.items()},
        "row_truncation": 257,
        "score_positions": [base.SCORE_START, base.CONTEXT - 1],
        "covered_types": base.COVERAGE,
        "dimension": D,
        "site_order": [list(value) for value in base.SITE_ORDER],
        "ridge_scale": base.RIDGE_SCALE,
        "effective_ridge": base.RIDGE,
        "arms": list(arm_descriptors()),
        "float64_fit_float32_deployment": (
            "contiguous .float factors; shared product then one private product/add"
        ),
        "call_schedule": expected_call_schedule(),
        "resource_ceiling": {
            "wall_seconds": base.MAX_WALL_SECONDS,
            "peak_allocated_cuda_bytes": base.MAX_ALLOCATED_CUDA_BYTES,
        },
        "authority_scope": "discovery_only_no_validation_final_or_semantic_coordinates",
    }
    body = {
        "schema": "hierarchical_shared_private_rrr_real_v1_authority",
        "status": "frozen_before_any_row_tensor_or_model_load",
        "source_closure": dict(source),
        "input_file_sha256s": dict(inputs),
        "checkpoint": dict(checkpoint),
        "protocol": protocol,
        "outputs": {
            "results": str(RESULTS), "failure": str(FAILURE), "receipt": str(RECEIPT),
        },
    }
    return {**body, "authority_sha256": base.logical_sha256(body)}


def build_spectral_state(embedding: torch.Tensor, table: torch.Tensor) -> base.SpectralState:
    x = embedding.detach().cpu().float().double()
    y = table.double()
    if x.shape != (base.COVERAGE, D) or y.shape != (N_SITES, base.COVERAGE, D):
        raise RuntimeError("hierarchical RRR fit tensors have wrong shapes")
    gram = x.T @ x
    crosses = tuple(x.T @ y[site] for site in range(N_SITES))
    y2 = tuple(float(y[site].square().sum()) for site in range(N_SITES))
    regularized = 0.5 * (gram + gram.T) + base.RIDGE * torch.eye(D, dtype=torch.float64)
    chol = torch.linalg.cholesky(regularized)
    solved = tuple(torch.cholesky_solve(cross, chol) for cross in crosses)
    merits = tuple(0.5 * (cross.T @ solution + solution.T @ cross)
                   for cross, solution in zip(crosses, solved, strict=True))
    independent = tuple(base._descending_eigh(merit) for merit in merits)
    global_values, global_vectors = base._descending_eigh(sum(
        merits[1:], merits[0].clone(),
    ))
    state = base.SpectralState(
        gram=gram, crosses=crosses, y2=y2, solved=solved,
        independent_values=tuple(item[0] for item in independent),
        independent_vectors=tuple(item[1] for item in independent),
        global_values=global_values, global_vectors=global_vectors,
        typed_values={}, typed_vectors={}, legacy_svd=None,
    )
    state.hierarchical_merits = merits
    state.hierarchical_residual_cache = None
    return state


@dataclass
class HierarchicalProgram:
    name: str
    descriptor: dict[str, Any]
    deployed: hybrid.DeployedHierarchicalProgram
    diagnostics: dict[str, Any]


def _allocation_cutoff_gap(spectra: Sequence[torch.Tensor], slots: int) -> float | None:
    values = sorted((float(value) for spectrum in spectra for value in spectrum), reverse=True)
    if slots == 0 or slots == len(values):
        return None
    return values[slots - 1] - values[slots]


def _private_boundary_gaps(
    spectra: Sequence[torch.Tensor], ranks: Sequence[int],
) -> list[float | None]:
    output = []
    for values, rank in zip(spectra, ranks, strict=True):
        output.append(None if rank == 0 or rank == values.numel()
                      else float(values[rank - 1] - values[rank]))
    return output


def _fit_from_residual(
    state: base.SpectralState, shared: torch.Tensor,
    residual: hybrid.ResidualEigensystem, budget: int,
) -> hybrid.HierarchicalFit:
    allocation = hybrid.allocate_private_ranks(
        residual.eigenvalues, dimension=D, shared_rank=shared.shape[1],
        total_float_budget=budget,
    )
    shared_inputs = tuple((value @ shared).contiguous() for value in state.solved)
    private_bases = tuple(
        residual.eigenvectors[site][:, :rank].contiguous()
        for site, rank in enumerate(allocation.private_ranks)
    )
    private_inputs = tuple(
        state.solved[site] @ private_bases[site] for site in range(N_SITES)
    )
    orthogonality = 0.0
    for basis in private_bases:
        combined = torch.cat((shared, basis), dim=1)
        if combined.shape[1]:
            orthogonality = max(orthogonality, float((
                combined.T @ combined - torch.eye(combined.shape[1], dtype=torch.float64)
            ).abs().max()))
    merits = state.hierarchical_merits
    return hybrid.HierarchicalFit(
        shared_basis=shared,
        shared_input_maps=shared_inputs,
        private_bases=private_bases,
        private_input_maps=private_inputs,
        residual_eigenvalues=residual.eigenvalues,
        allocation=allocation,
        price=hybrid.hierarchical_price(N_SITES, D, shared.shape[1], allocation.private_ranks),
        explained_shared_merit=sum(float(torch.trace(shared.T @ merit @ shared))
                                   for merit in merits),
        explained_private_merit=allocation.selected_residual_merit,
        combined_orthogonality_max_abs=orthogonality,
    )


def _residual_for_q0(state: base.SpectralState, q0: int) -> hybrid.ResidualEigensystem:
    cached = state.hierarchical_residual_cache
    if cached is not None and cached[0] == q0:
        return cached[1]
    if q0 == 0:
        residual = hybrid.ResidualEigensystem(
            complement_basis=torch.eye(D, dtype=torch.float64),
            eigenvalues=state.independent_values,
            eigenvectors=state.independent_vectors,
        )
    elif q0 == D:
        residual = hybrid.ResidualEigensystem(
            complement_basis=torch.empty((D, 0), dtype=torch.float64),
            eigenvalues=tuple(torch.empty(0, dtype=torch.float64) for _ in range(N_SITES)),
            eigenvectors=tuple(torch.empty((D, 0), dtype=torch.float64)
                               for _ in range(N_SITES)),
        )
    else:
        residual = hybrid.residual_eigensystems(
            state.hierarchical_merits, state.global_vectors[:, :q0],
        )
    state.hierarchical_residual_cache = (q0, residual)
    return residual


def _endpoint_controls(
    state: base.SpectralState, fit: hybrid.HierarchicalFit,
    deployed: hybrid.DeployedHierarchicalProgram, q0: int,
) -> dict[str, bool | None]:
    q0_equal: bool | None = None
    global_equal: bool | None = None
    deployed_maps = hybrid.deployed_coefficient_maps(deployed)
    if q0 == 0:
        expected = []
        for site, rank in enumerate(fit.allocation.private_ranks):
            basis64 = state.independent_vectors[site][:, :rank]
            basis32 = basis64.float().contiguous()
            input32 = (state.solved[site] @ basis64).float().contiguous()
            expected.append((input32 @ basis32.T).contiguous())
        q0_equal = all(torch.equal(left, right) for left, right in zip(
            deployed_maps, expected, strict=True,
        ))
    if q0 == D and sum(fit.allocation.private_ranks) == 0:
        shared64 = state.global_vectors[:, :q0]
        shared32 = shared64.float().contiguous()
        expected = tuple(
            (state.solved[site] @ shared64).float().contiguous() @ shared32.T
            for site in range(N_SITES)
        )
        global_equal = all(torch.equal(left, right) for left, right in zip(
            deployed_maps, expected, strict=True,
        ))
    if q0_equal is False or global_equal is False:
        raise RuntimeError("hierarchical RRR endpoint coefficient control failed")
    return {
        "q0_zero_exact_price_independent": q0_equal,
        "zero_private_exact_global": global_equal,
    }


def fit_program(descriptor: Mapping[str, Any], state: base.SpectralState) -> HierarchicalProgram:
    q0 = int(descriptor["shared_rank"])
    budget = int(descriptor["map_float_budget"])
    shared = state.global_vectors[:, :q0].contiguous()
    residual = _residual_for_q0(state, q0)
    fit = _fit_from_residual(state, shared, residual, budget)
    deployed = hybrid.materialize_float32_program(fit)
    hashes = hybrid.factor_hash_receipt(fit)
    endpoints = _endpoint_controls(state, fit, deployed, q0)
    shared_gap = None if q0 == 0 or q0 == D else float(
        state.global_values[q0 - 1] - state.global_values[q0]
    )
    diagnostics = {
        "explained_penalized_merit": (
            fit.explained_shared_merit + fit.explained_private_merit
        ),
        "explained_shared_merit": fit.explained_shared_merit,
        "explained_private_merit": fit.explained_private_merit,
        "penalized_residual_fraction": (
            sum(state.y2) - fit.explained_shared_merit - fit.explained_private_merit
        ) / sum(state.y2),
        "shared_rank": q0,
        "ranks_by_site": list(fit.allocation.private_ranks),
        "residual_eigenvalues": [value.tolist() for value in residual.eigenvalues],
        "shared_boundary_eigengap": shared_gap,
        "private_boundary_eigengaps": _private_boundary_gaps(
            residual.eigenvalues, fit.allocation.private_ranks,
        ),
        "allocation_cutoff_eigengap": _allocation_cutoff_gap(
            residual.eigenvalues, fit.allocation.private_rank_slots,
        ),
        "combined_orthogonality_max_abs_float64": fit.combined_orthogonality_max_abs,
        "map_float_count": fit.price.map_float_count,
        "map_float_bytes": fit.price.map_float_bytes,
        "common_table_float_count": COMMON_TABLE_FLOATS,
        "full_program_float_count": COMMON_TABLE_FLOATS + fit.price.map_float_count,
        "full_program_float_bytes": 4 * (COMMON_TABLE_FLOATS + fit.price.map_float_count),
        "dense_multiplies_per_uncovered_token": (
            fit.price.dense_multiplies_per_uncovered_token
        ),
        "deployed_hash_receipt": hashes,
        "endpoint_controls": endpoints,
        "finite": True,
    }
    semantic_validate_diagnostics(diagnostics, descriptor)
    return HierarchicalProgram(
        name=str(descriptor["name"]), descriptor=dict(descriptor),
        deployed=deployed, diagnostics=diagnostics,
    )


class AutonomousProgram:
    """All-36-site token-only hierarchical factor execution."""

    def __init__(self, model: torch.nn.Module, table: torch.Tensor, token_to_row: torch.Tensor,
                 factors: HierarchicalProgram, ledger: base.PhysicalCallLedger, device: str):
        self.model = model
        self.table = table.to(device)
        self.token_to_row = token_to_row.to(device)
        self.shared_basis = factors.deployed.shared_basis.to(device)
        self.shared_maps = tuple(value.to(device) for value in factors.deployed.shared_input_maps)
        self.private_bases = tuple(value.to(device) for value in factors.deployed.private_bases)
        self.private_maps = tuple(value.to(device) for value in factors.deployed.private_input_maps)
        self.arm = factors.name
        self.ledger = ledger

    def _write(self, index: int, tokens: torch.Tensor) -> torch.Tensor:
        embedding = self.model.transformer.wte(tokens)
        mapped = (embedding @ self.shared_maps[index]) @ self.shared_basis.T
        if self.private_maps[index].shape[1]:
            mapped = mapped + (
                (embedding @ self.private_maps[index]) @ self.private_bases[index].T
            )
        rows = self.token_to_row[tokens]
        covered = rows >= 0
        if bool(covered.any()):
            mapped = mapped.clone()
            mapped[covered] = self.table[index, rows[covered]]
        return mapped

    def attention(self, event: base.facade.AttentionEvent) -> tuple[torch.Tensor, torch.Tensor]:
        self.ledger.record_compiled_site(self.arm, "attn", event.site)
        write = self._write(base.SITE_TO_INDEX[("attn", event.site)], event.tokens)
        sentinel = torch.zeros(
            (*event.tokens.shape, event.block.attn.n_head, event.block.attn.head_dim),
            dtype=write.dtype, device=write.device,
        )
        return write, sentinel

    def mlp(self, event: base.facade.EarlyMLPEvent) -> torch.Tensor:
        self.ledger.record_compiled_site(self.arm, "mlp", event.site)
        return self._write(base.SITE_TO_INDEX[("mlp", event.site)], event.tokens)


def expected_program_price(descriptor: Mapping[str, Any]) -> tuple[int, tuple[int, ...] | None]:
    return int(descriptor["map_float_budget"]), None


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def semantic_validate_diagnostics(value: Mapping[str, Any], descriptor: Mapping[str, Any]) -> None:
    required = {
        "explained_penalized_merit", "explained_shared_merit", "explained_private_merit",
        "penalized_residual_fraction", "shared_rank", "ranks_by_site",
        "residual_eigenvalues", "shared_boundary_eigengap",
        "private_boundary_eigengaps", "allocation_cutoff_eigengap",
        "combined_orthogonality_max_abs_float64", "map_float_count", "map_float_bytes",
        "common_table_float_count", "full_program_float_count", "full_program_float_bytes",
        "dense_multiplies_per_uncovered_token", "deployed_hash_receipt",
        "endpoint_controls", "finite",
    }
    if set(value) != required or value["shared_rank"] != descriptor["shared_rank"] or (
        value["map_float_count"] != descriptor["map_float_budget"]
    ) or value["finite"] is not True:
        raise RuntimeError("hierarchical RRR diagnostic schema changed")
    q0 = int(descriptor["shared_rank"])
    spectra = tuple(torch.tensor(item, dtype=torch.float64) for item in value[
        "residual_eigenvalues"
    ])
    allocation = hybrid.allocate_private_ranks(
        spectra, dimension=D, shared_rank=q0,
        total_float_budget=int(descriptor["map_float_budget"]),
    )
    ranks = tuple(value["ranks_by_site"])
    price = hybrid.hierarchical_price(N_SITES, D, q0, allocation.private_ranks)
    if ranks != allocation.private_ranks or value["map_float_bytes"] != 4 * price.map_float_count or (
        value["common_table_float_count"] != COMMON_TABLE_FLOATS
    ) or value["full_program_float_count"] != COMMON_TABLE_FLOATS + price.map_float_count or (
        value["full_program_float_bytes"] != 4 * value["full_program_float_count"]
    ) or value["dense_multiplies_per_uncovered_token"] != (
        price.dense_multiplies_per_uncovered_token
    ) or value["private_boundary_eigengaps"] != _private_boundary_gaps(
        spectra, ranks,
    ) or value["allocation_cutoff_eigengap"] != _allocation_cutoff_gap(
        spectra, allocation.private_rank_slots,
    ):
        raise RuntimeError("hierarchical RRR allocation or price replay changed")
    controls = value["endpoint_controls"]
    if controls != {
        "q0_zero_exact_price_independent": True if q0 == 0 else None,
        "zero_private_exact_global": True if q0 == D else None,
    }:
        raise RuntimeError("hierarchical RRR endpoint control changed")
    hashes = value["deployed_hash_receipt"]
    if hashes.get("serialized_program_authority") is not False or hashes.get(
        "raw_factor_hashes_reported"
    ) is not False or hashes.get("hash_currency") != (
        "float32_deployed_projectors_and_coefficient_maps"
    ) or not _is_sha(hashes.get("sha256")) or not _is_sha(
        hashes.get("shared_projector_sha256")
    ) or len(hashes.get("site_projector_sha256s", ())) != N_SITES or len(
        hashes.get("coefficient_map_sha256s", ())
    ) != N_SITES or not all(_is_sha(item) for item in (
        *hashes["site_projector_sha256s"], *hashes["coefficient_map_sha256s"],
    )):
        raise RuntimeError("hierarchical RRR deployed hash receipt changed")
    numeric = (
        "explained_penalized_merit", "explained_shared_merit", "explained_private_merit",
        "penalized_residual_fraction", "combined_orthogonality_max_abs_float64",
    )
    if any(isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or not (
        math.isfinite(value[key])
    ) for key in numeric) or value["combined_orthogonality_max_abs_float64"] > 1e-8:
        raise RuntimeError("hierarchical RRR numerical diagnostic changed")


def _parent_arms() -> Mapping[str, Any]:
    return json.loads(PARENT_RESULTS.read_text())["arms"]


def comparison_ledger(arms: Mapping[str, Any]) -> dict[str, Any]:
    parent = _parent_arms()
    output: dict[str, Any] = {}
    for budget in GRID:
        middle = arms[arm_name(budget, 128)]["roles"]
        endpoint = arms[arm_name(budget, 0)]["roles"]
        comparison = {
            role: {"q128_minus_q0_ce": middle[role]["all"]["ce"] - endpoint[role]["all"]["ce"]}
            for role in base.ROLE_PATHS
        }
        if budget == "global_q512":
            other = arms[arm_name(budget, 512)]["roles"]
            for role in base.ROLE_PATHS:
                comparison[role]["q128_minus_q512_ce"] = (
                    middle[role]["all"]["ce"] - other[role]["all"]["ce"]
                )
        elif budget == "typed_q512":
            other = parent["typed_q512"]["roles"]
            for role in base.ROLE_PATHS:
                comparison[role]["q128_minus_parent_typed_q512_ce"] = (
                    middle[role]["all"]["ce"] - other[role]["all"]["ce"]
                )
        output[budget] = comparison
    return output


def _result_gates(arms: Mapping[str, Any], coverage: int) -> dict[str, Any]:
    del coverage
    comparisons = comparison_ledger(arms)
    roles = tuple(base.ROLE_PATHS)
    primary = all(
        comparisons["global_q512"][role][key] <= -0.01
        for role in roles for key in ("q128_minus_q0_ce", "q128_minus_q512_ce")
    )
    typed = all(
        comparisons["typed_q512"][role][key] <= -0.01
        for role in roles for key in (
            "q128_minus_q0_ce", "q128_minus_parent_typed_q512_ce",
        )
    )
    large = all(
        comparisons["independent_q512"][role]["q128_minus_q0_ce"] <= -0.005
        for role in roles
    )
    covered_spread = {
        role: max(payload["roles"][role]["covered"]["ce"] for payload in arms.values()) -
              min(payload["roles"][role]["covered"]["ce"] for payload in arms.values())
        for role in roles
    }
    parent = _parent_arms()
    endpoint_pairs = {
        arm_name("global_q512", 0): "price_global_q512",
        arm_name("global_q512", 512): "global_q512",
        arm_name("typed_q512", 0): "price_typed_q512",
    }
    endpoint_replay = {
        name: {role: abs(
            arms[name]["roles"][role]["all"]["ce"] - parent[parent_name]["roles"][role]["all"]["ce"]
        ) <= 0.002 for role in roles}
        for name, parent_name in endpoint_pairs.items()
    }
    literal_endpoints = all(
        payload["diagnostics"]["endpoint_controls"][key] is True
        for payload in arms.values() for key in (
            "q0_zero_exact_price_independent", "zero_private_exact_global",
        ) if payload["diagnostics"]["endpoint_controls"][key] is not None
    )
    integrity = (
        all(value <= 1e-6 for value in covered_spread.values()) and
        all(all(values.values()) for values in endpoint_replay.values()) and literal_endpoints
    )
    return {
        "primary_global_budget_pass": primary,
        "typed_budget_pass": typed,
        "large_budget_diagnostic_pass": large,
        "covered_identity_spread_by_role": covered_spread,
        "covered_identity_control": all(value <= 1e-6 for value in covered_spread.values()),
        "parent_endpoint_replay": endpoint_replay,
        "literal_endpoint_controls": literal_endpoints,
        "integrity_conjunction": integrity,
        "registered_status": (
            "primary_and_typed_pass" if primary and typed and integrity
            else "registered_predictions_not_jointly_supported"
        ),
    }


_BASE_DEFAULTS = {
    name: getattr(base, name) for name in (
        "PROTOCOL_VERSION", "AUTHORITY", "RESULTS", "FAILURE", "RECEIPT", "LOCK",
        "SOURCE_PATHS", "FILE_PINS", "RECOVERY_PARENT", "arm_descriptors",
        "expected_call_schedule", "authority_payload", "build_spectral_state",
        "fit_program", "AutonomousProgram", "expected_program_price",
        "semantic_validate_diagnostics", "comparison_ledger", "_result_gates",
    )
}


def configure_base() -> None:
    assignments = {
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "AUTHORITY": AUTHORITY, "RESULTS": RESULTS, "FAILURE": FAILURE,
        "RECEIPT": RECEIPT, "LOCK": LOCK, "SOURCE_PATHS": SOURCE_PATHS,
        "FILE_PINS": FILE_PINS, "RECOVERY_PARENT": None,
        "arm_descriptors": arm_descriptors,
        "expected_call_schedule": expected_call_schedule,
        "authority_payload": authority_payload,
        "build_spectral_state": build_spectral_state,
        "fit_program": fit_program,
        "AutonomousProgram": AutonomousProgram,
        "expected_program_price": expected_program_price,
        "semantic_validate_diagnostics": semantic_validate_diagnostics,
        "comparison_ledger": comparison_ledger,
        "_result_gates": _result_gates,
    }
    for name, value in assignments.items():
        setattr(base, name, value)


def restore_base_defaults() -> None:
    for name, value in _BASE_DEFAULTS.items():
        setattr(base, name, value)


def run(*, device: str = "cuda") -> dict[str, Any]:
    configure_base()
    verify_parent()
    return base.run(device=device)


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
