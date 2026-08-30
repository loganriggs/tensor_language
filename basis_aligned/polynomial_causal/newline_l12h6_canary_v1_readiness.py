"""Nonauthorizing source/readiness boundary for the newline L12H6 canary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import subprocess
from pathlib import Path
from typing import Mapping

import torch

from circuit_newline_fixed_crew_v1 import NewlineMaskSpec, build_newline_masks
import newline_l12h6_canary_rows_v1 as rows_contract
import newline_l12h6_token_registry_v1 as token_registry


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/NEWLINE_FIXED_CREW_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/NEWLINE_L12H6_CANARY_V1_EXECUTION_AMENDMENT.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/circuit_campaign_statistics.py",
    "basis_aligned/polynomial_causal/circuit_newline_fixed_crew_v1.py",
    "basis_aligned/polynomial_causal/newline_l12h6_canary_rows_v1.py",
    "basis_aligned/polynomial_causal/newline_l12h6_canary_v1_readiness.py",
    "basis_aligned/polynomial_causal/newline_l12h6_token_registry_v1.py",
    "basis_aligned/polynomial_causal/tensor_preserving_attention.py",
    "basis_aligned/polynomial_causal/test_circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/test_circuit_campaign_statistics.py",
    "basis_aligned/polynomial_causal/test_circuit_newline_fixed_crew_v1.py",
    "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/test_newline_l12h6_canary_rows_v1.py",
    "basis_aligned/polynomial_causal/test_newline_l12h6_canary_v1_readiness.py",
    "basis_aligned/polynomial_causal/test_newline_l12h6_token_registry_v1.py",
    "basis_aligned/polynomial_causal/test_tensor_preserving_attention.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
)
LAUNCH_BLOCKERS = (
    "audited prose/code/list candidate enumerator",
    "create-only row publisher and receipt",
    "external independent audit and pre-forward authority",
    "checkpoint-derived exact program manifest",
    "terminal facade transaction with failure/receipt exclusivity",
)


def source_closure(commit: str) -> dict[str, str]:
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("newline readiness commit is malformed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    result = {}
    for relative in SOURCE_PATHS:
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        live = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if hashlib.sha256(live).hexdigest() != digest:
            raise RuntimeError(f"newline readiness source drift: {relative}")
        result[relative] = digest
    return result


@dataclass(frozen=True)
class RowReadiness:
    schema: str
    token_registry_sha256: str
    role_summaries: tuple[tuple[str, Mapping[str, object]], ...]
    allocation_seed: str
    model_forward_authorized: bool
    row_publication_authorized: bool

    def __post_init__(self) -> None:
        if self.schema != "newline_l12h6_canary_v1_row_readiness" or (
            self.token_registry_sha256 != token_registry.REGISTRY_SHA256
        ) or tuple(role for role, _summary in self.role_summaries) != rows_contract.ROLE_ORDER:
            raise ValueError("newline row readiness identity changed")
        if not isinstance(self.allocation_seed, str) or not self.allocation_seed or (
            self.model_forward_authorized is not False or self.row_publication_authorized is not False
        ):
            raise ValueError("newline readiness cannot authorize publication or forwards")


def build_readiness(
    roles: tuple[rows_contract.FrozenRole, ...],
    registry: Mapping[str, object],
    spec: NewlineMaskSpec,
    *,
    allocation_seed: str,
) -> RowReadiness:
    token_registry.validate_registry(registry)
    expected_spec = NewlineMaskSpec(
        newline_token_ids=tuple(registry["newline"]),
        punctuation_token_ids=tuple(registry["punctuation"]),
        capitalized_token_ids=tuple(registry["capitalized"]),
        quote_bracket_token_ids=tuple(registry["quote_bracket"]),
        first_prediction=64,
        jitter_offsets=(2, -2, 3, -3, 4, -4, 8, -8, 16, -16, 32, -32),
        random_seed=2_026_083_000,
    )
    if spec != expected_spec:
        raise RuntimeError("newline mask spec differs from frozen token registry/rules")
    rows_contract.validate_role_disjointness(roles)
    for role in roles:
        replayed = build_newline_masks(role.rows, spec)
        if tuple(role.masks.as_mapping()) != tuple(replayed.as_mapping()) or any(
            not torch.equal(role.masks.as_mapping()[name], replayed.as_mapping()[name])
            for name in replayed.as_mapping()
        ) or role.support != rows_contract.support_census(replayed, role.records):
            raise RuntimeError(f"newline {role.role} masks/support do not replay from rows")
    summaries = tuple((role.role, rows_contract.role_summary(role)) for role in roles)
    return RowReadiness(
        "newline_l12h6_canary_v1_row_readiness", token_registry.REGISTRY_SHA256,
        summaries, allocation_seed, False, False,
    )


def require_launch_ready(_readiness: RowReadiness) -> None:
    raise RuntimeError("newline L12H6 remains launch-NO-GO: " + "; ".join(LAUNCH_BLOCKERS))


__all__ = (
    "LAUNCH_BLOCKERS", "RowReadiness", "SOURCE_PATHS", "build_readiness",
    "require_launch_ready", "source_closure",
)
