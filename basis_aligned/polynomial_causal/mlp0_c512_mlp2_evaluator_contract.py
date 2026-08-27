"""Pure authority/evaluator contract for the physical C512 -> MLP2 assay.

These helpers define identities, phase-specific call counts, contrast orientation,
and control receipts without loading the language model or performing a forward.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
import torch

from mlp0_c512_mlp2_compensation import additive_factorial_prediction
from score_mlp0_c512_mlp2_compensation_v1 import (
    COMPARISON_DEFINITIONS,
    CONTRAST_ORIENTATIONS,
    CONTRASTS,
    integer_array_sha256,
    ordered_ids_sha256,
)


D_MODEL = 1152
N_LAYERS = 18
T = 256
BATCH_WINDOWS = 4
FIT_BATCH = 4

ARM_CARRIED_PATH = {
    "OO": "O", "OC": "O", "O0": "O", "ON": "O",
    "CC": "C", "CO": "C", "C0": "C", "CS": "C",
}


def build_unit_identity(row_receipt: Mapping[str, object]) -> dict[str, object]:
    records = row_receipt["document_provenance"]["sets"]["eval"]
    by_ordinal: dict[int, str] = {}
    wave_by_ordinal: dict[int, str] = {}
    row_units: list[int] = []
    for record in records:
        ordinal = int(record["source_document_ordinal"])
        document = str(record["document_id"])
        wave = str(record["wave"])
        if ordinal in by_ordinal and by_ordinal[ordinal] != document:
            raise RuntimeError("one source-document ordinal maps to multiple documents")
        if ordinal in wave_by_ordinal and wave_by_ordinal[ordinal] != wave:
            raise RuntimeError("one source document crosses replication waves")
        by_ordinal[ordinal] = document
        wave_by_ordinal[ordinal] = wave
        row_units.append(ordinal)
    if sorted(by_ordinal) != list(range(384)):
        raise RuntimeError("source-document ordinals are not exactly 0..383")
    waves = [wave_by_ordinal[index] for index in range(384)]
    if waves != ["A"] * 192 + ["B"] * 192:
        raise RuntimeError("the registered 192/192 wave split changed")
    return {
        "unit_kind": "source_document",
        "ordered_ids": [by_ordinal[index] for index in range(384)],
        "row_to_unit": row_units + row_units,
        "wave_labels": waves,
    }


def unit_identity_hashes(identity: Mapping[str, object]) -> dict[str, str]:
    mapping = np.asarray(identity["row_to_unit"], dtype=np.int64)
    occupancy = np.bincount(mapping, minlength=len(identity["ordered_ids"]))
    return {
        "ordered_ids_sha256": ordered_ids_sha256(identity["ordered_ids"]),
        "row_to_unit_sha256": integer_array_sha256(mapping),
        "occupancy_sha256": integer_array_sha256(occupancy),
        "wave_labels_sha256": ordered_ids_sha256(identity["wave_labels"]),
    }


def expected_call_contract(n_eval_rows: int) -> dict[str, object]:
    """Freeze every MLP-site call by execution phase.

    Each evaluation batch has two O/C interface-capture passes, repeated in the
    preparation and scoring passes (4B).  Four independent parents contribute 4B
    at every MLP except MLP2, where the two omission parents skip its write (2B).
    Eight crossed arms call only the unchanged block-3..17 suffix (8B).
    """
    if n_eval_rows <= 0:
        raise ValueError("evaluation row count must be positive")
    eval_batches = (n_eval_rows + BATCH_WINDOWS - 1) // BATCH_WINDOWS
    parent = {str(site): 4 * eval_batches for site in range(N_LAYERS)}
    parent["2"] = 2 * eval_batches
    return {
        "n_eval_windows": n_eval_rows,
        "exact_call_counts": {
            "candidate_original_down_calls": 0,
            "poison_canary_calls": 1,
            "c512_proxy_calls": 4 * eval_batches,
        },
        "exact_phase_site_call_counts": {
            "mlp1_teacher_capture": {"1": 4 * eval_batches},
            "mlp2_teacher_capture": {"2": 4 * eval_batches},
            "parent_replay_mlp_sites": parent,
            "crossed_suffix_replay": {
                str(site): 8 * eval_batches for site in range(3, N_LAYERS)
            },
            "crossed_forbidden_teacher": {"1": 0, "2": 0},
        },
    }


def contrast_logits(
    logits: Mapping[str, torch.Tensor]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if set(logits) != {"OO", "CC", "CO", "OC", "O0", "C0", "CS", "ON"}:
        raise ValueError("raw logits do not contain the exact eight-arm family")
    additive = additive_factorial_prediction(logits)
    output = {
        "observational": (logits["OO"], logits["CC"]),
        "prewrite_state": (logits["OO"], logits["CO"]),
        "write_on_exact_state": (logits["OO"], logits["OC"]),
        "write_on_candidate_state": (logits["CO"], logits["CC"]),
        "interaction": (logits["CC"], additive),
        "omission_exposure": (logits["O0"], logits["C0"]),
        "alignment_null": (logits["OO"], logits["CS"]),
        "sensitivity": (logits["OO"], logits["ON"]),
    }
    if tuple(output) != CONTRASTS:
        raise RuntimeError("contrast order drifted from the scorer")
    observed = {
        key: [
            next((arm for arm, value in logits.items() if value is pair[0]),
                 "additive_CO_plus_OC_minus_OO"),
            next((arm for arm, value in logits.items() if value is pair[1]),
                 "additive_CO_plus_OC_minus_OO"),
        ]
        for key, pair in output.items()
    }
    if observed != {key: list(value) for key, value in CONTRAST_ORIENTATIONS.items()}:
        raise RuntimeError("runtime contrast orientation differs from the frozen scorer")
    return output


def carried_inputs_for_arm(
    arm: str, interfaces: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the block-2 carried value state and x0 bound to an arm's state path."""
    if arm not in ARM_CARRIED_PATH or set(interfaces) != {"O", "C"}:
        raise ValueError("arm or physical interface family is not registered")
    path = ARM_CARRIED_PATH[arm]
    values = interfaces[path]
    if "v1" not in values or "x0" not in values:
        raise ValueError("physical interface is missing carried suffix inputs")
    return values["v1"], values["x0"]


def coverage_by_wave(
    valid: torch.Tensor, unit_ids: torch.Tensor, wave_labels: list[str]
) -> dict[str, float]:
    if valid.ndim != 2 or unit_ids.shape != (valid.shape[0],) or len(wave_labels) != 384:
        raise ValueError("coverage identity tensors are incompatible")
    if bool(((unit_ids < 0) | (unit_ids >= len(wave_labels))).any()):
        raise ValueError("coverage contains an out-of-range source-document id")
    wave_code = torch.tensor([label == "B" for label in wave_labels], dtype=torch.bool)
    if wave_labels != ["A"] * 192 + ["B"] * 192:
        raise ValueError("wave labels changed")
    row_wave_b = wave_code[unit_ids.cpu()]
    first = ~row_wave_b
    second = row_wave_b
    return {
        "wave_A": float(valid[first].float().mean()),
        "wave_B": float(valid[second].float().mean()),
        "pooled": float(valid.float().mean()),
    }


def control_contract_sha256() -> str:
    """Hash the pre-forward deterministic recipe, never an activation realization."""
    values = {
        "algorithm": "largest_occupancy_circular_derangement_within_wave_x_cell",
        "permutation_unit": "token_position",
        "multiset": "delta_mlp2_write_vectors",
        "arm_carried_path": ARM_CARRIED_PATH,
        "comparison_definitions": {
            key: list(value) for key, value in COMPARISON_DEFINITIONS.items()
        },
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def control_realization_sha256(
    permutation: torch.Tensor,
    recipient_units: torch.Tensor,
    donor_units: torch.Tensor,
    groups: torch.Tensor,
) -> str:
    values = {
        "permutation": permutation.detach().cpu().long().tolist(),
        "recipient_units": recipient_units.detach().cpu().long().tolist(),
        "donor_units": donor_units.detach().cpu().long().tolist(),
        "groups": groups.detach().cpu().long().tolist(),
        "comparison_definitions": {
            key: list(value) for key, value in COMPARISON_DEFINITIONS.items()
        },
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_derangement(
    permutation: torch.Tensor,
    recipient_units: torch.Tensor,
    donor_units: torch.Tensor,
    recipient_groups: torch.Tensor,
    donor_groups: torch.Tensor,
) -> dict[str, bool]:
    if not (permutation.shape == recipient_units.shape == donor_units.shape
            == recipient_groups.shape == donor_groups.shape):
        raise ValueError("derangement identity arrays have different shapes")
    if permutation.dtype not in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    }:
        raise ValueError("derangement permutation must have integer dtype")
    permutation = permutation.long()
    n = permutation.numel()
    bijection = bool(torch.equal(torch.sort(permutation).values,
                                 torch.arange(n, device=permutation.device)))
    indexed = bool(
        bijection
        and torch.equal(donor_units, recipient_units[permutation])
        and torch.equal(donor_groups, recipient_groups[permutation])
    )
    return {
        "derangement_bijection": bijection,
        "donor_arrays_indexed_by_permutation": indexed,
        "derangement_no_same_document": bool((recipient_units != donor_units).all()),
        "derangement_wave_cell_preserving": bool(
            torch.equal(recipient_groups, donor_groups)
        ),
    }
