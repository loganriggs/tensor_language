#!/usr/bin/env python3
"""R591 diagnostic-only replay/native numerical decomposition.

The managed preflight path is model-free.  The real path emits one strict JSON
object to stdout and has no result, receipt, evidence, score, selection, or
publication namespace.  It diagnoses R585's failed replay/native identity; it
cannot produce a scientific terminal.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT.parent / "polynomial_causal"
OPS = ROOT / "ops"

R585 = OPS / "induction_selector_payload_frozen_factor_rung585.py"
R585_TEST = OPS / "test_induction_selector_payload_frozen_factor_rung585.py"
R585_DRYRUN = ROOT / "induction_selector_payload_frozen_factor_rung585_dryrun.json"
FACADE = POLY / "bilin18_observed_model_facade.py"
INDUCTION = POLY / "circuit_induction_tensor.py"
MANIFEST = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"
DEPENDENCY_LOCK = ROOT / "induction_selector_payload_frozen_factor_rung585_dependency_lock.json"
METHOD_HANDOFF_V5 = OPS / "circuit_causal_validity_next_wave_handoff_rung585_v5_addendum.json"
PREREGISTRATION = POLY / "INDUCTION_REPLAY_NATIVE_NUMERICS_RUNG591_PREREGISTRATION.md"

SOURCE_HASHES = {
    R585: "fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b",
    R585_TEST: "fcaba664269de12a41a5adb8ff089fc9963eeec91577ef94993ff032c02fc885",
    R585_DRYRUN: "580a570426ce48c9e43f5fce82c976dece6c71e8a11c1b057054c17cf958dcf8",
    FACADE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    INDUCTION: "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
    MANIFEST: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    DEPENDENCY_LOCK: "908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7",
    METHOD_HANDOFF_V5: "810d15aa7f86a9896ca56e48c7ea33c60b10f6b0d266acefa5f3441333c8fe80",
    PREREGISTRATION: "e72cb386d65c68f55b767c8141c3c4d774b3c8ad9387ac7f8ad43bebef118593",
}

CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
R585_COMMIT = "c4288dbe8ee6213dfc4dcb538024dc119fbb642e"
SCHEMA = "induction_replay_native_numerics_rung591_diagnostic_v1"
DRYRUN_SCHEMA = "induction_replay_native_numerics_rung591_dryrun_v1"
STATUS = "diagnostic_only_no_scientific_terminal"
TOLERANCE = 1e-5
VOCABULARY_SIZE = 50_304
PAD_TOKEN = 50_256
FIT_ENDPOINTS = 1_728
BATCH = 32
FIT_LENGTHS = (19, 20, 27, 28)
ALL_LENGTHS = (19, 20, 21, 22, 27, 28, 29, 30)
EXPECTED_LENGTH_HISTOGRAMS = {
    "FIT": {19: 960, 20: 480, 27: 192, 28: 96},
    "SELECT": {21: 480, 22: 240, 29: 96, 30: 48},
}
SITES = ((5, 5), (7, 3), (8, 3), (8, 4))
TERM_NAMES = tuple(f"L{site}H{head}" for site, head in SITES)
DISPATCHERS = ("N", "F", "R")
PANEL_SCHEDULES = ("L_native", "L_30", "M_30")

EXPECTED_PANEL_SHA256 = "6b56a6740dbea7d0765d6a8668361ff43b06562152f091f6969ca8591522ebe4"
EXPECTED_SUPPORT_SHA256 = "e2de29dcf3cb37187060ab72775533086612bbb349777d48bd9f8feb8911e9fa"
EXPECTED_CALL_MANIFEST_SHA256 = "1e838190752e72eed6f35119c3e99bfb7620e787ae73c7a052046160d600ad3f"

REGISTERED_PREDICATES = {
    "pred_a_observer_identity": "F and N agree on identical token tensors at absolute 1e-5",
    "pred_b_hook_identity": "R and N agree on identical token tensors at absolute 1e-5",
    "pred_c_padding_identity": "native padded and unpadded logits agree at absolute 1e-5",
    "pred_d_membership_identity": "native fixed-shape batch memberships agree at absolute 1e-5",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def require_finite_json(value: object, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"nonfinite value at {path}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            require_finite_json(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"non-string JSON key at {path}")
            require_finite_json(child, f"{path}.{key}")
        return
    raise TypeError(f"non-JSON type {type(value).__name__} at {path}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_sources() -> dict[str, str]:
    observed = {}
    for path, expected in SOURCE_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen source mismatch: {path}")
        observed[str(path)] = expected
    return observed


def load_authority() -> tuple[object, dict[str, object]]:
    verify_sources()
    r585 = load_module(R585, "r591_pinned_r585_authority")
    execution = r585.build_execution_authority()
    return r585, execution


def audit_equality_support(
    endpoints: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Independently prove A/C exhaust canonical successor support."""
    rows = []
    for endpoint in endpoints:
        tokens = list(map(int, endpoint["token_ids"]))
        query = int(endpoint["final_position"])
        canonical = [
            key for key in range(1, query + 1)
            if tokens[key - 1] == tokens[query]
        ]
        registered = [
            int(payload)
            for source, payload in zip(
                endpoint["source_positions"], endpoint["payload_positions"]
            )
            if tokens[int(source)] == tokens[query]
        ]
        rows.append({
            "split": str(endpoint["split"]),
            "endpoint_id": str(endpoint["endpoint_id"]),
            "canonical_support_positions": canonical,
            "registered_support_positions": registered,
            "extra_positions": sorted(set(canonical) - set(registered)),
            "missing_positions": sorted(set(registered) - set(canonical)),
        })
    digest = content_sha256(rows)
    histogram = Counter(len(row["canonical_support_positions"]) for row in rows)
    result = {
        "endpoint_count": len(rows),
        "canonical_support_count_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "extra_position_count": sum(len(row["extra_positions"]) for row in rows),
        "missing_position_count": sum(len(row["missing_positions"]) for row in rows),
        "ordered_census_sha256": digest,
    }
    if result != {
        "endpoint_count": 2_592,
        "canonical_support_count_histogram": {"0": 432, "1": 2_160},
        "extra_position_count": 0,
        "missing_position_count": 0,
        "ordered_census_sha256": EXPECTED_SUPPORT_SHA256,
    }:
        raise RuntimeError("canonical equality support census changed")
    return result


def select_panel_for_lengths(
    endpoints: Sequence[Mapping[str, object]], lengths: Sequence[int], per_length: int,
) -> list[Mapping[str, object]]:
    fit = [row for row in endpoints if row["split"] == "FIT"]
    panel = []
    for length in lengths:
        choices = sorted(
            (row for row in fit if int(row["length"]) == length),
            key=lambda row: str(row["endpoint_id"]),
        )
        if len(choices) < per_length:
            raise RuntimeError(f"insufficient FIT endpoints at length {length}")
        panel.extend(choices[:per_length])
    return panel


def select_panel(endpoints: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    histogram = {
        split: dict(sorted(Counter(
            int(row["length"]) for row in endpoints if row["split"] == split
        ).items()))
        for split in ("FIT", "SELECT")
    }
    if histogram != EXPECTED_LENGTH_HISTOGRAMS:
        raise RuntimeError("FIT/SELECT length histogram changed")
    panel = select_panel_for_lengths(endpoints, FIT_LENGTHS, 64)
    membership = [
        {"length": int(row["length"]), "endpoint_id": str(row["endpoint_id"])}
        for row in panel
    ]
    if len(panel) != 256 or len({row["endpoint_id"] for row in panel}) != 256:
        raise RuntimeError("controlled panel census changed")
    if content_sha256(membership) != EXPECTED_PANEL_SHA256:
        raise RuntimeError("controlled panel membership changed")
    return panel


def _batch(
    records: Sequence[Mapping[str, object]], schedule: str, batch_index: int,
    padding_length: int,
) -> dict[str, object]:
    rows = list(records)
    if len(rows) != BATCH or padding_length < max(int(row["length"]) for row in rows):
        raise RuntimeError("R591 batch shape changed")
    return {
        "schedule": schedule,
        "batch_index": batch_index,
        "padding_length": padding_length,
        "records": rows,
    }


def full_fit_schedules(
    endpoints: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    fit = [row for row in endpoints if row["split"] == "FIT"]
    if len(fit) != FIT_ENDPOINTS:
        raise RuntimeError("FIT endpoint census changed")
    mixed_order = sorted(fit, key=lambda row: str(row["endpoint_id"]))
    length_order = sorted(fit, key=lambda row: (int(row["length"]), str(row["endpoint_id"])))

    def make(ordered, name):
        groups = [ordered[start:start + BATCH] for start in range(0, len(ordered), BATCH)]
        return [
            _batch(group, name, index, max(int(row["length"]) for row in group))
            for index, group in enumerate(groups)
        ]

    result = {
        "M": make(mixed_order, "M",),
        "L": make(length_order, "L",),
    }
    if any(len(value) != 54 for value in result.values()):
        raise RuntimeError("full-FIT batch price changed")
    return result


def panel_schedules(
    panel: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_length = {
        length: sorted(
            (row for row in panel if int(row["length"]) == length),
            key=lambda row: str(row["endpoint_id"]),
        )
        for length in FIT_LENGTHS
    }
    if any(len(rows) != 64 for rows in by_length.values()):
        raise RuntimeError("panel length census changed")
    native, padded = [], []
    for length in FIT_LENGTHS:
        for group_index in range(2):
            rows = by_length[length][32 * group_index:32 * group_index + 32]
            native.append(_batch(rows, "L_native", len(native), length))
            padded.append(_batch(rows, "L_30", len(padded), 30))
    mixed = []
    for index in range(8):
        records = [
            row
            for length in FIT_LENGTHS
            for row in by_length[length][8 * index:8 * index + 8]
        ]
        mixed.append(_batch(records, "M_30", index, 30))
    result = {"L_native": native, "L_30": padded, "M_30": mixed}
    expected = {str(row["endpoint_id"]) for row in panel}
    for name, batches in result.items():
        realized = [str(row["endpoint_id"]) for batch in batches for row in batch["records"]]
        if len(batches) != 8 or len(realized) != 256 or set(realized) != expected:
            raise RuntimeError(f"panel schedule {name} changed")
    return result


def build_call_manifest(
    endpoints: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    full = full_fit_schedules(endpoints)
    panel = panel_schedules(select_panel(endpoints))
    conditions = [
        ("full_fit", "R", "M", full["M"]),
        ("full_fit", "N", "M", full["M"]),
        ("full_fit", "N", "L", full["L"]),
    ]
    conditions.extend(
        ("panel", dispatcher, schedule, panel[schedule])
        for dispatcher in DISPATCHERS for schedule in PANEL_SCHEDULES
    )
    manifest = []
    for phase, dispatcher, schedule, batches in conditions:
        for batch in batches:
            records = batch["records"]
            manifest.append({
                "call_index": len(manifest),
                "phase": phase,
                "dispatcher": dispatcher,
                "schedule": schedule,
                "batch_index": int(batch["batch_index"]),
                "batch_size": len(records),
                "padding_length": int(batch["padding_length"]),
                "endpoint_ids": [str(row["endpoint_id"]) for row in records],
            })
    if len(manifest) != 234 or any(row["batch_size"] != 32 for row in manifest):
        raise RuntimeError("R591 forward-call census changed")
    counts = Counter(row["dispatcher"] for row in manifest)
    if counts != {"N": 132, "F": 24, "R": 78}:
        raise RuntimeError("R591 dispatcher-call census changed")
    if content_sha256(manifest) != EXPECTED_CALL_MANIFEST_SHA256:
        raise RuntimeError("R591 ordered forward-call manifest changed")
    return manifest


def forward_shape_contract(manifest: Sequence[Mapping[str, object]]) -> dict[str, object]:
    shapes = Counter(
        f"{row['batch_size']}x{row['padding_length']}" for row in manifest
    )
    return {
        "call_count": len(manifest),
        "ordered_manifest_sha256": content_sha256(list(manifest)),
        "shape_histogram": {key: shapes[key] for key in sorted(shapes)},
        "all_batch_sizes": sorted({int(row["batch_size"]) for row in manifest}),
        "all_padding_lengths": sorted({int(row["padding_length"]) for row in manifest}),
        "facade_require_production": False,
        "production_model_validated_separately": True,
        "token_dtype": "torch.int64",
        "token_minimum": 0,
        "token_maximum_exclusive": 50_257,
    }


def operation_census(manifest: Sequence[Mapping[str, object]]) -> dict[str, object]:
    endpoint_forwards = sum(int(row["batch_size"]) for row in manifest)
    factor_endpoints = sum(
        int(row["batch_size"]) for row in manifest if row["dispatcher"] in ("F", "R")
    )
    return {
        "model_forwards": len(manifest),
        "model_backwards": 0,
        "model_weights_updated": False,
        "endpoint_forwards": endpoint_forwards,
        "dispatcher_forward_counts": {
            dispatcher: sum(row["dispatcher"] == dispatcher for row in manifest)
            for dispatcher in DISPATCHERS
        },
        "factor_endpoint_site_operations": factor_endpoints * len(SITES),
        "factor_endpoint_site_role_operations": factor_endpoints * len(SITES) * 2,
        "factor_operations_by_site": {
            term: factor_endpoints for term in TERM_NAMES
        },
        "evaluated_splits": ["FIT"],
        "forbidden_splits_opened": [],
    }


def build_dryrun() -> dict[str, object]:
    sources = verify_sources()
    _, execution = load_authority()
    support = audit_equality_support(execution["endpoints"])
    panel = select_panel(execution["endpoints"])
    manifest = build_call_manifest(execution["endpoints"])
    payload = {
        "schema": DRYRUN_SCHEMA,
        "rung": 591,
        "status": "model_free_dryrun",
        "scientific_status": STATUS,
        "registered_tolerance": TOLERANCE,
        "r585_commit": R585_COMMIT,
        "source_sha256": sources,
        "checkpoint_weights_sha256": CHECKPOINT_SHA256,
        "panel": {
            "endpoint_count": len(panel),
            "length_counts": {
                str(length): sum(int(row["length"]) == length for row in panel)
                for length in FIT_LENGTHS
            },
            "ordered_membership_sha256": content_sha256([
                {"length": int(row["length"]), "endpoint_id": str(row["endpoint_id"])}
                for row in panel
            ]),
        },
        "split_length_histograms": {
            split: {str(length): count for length, count in counts.items()}
            for split, counts in EXPECTED_LENGTH_HISTOGRAMS.items()
        },
        "equality_support_audit": support,
        "forward_call_shape_contract": forward_shape_contract(manifest),
        "operation_census": operation_census(manifest),
        "dispatchers": list(DISPATCHERS),
        "panel_schedules": list(PANEL_SCHEDULES),
        "output_boundary": {
            "stdout_json_only": True,
            "writes_result": False,
            "writes_receipt": False,
            "writes_evidence": False,
            "calls_scoring": False,
            "calls_selection": False,
            "publishes_scientific_terminal": False,
        },
        "registered_interpretation_order": [
            "observer_failure", "hook_dominated", "padding_dominated",
            "membership_gemm_dominated", "mixed", "missing_factor",
        ],
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
    }
    require_finite_json(payload)
    return payload


def _padded_tokens(batch: Mapping[str, object], *, torch, device):
    records = batch["records"]
    length = int(batch["padding_length"])
    tokens = torch.full(
        (len(records), length), PAD_TOKEN, dtype=torch.long, device=device
    )
    for index, row in enumerate(records):
        ids = torch.as_tensor(row["token_ids"], dtype=torch.long, device=device)
        if len(ids) > length:
            raise RuntimeError("forced padding length truncates an endpoint")
        tokens[index, :len(ids)] = ids
    return tokens


def resolve_factor_write(mode: str, native_write, terms, batch, *, torch):
    """F is observational; R applies exactly R585's four-site delta."""
    if mode == "F":
        return native_write
    if mode != "R":
        raise ValueError("factor write is defined only for F/R")
    modified = native_write.clone()
    device = native_write.device
    for local, row_terms in enumerate(terms):
        query = int(batch[local]["final_position"])
        total_delta = torch.zeros(1152, device=device, dtype=native_write.dtype)
        for term_name in TERM_NAMES:
            if term_name in row_terms:
                term = row_terms[term_name]
                total_delta += (
                    term["term"].to(device=device, dtype=native_write.dtype)
                    - term["canonical"].to(device=device, dtype=native_write.dtype)
                )
        modified[local, query] += total_delta
    return modified


def _blank_local_exactness() -> dict[str, dict[str, float]]:
    return {
        term: {
            "term_canonical_max_abs": 0.0,
            "canonical_plus_remainder_head_max_abs": 0.0,
            "reconstructed_attention_native_write_max_abs": 0.0,
        }
        for term in TERM_NAMES
    }


def collect_condition(
    model, batches: Sequence[Mapping[str, object]], mode: str, *,
    torch, functional, facade, induction, r585,
) -> dict[str, object]:
    if mode not in DISPATCHERS:
        raise ValueError("unknown R591 dispatcher")
    device = next(model.parameters()).device
    logits_by_endpoint = {}
    locations = {}
    local = _blank_local_exactness()
    calls = 0
    with torch.inference_mode():
        for batch in batches:
            records = batch["records"]
            tokens = _padded_tokens(batch, torch=torch, device=device)

            def attention(event):
                if mode == "N" or event.site not in {site for site, _ in SITES}:
                    return event.block.attn(event.state, event.first_value)
                native_write, next_value, terms, full_error = r585.factorize_attention_event(
                    event, records, torch=torch, functional=functional, induction=induction
                )
                for row_terms in terms:
                    for term_name, term in row_terms.items():
                        cell = local[term_name]
                        cell["term_canonical_max_abs"] = max(
                            cell["term_canonical_max_abs"], float(term["factor_error"])
                        )
                        cell["canonical_plus_remainder_head_max_abs"] = max(
                            cell["canonical_plus_remainder_head_max_abs"],
                            float(term["reconstruction_error"]),
                        )
                        cell["reconstructed_attention_native_write_max_abs"] = max(
                            cell["reconstructed_attention_native_write_max_abs"],
                            float(full_error),
                        )
                write = resolve_factor_write(
                    mode, native_write, terms, records, torch=torch
                )
                return write, next_value

            def mlp(event):
                return event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(
                model, tokens, attention, mlp, require_production=False
            )
            calls += 1
            if tuple(logits.shape) != (len(records), int(batch["padding_length"]), VOCABULARY_SIZE):
                raise RuntimeError("R591 full-logit shape changed")
            for row_index, endpoint in enumerate(records):
                endpoint_id = str(endpoint["endpoint_id"])
                query = int(endpoint["final_position"])
                vector = logits[row_index, query].float().detach().cpu().contiguous()
                if not bool(torch.isfinite(vector).all()):
                    raise RuntimeError("R591 observed nonfinite logits")
                if endpoint_id in logits_by_endpoint:
                    raise RuntimeError("R591 duplicate endpoint within condition")
                logits_by_endpoint[endpoint_id] = vector
                locations[endpoint_id] = {
                    "endpoint_id": endpoint_id,
                    "length": int(endpoint["length"]),
                    "batch_index": int(batch["batch_index"]),
                    "row_index": row_index,
                }
    return {
        "logits": logits_by_endpoint,
        "locations": locations,
        "local_exactness": local,
        "calls": calls,
    }


def difference_summary(left: Mapping[str, object], right: Mapping[str, object], *, torch):
    left_logits, right_logits = left["logits"], right["logits"]
    if set(left_logits) != set(right_logits):
        raise RuntimeError("paired diagnostic endpoint membership changed")
    maximum = -1.0
    maximum_rms = 0.0
    argmax = None
    exceeding = 0
    for endpoint_id in sorted(left_logits):
        difference = left_logits[endpoint_id] - right_logits[endpoint_id]
        if not bool(torch.isfinite(difference).all()):
            raise RuntimeError("nonfinite paired difference")
        endpoint_max, token = torch.max(torch.abs(difference), dim=0)
        endpoint_value = float(endpoint_max)
        endpoint_rms = float(torch.sqrt(torch.mean(difference.double().square())))
        maximum_rms = max(maximum_rms, endpoint_rms)
        if endpoint_value > TOLERANCE:
            exceeding += 1
        if endpoint_value > maximum:
            maximum = endpoint_value
            location = dict(left["locations"][endpoint_id])
            argmax = {**location, "token_id": int(token)}
    total = len(left_logits)
    if total == 0 or argmax is None:
        raise RuntimeError("empty paired diagnostic")
    return {
        "max_abs": maximum,
        "max_rms": maximum_rms,
        "argmax": argmax,
        "endpoints_over_1e_5": exceeding,
        "fraction_endpoints_over_1e_5": exceeding / total,
        "endpoint_count": total,
    }


def decomposition_residual_summary(rm, nm, nl, *, torch):
    identifiers = set(rm["logits"])
    if identifiers != set(nm["logits"]) or identifiers != set(nl["logits"]):
        raise RuntimeError("full-FIT decomposition membership changed")
    maximum = -1.0
    argmax = None
    for endpoint_id in sorted(identifiers):
        total = rm["logits"][endpoint_id] - nl["logits"][endpoint_id]
        hook = rm["logits"][endpoint_id] - nm["logits"][endpoint_id]
        batch_pad = nm["logits"][endpoint_id] - nl["logits"][endpoint_id]
        residual = total - (hook + batch_pad)
        value, token = torch.max(torch.abs(residual), dim=0)
        numeric = float(value)
        if numeric > maximum:
            maximum = numeric
            argmax = {
                **rm["locations"][endpoint_id], "token_id": int(token)
            }
    if argmax is None or maximum > TOLERANCE:
        raise RuntimeError("full-FIT vector decomposition identity failed")
    return {"max_abs": maximum, "argmax": argmax, "endpoint_count": len(identifiers)}


def interpret(comparisons: Mapping[str, object]) -> dict[str, object]:
    panel = comparisons["panel"]
    observer = any(
        panel["observer"][schedule]["max_abs"] > TOLERANCE
        for schedule in PANEL_SCHEDULES
    )
    hook = comparisons["full_fit"]["hook"]["max_abs"] > TOLERANCE or any(
        panel["hook"][schedule]["max_abs"] > TOLERANCE
        for schedule in PANEL_SCHEDULES
    )
    padding = any(
        panel["padding"][dispatcher]["max_abs"] > TOLERANCE
        for dispatcher in DISPATCHERS
    )
    membership = any(
        panel["membership"][dispatcher]["max_abs"] > TOLERANCE
        for dispatcher in DISPATCHERS
    )
    active = [
        label for label, value in (
            ("observer_failure", observer), ("hook", hook),
            ("padding", padding), ("membership_gemm", membership),
        ) if value
    ]
    if len(active) > 1:
        classification = "mixed"
    elif observer:
        classification = "observer_failure"
    elif hook:
        classification = "hook_dominated"
    elif padding:
        classification = "padding_dominated"
    elif membership:
        classification = "membership_gemm_dominated"
    elif comparisons["full_fit"]["total"]["max_abs"] > TOLERANCE:
        classification = "missing_factor"
    else:
        classification = "all_registered_components_within_threshold"
    return {
        "classification": classification,
        "active_components": active,
        "threshold_unchanged": TOLERANCE,
        "licenses_r585_science": False,
    }


def _load_runtime():
    for path in (ROOT, OPS, POLY):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import torch
    import torch.nn.functional as functional
    facade = load_module(FACADE, "r591_pinned_facade")
    induction = load_module(INDUCTION, "r591_pinned_induction")
    r585 = load_module(R585, "r591_pinned_r585_runtime")
    return torch, functional, facade, induction, r585


def run_diagnostic() -> dict[str, object]:
    sources = verify_sources()
    _, execution = load_authority()
    support = audit_equality_support(execution["endpoints"])
    panel = select_panel(execution["endpoints"])
    full_schedules = full_fit_schedules(execution["endpoints"])
    controlled = panel_schedules(panel)
    manifest = build_call_manifest(execution["endpoints"])
    torch, functional, facade, induction, r585 = _load_runtime()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True
    )
    facade.validate_production_model(model)
    if checkpoint.weights_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash changed")

    common = {
        "torch": torch, "functional": functional, "facade": facade,
        "induction": induction, "r585": r585,
    }
    rm = collect_condition(model, full_schedules["M"], "R", **common)
    nm = collect_condition(model, full_schedules["M"], "N", **common)
    nl = collect_condition(model, full_schedules["L"], "N", **common)
    comparisons = {
        "full_fit": {
            "total": difference_summary(rm, nl, torch=torch),
            "hook": difference_summary(rm, nm, torch=torch),
            "batch_plus_padding": difference_summary(nm, nl, torch=torch),
            "decomposition_residual": decomposition_residual_summary(
                rm, nm, nl, torch=torch
            ),
        },
        "panel": {"padding": {}, "membership": {}, "observer": {}, "hook": {}},
    }
    local_exactness = {"full_fit:R_M": rm["local_exactness"]}
    del rm, nm, nl

    cells = {}
    for dispatcher in DISPATCHERS:
        for schedule in PANEL_SCHEDULES:
            key = (dispatcher, schedule)
            cells[key] = collect_condition(
                model, controlled[schedule], dispatcher, **common
            )
            if dispatcher in ("F", "R"):
                local_exactness[f"panel:{dispatcher}_{schedule}"] = cells[key][
                    "local_exactness"
                ]
    for dispatcher in DISPATCHERS:
        comparisons["panel"]["padding"][dispatcher] = difference_summary(
            cells[(dispatcher, "L_30")], cells[(dispatcher, "L_native")], torch=torch
        )
        comparisons["panel"]["membership"][dispatcher] = difference_summary(
            cells[(dispatcher, "M_30")], cells[(dispatcher, "L_30")], torch=torch
        )
    for schedule in PANEL_SCHEDULES:
        comparisons["panel"]["observer"][schedule] = difference_summary(
            cells[("F", schedule)], cells[("N", schedule)], torch=torch
        )
        comparisons["panel"]["hook"][schedule] = difference_summary(
            cells[("R", schedule)], cells[("N", schedule)], torch=torch
        )

    realized_calls = 54 + 54 + 54 + sum(
        cells[(dispatcher, schedule)]["calls"]
        for dispatcher in DISPATCHERS for schedule in PANEL_SCHEDULES
    )
    if realized_calls != 234:
        raise RuntimeError("realized R591 forward count changed")
    payload = {
        "schema": SCHEMA,
        "rung": 591,
        "status": STATUS,
        "registered_tolerance": TOLERANCE,
        "r585_commit": R585_COMMIT,
        "source_sha256": sources,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "panel": build_dryrun()["panel"],
        "equality_support_audit": support,
        "forward_call_shape_contract": forward_shape_contract(manifest),
        "operation_census": operation_census(manifest),
        "comparisons": comparisons,
        "local_exactness": local_exactness,
        "interpretation": interpret(comparisons),
        "model_forwards": realized_calls,
        "model_backwards": 0,
        "model_weights_updated": False,
        "evaluated_splits": ["FIT"],
        "forbidden_splits_opened": [],
    }
    require_finite_json(payload)
    return payload


def main() -> None:
    mode = os.environ.get("BQLIB_DRYRUN")
    if mode == "1":
        payload = build_dryrun()
    elif mode is None:
        payload = run_diagnostic()
    else:
        raise RuntimeError("BQLIB_DRYRUN must be absent or exactly '1'")
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
