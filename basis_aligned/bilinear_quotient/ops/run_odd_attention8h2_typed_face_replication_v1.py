#!/usr/bin/env python3
# BQGATE:1152bodyforwards;144prefixes;300seconds;no fitting.
"""pred_a instrument; pred_b frozen OOD; pred_c composition.
pred_d reusable typed face; pred_e unrelated-token selectivity.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import numpy as np
import torch

import sparse_interaction_graph as sparse
from odd_attention8h2_sparse_lattice_runtime import READOUTS, measure_lattice
from regional_cue_row_check_v1 import validate
from run_even_value_factorial_native_v1 import setup, cpu_control
from sparse_path_stability_atlas_v1 import digest


STEM = "ODD_ATTENTION8H2_TYPED_FACE_REPLICATION_V1"
PORTS = ("routing", "current_value", "inherited_value")


def row_hash(row):
    copied = dict(row)
    claimed = copied.pop("row_sha256")
    payload = json.dumps(copied, sort_keys=True, separators=(",", ":")).encode()
    return claimed, hashlib.sha256(payload).hexdigest()


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(path) == value for path, value in binding.items())
    discovery = json.loads((P / "ODD_ATTENTION8H2_CHAIN_FRESH_V1_ROWS.json").read_text())["rows"]
    replication_doc = json.loads((P / (STEM + "_ROWS.json")).read_text())
    replication = replication_doc["rows"]
    prior_doc = json.loads((P / "ODD_ATTENTION8H2_SPARSE_GRAPH_V1_ROWS.json").read_text())
    assert len(discovery) == 48 and len(replication) == 96
    validate(discovery)
    assert all(row_hash(row)[0] == row_hash(row)[1] for row in replication)
    prior_sources = {item["source_row"] for item in prior_doc["contexts"]}
    replication_sources = {item["source_row"] for item in replication_doc["contexts"]}
    assert not (prior_sources & replication_sources)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        control = json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())
        assert control["pred_a"] and control["prior_source_overlap"] == 0
        assert sparse.face_masks((0, 1), 3) == (1, 2, 3)
        assert sparse.face_masks((0, 2), 3) == (1, 4, 5)
        print("1152bodyforwards;144prefixes;8corners;2typedfaces;CPUcontrol", cpu_control())
        return

    output = P / (STEM + "_RESULT.json")
    artifact = P / (STEM + "_ARTIFACT.pt")
    assert not output.exists() and not artifact.exists()
    started = time.perf_counter()
    signal.alarm(300)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    model = load_model_fast().cuda().eval()
    graph, _, _, _ = setup("cuda")
    measured = measure_lattice(model, graph, discovery, replication)
    values = measured["values"]
    assert measured["body_forwards"] == 1152 and bool(torch.isfinite(values).all())
    effects = values - values[0:1]
    target_corners = {corner: effects[corner, :, 0].numpy() for corner in range(8)}
    dividends = sparse.full_dividends(target_corners, 3)
    atoms = {mask: dividends[mask] for mask in range(1, 8)}
    target = target_corners[7]
    discovery_indices = tuple(range(48))
    replication_indices = tuple(range(48, 144))
    candidate_faces = (
        sparse.face_masks((0, 1), 3),
        sparse.face_masks((0, 2), 3),
    )
    selected, face_records = sparse.select_graph_family(
        target, atoms, discovery_indices, candidate_faces
    )
    reports = sparse.evaluate_frozen(
        target, atoms, selected,
        {"discovery": discovery_indices, "replication_ood": replication_indices},
    )
    prediction = sparse.compose(atoms, selected)
    closure_error = float(
        np.linalg.norm(sparse.reconstruct(dividends, 7) - target)
        / max(np.linalg.norm(target), 1e-30)
    )
    sign_reversals = int(np.sum(
        (prediction[list(replication_indices)] * target[list(replication_indices)] < 0)
        & (np.abs(target[list(replication_indices)]) >= 1e-5)
    ))
    composition_cells = {}
    for cue in ("British", "American"):
        indices = [48 + index for index, row in enumerate(replication) if row["cue"] == cue]
        composition_cells["cue_" + cue.lower()] = sparse.metrics(target, prediction, indices)
    for endpoint in range(6):
        indices = [48 + index for index, row in enumerate(replication) if row["endpoint"] == endpoint]
        composition_cells[f"endpoint_{endpoint}"] = sparse.metrics(target, prediction, indices)
    predicted_readouts = []
    for readout in range(len(READOUTS)):
        corners = {corner: effects[corner, :, readout].numpy() for corner in range(8)}
        terms = sparse.full_dividends(corners, 3)
        predicted_readouts.append(sparse.compose(terms, selected))
    predicted_readouts = np.stack(predicted_readouts, axis=-1)
    target_rms = float(np.sqrt(np.mean(predicted_readouts[list(replication_indices), 0] ** 2)))
    control_ratios = {
        READOUTS[index][0]: float(
            np.sqrt(np.mean(predicted_readouts[list(replication_indices), index] ** 2))
            / max(target_rms, 1e-30)
        )
        for index in range(1, len(READOUTS))
    }
    required = sparse.required_corners(selected, 3)
    selected_rms = {
        str(mask): float(np.sqrt(np.mean(atoms[mask][list(replication_indices)] ** 2)))
        for mask in selected
    }
    instrument = (
        max(measured["source_replays"]) <= 1e-5
        and max(measured["factorial_errors"]) <= 1e-5
        and max(measured["value_errors"]) <= 1e-5
        and max(measured["full_channel_errors"]) <= 1e-5
        and closure_error <= 1e-5
        and max(measured["outside"]) == 0
        and all(min(norms) >= 1e-8 for norms in measured["write_norms"].values())
        and not (prior_sources & replication_sources)
    )
    pred_b = (
        reports["replication_ood"]["relative_l2"] <= .35
        and reports["replication_ood"]["cosine"] >= .9
        and sign_reversals == 0
    )
    pred_c = all(cell["relative_l2"] <= .35 for cell in composition_cells.values())
    pred_d = (
        reports["discovery"]["relative_l2"] <= .30
        and len(required) <= 4
        and min(selected_rms.values()) >= 1e-8
    )
    pred_e = max(control_ratios.values()) <= .5
    torch.save({
        "values": values, "effects": effects, "ports": PORTS,
        "candidate_faces": candidate_faces, "selected_masks": selected,
        "required_corners": required,
    }, artifact)
    result = {
        "terminal": "valid_odd_attention8h2_typed_face_replication" if all(
            (instrument, pred_b, pred_c, pred_d, pred_e)
        ) else "valid_odd_attention8h2_typed_face_replication_near_miss",
        "pred_a": instrument, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e,
        "ports": list(PORTS), "candidate_faces": [list(face) for face in candidate_faces],
        "face_discovery_records": [
            {"masks": list(record["masks"]), "relative_l2": record["relative_l2"]}
            for record in face_records
        ],
        "selected_masks": list(selected), "required_corners": list(required),
        "metrics": reports, "composition_cells": composition_cells,
        "material_replication_sign_reversals": sign_reversals,
        "selected_atom_replication_rms": selected_rms,
        "target_replication_rms": target_rms, "control_ratios": control_ratios,
        "max_head_source_replay_error": max(measured["source_replays"]),
        "max_factorial_error": max(measured["factorial_errors"]),
        "max_value_sum_error": max(measured["value_errors"]),
        "max_full_corner_channel_error": max(measured["full_channel_errors"]),
        "mobius_closure_error": closure_error,
        "max_outside_hybrid_delta": max(measured["outside"]),
        "prior_source_overlap": len(prior_sources & replication_sources),
        "body_forwards": measured["body_forwards"], "fit_parameters": 0,
        "model_updates": 0, "seconds": time.perf_counter() - started,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "scope": (
            "Typed routing-times-value face selected on authored discovery and frozen before a second "
            "source-disjoint skip39000 FineWeb panel. Conditional native-suffix intervention; package "
            "extraction and equal-norm removal are separate remaining gates."
        ),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_shas"}, indent=2))
    signal.alarm(0)


if __name__ == "__main__":
    main()
