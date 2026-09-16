"""Model-free positive/negative red-team of the prospective odd sparse graph."""
from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import sparse_interaction_graph as sparse


HERE = Path(__file__).resolve().parent
STEM = "ODD_ATTENTION8H2_SPARSE_GRAPH_V1"
ARTIFACT = HERE / (STEM + "_ARTIFACT.pt")
RESULT = HERE / (STEM + "_RESULT.json")
ROWS = HERE / (STEM + "_ROWS.json")
OUTPUT = HERE / (STEM + "_AUDIT.json")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result = json.loads(RESULT.read_text())
    natural = json.loads(ROWS.read_text())["rows"]
    artifact = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    effects = artifact["effects"].double().numpy()
    assert effects.shape == (8, 144, 5)
    discovery = tuple(range(48))
    ood = tuple(range(48, 144))
    corners = {corner: effects[corner, :, 0] for corner in range(8)}
    dividends = sparse.full_dividends(corners, 3)
    atoms = {mask: dividends[mask] for mask in range(1, 8)}
    target = corners[7]
    replay_selected, replay_curve = sparse.greedy_select(target, atoms, discovery, steps=3)
    registered = tuple(result["selected_masks"])
    assert replay_selected == registered
    replay = sparse.evaluate_frozen(
        target, atoms, registered, {"discovery": discovery, "fresh_natural_ood": ood}
    )

    combinations = []
    for masks in itertools.combinations(range(1, 8), 3):
        prediction = sparse.compose(atoms, masks)
        cue_errors = {}
        for cue in ("British", "American"):
            indices = [48 + index for index, row in enumerate(natural) if row["cue"] == cue]
            cue_errors[cue.lower()] = sparse.metrics(target, prediction, indices)["relative_l2"]
        combinations.append({
            "masks": list(masks),
            "discovery_relative_l2": sparse.metrics(target, prediction, discovery)["relative_l2"],
            "ood_relative_l2": sparse.metrics(target, prediction, ood)["relative_l2"],
            "maximum_cue_relative_l2": max(cue_errors.values()),
            "cue_relative_l2": cue_errors,
            "required_corners": list(sparse.required_corners(masks, 3)),
        })
    by_discovery = sorted(combinations, key=lambda item: (item["discovery_relative_l2"], item["masks"]))
    by_ood = sorted(combinations, key=lambda item: (item["ood_relative_l2"], item["masks"]))
    by_ood_minimax = sorted(combinations, key=lambda item: (item["maximum_cue_relative_l2"], item["masks"]))
    registered_record = next(item for item in combinations if tuple(item["masks"]) == tuple(sorted(registered)))

    atom_diagnostics = {}
    for mask, atom in atoms.items():
        atom_diagnostics[str(mask)] = {
            "discovery_rms": float(np.sqrt(np.mean(atom[list(discovery)] ** 2))),
            "ood_rms": float(np.sqrt(np.mean(atom[list(ood)] ** 2))),
            "discovery_cosine_with_target": sparse.metrics(target, atom, discovery)["cosine"],
            "ood_cosine_with_target": sparse.metrics(target, atom, ood)["cosine"],
        }

    selected_prediction = sparse.compose(atoms, registered)
    exact_registered_metrics = {
        "discovery": sparse.metrics(target, selected_prediction, discovery),
        "fresh_natural_ood": sparse.metrics(target, selected_prediction, ood),
    }
    reported_metric_error = max(
        abs(exact_registered_metrics[panel][metric] - result["metrics"][panel][metric])
        for panel in exact_registered_metrics
        for metric in ("target_rms", "prediction_rms", "relative_l2", "cosine", "aligned_recovery")
    )
    registered_discovery_rank = 1 + next(
        index for index, item in enumerate(by_discovery)
        if tuple(item["masks"]) == tuple(sorted(registered))
    )
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": "valid_odd_attention8h2_sparse_graph_selection_near_miss",
        "checks": {
            "generic_greedy_replay_exact": replay_selected == registered,
            "selection_curve_replay_max_abs": float(max(
                abs(left - right) for left, right in zip(replay_curve, result["selection_curve"])
            )),
            "reported_metric_replay_max_abs": reported_metric_error,
            "registered_is_global_discovery_optimum_among_three_atom_sets": registered_discovery_rank == 1,
            "registered_pred_c_correctly_false": (
                not result["pred_c"] and registered_record["maximum_cue_relative_l2"] > .35
            ),
        },
        "registered": registered_record,
        "registered_discovery_rank_of_35": registered_discovery_rank,
        "best_discovery_three_atom_graphs": by_discovery[:5],
        "posthoc_ood_oracle_best": by_ood[0],
        "posthoc_ood_minimax_best": by_ood_minimax[0],
        "atom_diagnostics": atom_diagnostics,
        "interpretation": (
            "No implementation rescue: the registered masks replay exactly and are the exhaustive discovery optimum. "
            "The 35.318% British-cell error is a genuine preregistered miss. OOD-oracle alternatives are diagnostic "
            "only and may not replace the frozen graph."
        ),
        "authority_sha256": {
            "artifact": digest(ARTIFACT),
            "result": digest(RESULT),
            "rows": digest(ROWS),
            "sparse_interaction_graph": digest(HERE / "sparse_interaction_graph.py"),
        },
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
