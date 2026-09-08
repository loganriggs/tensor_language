#!/usr/bin/env python3
"""Test singleton-ranked cumulative downstream module clamps for the is-was bypass."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_writer_replay_module_capture_finiteness_and_exact_price pred_b_original_fit_selects_a_qualified_cumulative_module_prefix pred_c_selected_prefix_validates_without_reselection_on_holdout pred_d_selected_prefix_prunes_at_least_one_module pred_e_selected_prefix_is_temporally_selective
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2 as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_downstream_module_singleton_ordered_greedy_v1.json"
ATLAS_RESULT = ROOT / "circuits/followups/temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2_result.json"
ATLAS_RUNNER = ROOT / "ops/run_temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2.py"
GREEDY_RESULT = atlas.GREEDY_RESULT
OUT = ROOT / "circuits/followups/temporal_iswas_downstream_module_singleton_ordered_greedy_v1_result.json"
EXPECTED = {
    "prior": "07105aa3a21f3de9cf167690961b5ab507bf8a9e86fe6d6606fe2c89dc62f6fa",
    "atlas_result": "88537c2561098239472072b277ab29bef868cc1d648d672024f306fed814990b",
    "atlas_runner": "719394cd2fd134a02afe0365b6603bff4ae3882b5ddd5f2b586930e41b6f3902",
    "greedy_result": "6a2ae6598ae95269f7d9e24e13196155342f690d331c78efed5115a8cf84bda2",
}
ORDER = ("A11", "M11", "M12", "M15", "M13", "M16", "M10", "A15",
         "M14", "A10", "A13", "A17", "A16", "A14", "A12", "M17")
PRICE = {"checkpoint_loads": 1, "model_forwards": 19,
         "sequence_evaluations": 2432, "scored_token_positions": 4864,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_writer_replay_module_capture_finiteness_and_exact_price",
    "pred_b_original_fit_selects_a_qualified_cumulative_module_prefix",
    "pred_c_selected_prefix_validates_without_reselection_on_holdout",
    "pred_d_selected_prefix_prunes_at_least_one_module",
    "pred_e_selected_prefix_is_temporally_selective",
)
BARS = {"replay": 1e-5, "fit_recovery_min": .5, "fit_recovery_max": 1.5,
        "fit_cosine": .9, "fit_residual": .65, "fit_direction": .9,
        "holdout_recovery_min": .4, "holdout_recovery_max": 1.75,
        "holdout_cosine": .85, "holdout_residual": .75,
        "holdout_direction": .8, "collateral": .01}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_prefix(backend, tokens, names, replacements, position_rows):
    calls = {name: 0 for name in names}
    handles = []

    def hook(name):
        def replace(_module, _inputs, output):
            calls[name] += 1
            return atlas.replace_positions(output, replacements[name], position_rows)
        return replace

    targets = atlas.module_targets(backend.model)
    for name in names:
        handles.append(targets[name].register_forward_hook(hook(name)))
    try:
        logits, _captures = atlas.mediation.parent._forward(backend, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if set(calls.values()) != {1}:
        raise RuntimeError("cumulative module patch coverage changed")
    return logits, calls


def qualified(row, phase):
    if phase == "FIT":
        return bool(BARS["fit_recovery_min"] <= row["signed_recovery"]
            <= BARS["fit_recovery_max"] and row["cosine"] >= BARS["fit_cosine"]
            and row["relative_residual"] <= BARS["fit_residual"]
            and row["direction_agreement"] >= BARS["fit_direction"])
    return bool(BARS["holdout_recovery_min"] <= row["signed_recovery"]
        <= BARS["holdout_recovery_max"] and row["cosine"] >= BARS["holdout_cosine"]
        and row["relative_residual"] <= BARS["holdout_residual"]
        and row["direction_agreement"] >= BARS["holdout_direction"])


def main():
    paths = {"prior": PRIOR, "atlas_result": ATLAS_RESULT,
             "atlas_runner": ATLAS_RUNNER, "greedy_result": GREEDY_RESULT}
    observed = {name: sha(path) for name, path in paths.items()}
    prior, atlas_result = json.loads(PRIOR.read_text()), json.loads(ATLAS_RESULT.read_text())
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": observed == EXPECTED, "order": list(ORDER), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    greedy = json.loads(GREEDY_RESULT.read_text())
    authority_ok = bool(observed == EXPECTED and prior["price"] == PRICE
        and prior["locked_module_order"] == list(ORDER)
        and atlas_result["terminal"] == "distributed_or_unstable_bypass"
        and atlas_result["eligible_fit_modules"] == []
        and atlas_result["selected_modules"] == []
        and atlas_result["selected_writer_heads"] == prior["locked_writer_heads"]
        and tuple(atlas_result["predictions"].values()) == (True, True, False, False, False))
    if not authority_ok:
        raise RuntimeError("module-prefix authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    mediation, loc, parent = atlas.mediation, atlas.mediation.parent.loc, atlas.mediation.parent
    backend = loc.producer.Bilin18TorchBackend.load("cuda")
    rows = loc.original.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    role, other = "iswas", "temporal"
    pairs = loc.parent.pair_indices(endpoints, lookup, role)
    other_pairs = loc.parent.pair_indices(endpoints, lookup, other)
    queries = [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
    regions = [loc.source_regions(endpoint["ids"], endpoints[pairs[index]][2]["ids"],
                                  queries[index])
               for index, (_row, _cell, endpoint) in enumerate(endpoints)]
    source_rows = [item[parent.LOCKED[role]] for item in regions]
    position_rows = [list(range(query + 1)) for query in queries]
    selected_heads = tuple(greedy["selected_prefixes"][role]["heads"])
    with torch.no_grad():
        native_logits, head_captures, native_l11, native_modules, native_calls = \
            atlas.capture_modules(backend, tokens)
        writer_logits, _unused, _writer_l11, writer_modules, writer_calls = \
            atlas.capture_modules(backend, tokens, head_captures=head_captures,
                selected_labels=selected_heads, pairs=pairs, position_rows=source_rows)
        reference_logits, _ = mediation.source._forward(
            backend, tokens, donor_v=native_l11["v"], pairs=pairs,
            value_positions=source_rows)
        prefix_logits, prefix_calls = {}, {}
        for index in range(1, len(ORDER) + 1):
            arm = f"P{index}"
            prefix_logits[arm], prefix_calls[arm] = patch_prefix(
                backend, tokens, ORDER[:index], writer_modules, position_rows)
    native = {name: loc.margins(native_logits, endpoints, name) for name in parent.LOCKED}
    writer_effect = loc.margins(writer_logits, endpoints, role) - native[role]
    writer_other = loc.margins(writer_logits, endpoints, other) - native[other]
    reference_effect = loc.margins(reference_logits, endpoints, role) - native[role]
    gold = native[role][pairs] - native[role]
    other_gold = native[other][other_pairs] - native[other]
    greedy_reports = greedy["panels"]["original"]["reports"]
    reports, replay_errors = [], []
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        replay = mediation.accounting.effect_metrics(
            writer_effect[chosen], reference_effect[chosen], writer_other[chosen])
        replay_gold = mediation.accounting.effect_metrics(
            writer_effect[chosen], gold[chosen], writer_other[chosen])
        old = next(row for row in greedy_reports if row["role"] == role
            and row["phase"] == phase and row["template_id"] == "ALL"
            and row["arm"] == greedy["selected_prefixes"][role]["arm"])
        for field in ("signed_recovery", "cosine", "relative_residual",
                      "direction_agreement", "non_target_to_target_gold_norm"):
            replay_errors.append(abs(replay[field] - old[field]))
        replay_errors.append(abs(replay_gold["non_target_to_target_gold_norm"]
                                 - old["command_gold_non_target_norm_ratio"]))
        for index in range(1, len(ORDER) + 1):
            arm = f"P{index}"
            effect = loc.margins(prefix_logits[arm], endpoints, role) - native[role]
            effect_other = loc.margins(prefix_logits[arm], endpoints, other) - native[other]
            metric = mediation.accounting.effect_metrics(
                effect[chosen], writer_effect[chosen], effect_other[chosen])
            temporal_norm = float(np.linalg.norm(other_gold[chosen]))
            reports.append({"arm": arm, "length": index, "modules": list(ORDER[:index]),
                "phase": phase, **metric, "temporal_command_gold_collateral":
                float(np.linalg.norm(effect_other[chosen]) / max(temporal_norm, 1e-30))})
    fit = {row["arm"]: row for row in reports if row["phase"] == "FIT"}
    holdout = {row["arm"]: row for row in reports if row["phase"] == "HOLDOUT"}
    selected_arm = next((f"P{index}" for index in range(1, len(ORDER) + 1)
                         if qualified(fit[f"P{index}"], "FIT")), None)
    selected = fit.get(selected_arm) if selected_arm else None
    atlas_a11 = {row["phase"]: row for row in atlas_result["reports"]
                 if row["module"] == "A11"}
    p1_replay = max(abs(fit["P1"][field] - atlas_a11["FIT"][field])
                    for field in ("signed_recovery", "cosine", "relative_residual",
                                  "direction_agreement"))
    p1_replay = max(p1_replay, *(abs(holdout["P1"][field]
        - atlas_a11["HOLDOUT"][field]) for field in
        ("signed_recovery", "cosine", "relative_residual", "direction_agreement")))
    A = bool(authority_ok and max(replay_errors) <= BARS["replay"]
        and p1_replay <= BARS["replay"] and set(native_calls.values()) == {1}
        and set(writer_calls.values()) == {1}
        and all(set(calls.values()) == {1} for calls in prefix_calls.values())
        and PRICE == prior["price"]
        and all(np.isfinite(value) for row in reports for value in row.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)))
    B = selected_arm is not None
    C = bool(selected_arm and qualified(holdout[selected_arm], "HOLDOUT"))
    D = bool(selected and selected["length"] < len(ORDER))
    E = bool(selected_arm and fit[selected_arm]["temporal_command_gold_collateral"]
        <= BARS["collateral"] and holdout[selected_arm]["temporal_command_gold_collateral"]
        <= BARS["collateral"])
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not A else "compact_distributed_module_prefix"
                if all((B, C, D, E)) else "no_stable_compact_module_prefix")
    result = {"schema": "temporal_iswas_downstream_module_singleton_ordered_greedy_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": observed, "selected_writer_heads": list(selected_heads),
        "order": list(ORDER), "reports": reports, "selected_arm": selected_arm,
        "selected_prefix": selected, "writer_replay_max_abs_error": max(replay_errors),
        "p1_singleton_replay_max_abs_error": p1_replay,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"selected_arm": selected_arm, "selected_prefix": selected,
        "fit": fit, "holdout": holdout,
        "writer_replay_max_abs_error": max(replay_errors),
        "p1_singleton_replay_max_abs_error": p1_replay,
        "predictions": predictions, "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
