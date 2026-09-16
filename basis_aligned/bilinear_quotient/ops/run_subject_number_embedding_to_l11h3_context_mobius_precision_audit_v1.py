#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_batch_replay pred_b_batch_drift_explains_failure pred_c_decomposition_stable pred_d_material_interaction_precision_qualified
"""Precision audit for the failed crossed context-Möbius replay gate."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_v1 as authority
import circuit_fast_screen_candidate_subject_number_embedding_removal_fresh as removal
import circuit_fast_screen_managed_runner as managed
import run_subject_number_embedding_to_l11h3_context_mobius_v1 as crossed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
PRIOR = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_SCALAR_DISCOVERY_V1_RESULT.json"
CROSSED = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_PRECISION_AUDIT_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_PRECISION_AUDIT_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_PRECISION_AUDIT_V1_RESULT.json"
LAYER, HEAD, HEAD_WIDTH = 11, 3, 128
BARS = {"maximum_decomposition_metric_drift": 1e-4,
        "maximum_geometry_error": 1e-5,
        "maximum_exact_batch_replay_error": 1e-4,
        "maximum_synthetic_error": 1e-12,
        "minimum_batch_alpha_drift_rms": 1e-6,
        "minimum_interaction_rms": 5.,
        "minimum_interaction_to_replay_ratio": 1e5}
PRICE = {"partial_forwards": 4, "sequences": 256, "behavior_logits": 0,
         "backwards": 0, "fits": 0, "permutations": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_batch_replay": None,
                       "pred_b_batch_drift_explains_failure": None,
                       "pred_c_decomposition_stable": None,
                       "pred_d_material_interaction_precision_qualified": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "removal_authority": Path(removal.__file__),
             "decoder": DECODER, "rank1": RANK1, "prior": PRIOR,
             "crossed_result": CROSSED, "crossed_runner": Path(crossed.__file__),
             "preregistration": PREREG}
    if binding["files"] != {name: sha(path) for name, path in paths.items()} \
            or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, rank, prior, first = (json.loads(path.read_text())
                                   for path in (DECODER, RANK1, PRIOR, CROSSED))
    if decoder["terminal"] != "embedding_number_decoder_frozen" \
            or rank["terminal"] != "rank1_frozen_weights_only" \
            or prior["terminal"] != "embedding_to_l11h3_scalar_null" \
            or first["terminal"] != "invalid" \
            or first["runner_sha256"] != sha(Path(crossed.__file__)):
        raise ValueError("parent status or runner changed")
    rows = authority.build_rows()
    return binding, decoder, rank, prior, first, rows


def plan():
    _, decoder, rank, prior, first, rows = load_bound()
    return {"schema": "subject_number_embedding_to_l11h3_context_mobius_precision_audit_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "batch_sizes": [64, 64],
            "parent_terminal": first["terminal"], "prior_terminal": prior["terminal"],
            "rank": rank["rank"],
            "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"],
            "authority_sha256": authority.canonical(rows), "bars": BARS,
            "price": PRICE, "binding_sha256": sha(BINDING)}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(600)
    binding, decoder, rank, prior, first, rows = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"],
                                dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"])
    unit_decoder = decoder_axis / decoder_axis.norm()
    writer_axis = torch.tensor(rank["axis"], dtype=torch.float32, device=device)
    writer_axis /= writer_axis.norm()
    head_slice = slice(HEAD * HEAD_WIDTH, (HEAD + 1) * HEAD_WIDTH)
    counts = {"partial_forwards": 0, "sequences": 0, "behavior_logits": 0,
              "backwards": 0, "fits": 0, "permutations": 0,
              "parameter_updates": 0}
    geometry_errors = []

    def batch_alpha(batch_rows):
        tokens = torch.tensor([row["token_ids"] for row in batch_rows],
                              dtype=torch.long, device=device)
        positions = torch.tensor([row["subject_position"] for row in batch_rows],
                                 dtype=torch.long, device=device)
        batch = torch.arange(len(batch_rows), device=device)
        with torch.no_grad():
            base_input = F.rms_norm(model.transformer.wte(tokens),
                                    (model.config.n_embd,)).float()
        subject = base_input[batch, positions].double()
        projection = subject @ unit_decoder
        target_projection = threshold / float(decoder_axis.norm())
        orthogonal = subject - projection[:, None] * unit_decoder
        scale = torch.sqrt((subject.square().sum(1) - target_projection ** 2)
                           / orthogonal.square().sum(1))
        removed_subject = target_projection * unit_decoder + scale[:, None] * orthogonal
        removed_input = base_input.clone(); removed_input[batch, positions] = removed_subject.float()
        geometry_errors.extend([
            float((removed_subject @ decoder_axis - threshold).abs().max()),
            float(((removed_subject.norm(dim=1) - subject.norm(dim=1)).abs()
                   / subject.norm(dim=1)).max())])

        def capture(initial):
            counts["partial_forwards"] += 1; counts["sequences"] += len(batch_rows)
            found = {}
            with torch.no_grad():
                x = initial; x0 = initial; first_value = None
                for layer, block in enumerate(model.transformer.h):
                    x = block.lambdas[0] * x + block.lambdas[1] * x0
                    handle = None
                    if layer == LAYER:
                        def hook(_module, arguments):
                            found["head"] = arguments[0][batch, positions, head_slice].detach().clone()
                        handle = block.attn.c_proj.register_forward_pre_hook(hook)
                    try:
                        attention, first_value = block.attn(F.rms_norm(x, (x.shape[-1],)),
                                                             first_value)
                    finally:
                        if handle is not None: handle.remove()
                    if layer == LAYER: break
                    x = x + attention
                    x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
            return found["head"]

        base_head, removed_head = capture(base_input), capture(removed_input)
        weight = model.transformer.h[LAYER].attn.c_proj.weight[:, head_slice]
        return (F.linear(base_head - removed_head, weight) @ writer_axis).double().cpu().numpy()

    replay_ids = [i for i, row in enumerate(rows) if row["template_id"] in ("near", "behind")]
    new_ids = [i for i, row in enumerate(rows) if row["template_id"] in ("under", "above")]
    replay_alpha, new_alpha = batch_alpha([rows[i] for i in replay_ids]), batch_alpha([rows[i] for i in new_ids])
    alpha = np.empty(128, dtype=np.float64); alpha[replay_ids] = replay_alpha; alpha[new_ids] = new_alpha
    matrix = alpha.reshape(32, 4)
    prior_rows = removal.build_rows()
    prior_alpha = np.asarray([record["alpha"] for record in prior["records"]])
    prior_keys = [(row["pair_index"], row["number"], row["template_id"])
                  for row in prior_rows]
    replay_keys = [(rows[i]["pair_index"], rows[i]["number"], rows[i]["template_id"])
                   for i in replay_ids]
    if replay_keys != prior_keys: raise ValueError("replay order differs")
    replay_error = replay_alpha - prior_alpha
    replay_max = float(np.abs(replay_error).max())
    replay_rms = float(np.sqrt(np.mean(replay_error ** 2)))
    first_alpha = np.asarray([record["alpha"] for record in first["records"]])
    batch_drift = first_alpha - alpha
    batch_drift_rms = float(np.sqrt(np.mean(batch_drift ** 2)))
    batch_drift_max = float(np.abs(batch_drift).max())
    fixtures = crossed.positive_redteam()
    grand, subject, context, interaction = crossed.decompose(matrix)
    raw_relative = float(np.linalg.norm(interaction) / np.linalg.norm(matrix))
    singular_values = np.linalg.svd(interaction, compute_uv=False)
    rank1_energy = float(singular_values[0] ** 2 / np.sum(singular_values ** 2))
    interaction_rms = float(np.sqrt(np.mean(interaction ** 2)))
    replay_ratio = interaction_rms / max(replay_rms, 1e-30)
    raw_drift = abs(raw_relative - first["decomposition"]["additive_raw_relative_l2"])
    rank1_drift = abs(rank1_energy - first["decomposition"]["rank1_interaction_energy"])
    finite = bool(np.isfinite(np.asarray([*alpha, *singular_values])).all())
    pred_a = bool(finite and max(geometry_errors) <= BARS["maximum_geometry_error"]
                  and replay_max <= BARS["maximum_exact_batch_replay_error"]
                  and max(fixtures.values()) <= BARS["maximum_synthetic_error"]
                  and counts == PRICE
                  and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    pred_b = bool(pred_a and batch_drift_rms >= BARS["minimum_batch_alpha_drift_rms"])
    pred_c = bool(pred_b and max(raw_drift, rank1_drift)
                  <= BARS["maximum_decomposition_metric_drift"])
    pred_d = bool(pred_c and interaction_rms >= BARS["minimum_interaction_rms"]
                  and replay_ratio >= BARS["minimum_interaction_to_replay_ratio"])
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c, pred_d)))
    terminal = "context_mobius_precision_qualified" if pred_d else "precision_audit_null" if pred_a else "invalid"
    result = {"schema": "subject_number_embedding_to_l11h3_context_mobius_precision_audit_v1_result",
              "terminal": terminal, "predictions": predictions,
              "instrument": {"finite": finite, "maximum_geometry_error": max(geometry_errors),
                             "exact_batch_replay_max_abs_error": replay_max,
                             "exact_batch_replay_rms_error": replay_rms,
                             "positive_redteam": fixtures, "counts": counts},
              "batch_effect": {"v1_to_rebatched_alpha_rms": batch_drift_rms,
                               "v1_to_rebatched_alpha_max_abs": batch_drift_max},
              "decomposition": {"additive_raw_relative_l2": raw_relative,
                                "v1_metric_absolute_drift": raw_drift,
                                "interaction_rms": interaction_rms,
                                "interaction_to_exact_replay_rms_ratio": replay_ratio,
                                "interaction_singular_values": singular_values.tolist(),
                                "rank1_interaction_energy": rank1_energy,
                                "v1_rank1_energy_absolute_drift": rank1_drift},
              "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical(rows),
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "wall_seconds": time.perf_counter() - started,
              "scope": "Precision qualification of the invalid V1 replay instrument; no new circuit or OOD claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    assert pred_a


if __name__ == "__main__":
    main()
