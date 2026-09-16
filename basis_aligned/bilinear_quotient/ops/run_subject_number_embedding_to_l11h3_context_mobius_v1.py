#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_instrument_and_positive_redteam pred_b_additive_subject_context_sufficient pred_c_explicit_interaction_material pred_d_rank1_interaction_candidate
"""Crossed subject/context Möbius decomposition of the L11H3 response scalar."""
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
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
PRIOR = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_SCALAR_DISCOVERY_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_V1_RESULT.json"
LAYER, HEAD, HEAD_WIDTH = 11, 3, 128
NULLS, SEED = 256, 20260921
BARS = {
    "maximum_additive_raw_relative_l2": .15,
    "maximum_geometry_error": 1e-5,
    "maximum_overlap_alpha_error": 1e-4,
    "maximum_synthetic_error": 1e-12,
    "minimum_interaction_rms": 5.,
    "minimum_interaction_to_replay_ratio": 1e5,
    "minimum_rank1_interaction_energy": .70,
}
PRICE = {"partial_forwards": 2, "sequences": 256, "permutations": 256,
         "behavior_logits": 0, "backwards": 0, "fits": 0,
         "parameter_updates": 0}
PREDICTION_REGISTRY = {
    "pred_a_instrument_and_positive_redteam": None,
    "pred_b_additive_subject_context_sufficient": None,
    "pred_c_explicit_interaction_material": None,
    "pred_d_rank1_interaction_candidate": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decompose(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    grand = float(matrix.mean())
    subject = matrix.mean(axis=1, keepdims=True) - grand
    context = matrix.mean(axis=0, keepdims=True) - grand
    interaction = matrix - grand - subject - context
    return grand, subject, context, interaction


def positive_redteam():
    u = np.linspace(-2., 3., 32); v = np.asarray([-4., -1., 2., 7.])
    additive = 11. + u[:, None] + v[None, :]
    additive_error = float(np.abs(decompose(additive)[3]).max())
    left = np.sin(np.arange(32, dtype=np.float64)); left -= left.mean()
    right = np.asarray([-3., -1., 1., 3.]); right -= right.mean()
    known = np.outer(left, right)
    mixed = additive + known
    recovered = decompose(mixed)[3]
    interaction_error = float(np.abs(recovered - known).max())
    return {"additive_interaction_max_abs": additive_error,
            "rank1_interaction_recovery_max_abs": interaction_error}


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "removal_authority": Path(removal.__file__),
             "decoder": DECODER, "rank1": RANK1, "prior": PRIOR,
             "preregistration": PREREG}
    if binding["files"] != {name: sha(path) for name, path in paths.items()} \
            or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["nulls"] != NULLS \
            or binding["null_seed"] != SEED or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, rank, prior = (json.loads(path.read_text())
                            for path in (DECODER, RANK1, PRIOR))
    if decoder["terminal"] != "embedding_number_decoder_frozen" \
            or rank["terminal"] != "rank1_frozen_weights_only" \
            or prior["terminal"] != "embedding_to_l11h3_scalar_null":
        raise ValueError("parent status changed")
    rows = authority.build_rows()
    if authority.canonical(rows) != binding["authority_sha256"]:
        raise ValueError("authority changed")
    return binding, decoder, rank, prior, rows


def plan():
    _, decoder, rank, prior, rows = load_bound()
    return {"schema": "subject_number_embedding_to_l11h3_context_mobius_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "subjects": 32,
            "contexts": [name for name, _ in authority.TEMPLATES],
            "replay_cells": 64, "unopened_crossed_cells": 64,
            "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"],
            "rank": rank["rank"], "prior_terminal": prior["terminal"],
            "authority_sha256": authority.canonical(rows), "bars": BARS,
            "price": PRICE, "binding_sha256": sha(BINDING)}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(600)
    binding, decoder, rank, prior, rows = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    tokens = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long, device=device)
    positions = torch.tensor([row["subject_position"] for row in rows],
                             dtype=torch.long, device=device)
    batch = torch.arange(len(rows), device=device)
    with torch.no_grad():
        base_input = F.rms_norm(model.transformer.wte(tokens),
                                (model.config.n_embd,)).float()
    subject_state = base_input[batch, positions].double()
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"],
                                dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"])
    unit_decoder = decoder_axis / decoder_axis.norm()
    projection = subject_state @ unit_decoder
    target_projection = threshold / float(decoder_axis.norm())
    orthogonal = subject_state - projection[:, None] * unit_decoder
    scale = torch.sqrt((subject_state.square().sum(1) - target_projection ** 2)
                       / orthogonal.square().sum(1))
    removed_subject = target_projection * unit_decoder + scale[:, None] * orthogonal
    removed_input = base_input.clone(); removed_input[batch, positions] = removed_subject.float()
    decoder_error = float((removed_subject @ decoder_axis - threshold).abs().max())
    norm_error = float(((removed_subject.norm(dim=1) - subject_state.norm(dim=1)).abs()
                        / subject_state.norm(dim=1)).max())
    writer_axis = torch.tensor(rank["axis"], dtype=torch.float32, device=device)
    writer_axis /= writer_axis.norm()
    head_slice = slice(HEAD * HEAD_WIDTH, (HEAD + 1) * HEAD_WIDTH)
    counts = {"partial_forwards": 0, "sequences": 0, "permutations": 0,
              "behavior_logits": 0, "backwards": 0, "fits": 0,
              "parameter_updates": 0}

    def capture(initial):
        counts["partial_forwards"] += 1; counts["sequences"] += len(rows)
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
    projection_weight = model.transformer.h[LAYER].attn.c_proj.weight[:, head_slice]
    full_delta = F.linear(base_head - removed_head, projection_weight)
    alpha = (full_delta @ writer_axis).double().cpu().numpy()
    matrix = alpha.reshape(32, 4)

    prior_rows = removal.build_rows()
    prior_map = {(row["pair_index"], row["number"], row["template_id"]): record["alpha"]
                 for row, record in zip(prior_rows, prior["records"])}
    overlap_errors = []
    for index, row in enumerate(rows):
        key = (row["pair_index"], row["number"], row["template_id"])
        if key in prior_map:
            overlap_errors.append(alpha[index] - prior_map[key])
    overlap_errors = np.asarray(overlap_errors)
    overlap_max = float(np.abs(overlap_errors).max())
    overlap_rms = float(np.sqrt(np.mean(overlap_errors ** 2)))
    fixtures = positive_redteam()

    grand, subject, context, interaction = decompose(matrix)
    centered = matrix - grand
    additive = grand + subject + context
    raw_relative = float(np.linalg.norm(interaction) / max(np.linalg.norm(matrix), 1e-30))
    centered_relative = float(np.linalg.norm(interaction) / max(np.linalg.norm(centered), 1e-30))
    total_centered_ss = float(np.sum(centered ** 2))
    singular_values = np.linalg.svd(interaction, compute_uv=False)
    sv_energy = singular_values ** 2
    rank1_energy = float(sv_energy[0] / max(sv_energy.sum(), 1e-30))
    rank2_energy = float(sv_energy[:2].sum() / max(sv_energy.sum(), 1e-30))
    interaction_rms = float(np.sqrt(np.mean(interaction ** 2)))
    replay_ratio = interaction_rms / max(overlap_rms, 1e-30)
    components = {
        "grand": {"value": grand, "rms": abs(grand)},
        "subject_main": {"rms": float(np.sqrt(np.mean(subject ** 2))),
                         "centered_ss_fraction": float(4 * np.sum(subject ** 2)
                                                       / max(total_centered_ss, 1e-30))},
        "context_main": {"rms": float(np.sqrt(np.mean(context ** 2))),
                         "centered_ss_fraction": float(32 * np.sum(context ** 2)
                                                       / max(total_centered_ss, 1e-30))},
        "interaction": {"rms": interaction_rms,
                        "centered_ss_fraction": float(np.sum(interaction ** 2)
                                                      / max(total_centered_ss, 1e-30))},
    }
    rng = np.random.default_rng(SEED); permutation_errors = []
    for _ in range(NULLS):
        permuted = np.stack([row[rng.permutation(4)] for row in matrix])
        permutation_errors.append(float(np.linalg.norm(decompose(permuted)[3])
                                        / max(np.linalg.norm(permuted), 1e-30)))
        counts["permutations"] += 1
    permutation_errors = np.asarray(permutation_errors)
    null_median = float(np.median(permutation_errors))
    observed_percentile = float(np.mean(permutation_errors <= raw_relative))
    finite = bool(np.isfinite(np.asarray([*alpha, *singular_values,
                                          *permutation_errors])).all())
    instrument = bool(finite and decoder_error <= BARS["maximum_geometry_error"]
                      and norm_error <= BARS["maximum_geometry_error"]
                      and overlap_max <= BARS["maximum_overlap_alpha_error"]
                      and max(fixtures.values()) <= BARS["maximum_synthetic_error"]
                      and counts == PRICE
                      and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    additive_ok = bool(instrument and raw_relative <= BARS["maximum_additive_raw_relative_l2"])
    interaction_material = bool(instrument and not additive_ok
                                and interaction_rms >= BARS["minimum_interaction_rms"]
                                and replay_ratio >= BARS["minimum_interaction_to_replay_ratio"])
    rank1_candidate = bool(interaction_material
                           and rank1_energy >= BARS["minimum_rank1_interaction_energy"])
    predictions = dict(zip(PREDICTION_REGISTRY,
                           (instrument, additive_ok, interaction_material, rank1_candidate)))
    terminal = ("invalid" if not instrument else
                "subject_context_additive_candidate" if additive_ok else
                "subject_context_rank1_interaction_candidate" if rank1_candidate else
                "subject_context_explicit_interaction_required" if interaction_material else
                "subject_context_interaction_inconclusive")
    result = {
        "schema": "subject_number_embedding_to_l11h3_context_mobius_v1_result",
        "terminal": terminal, "predictions": predictions,
        "instrument": {"finite": finite, "decoder_error": decoder_error,
                       "relative_norm_error": norm_error,
                       "overlap_cells": len(overlap_errors),
                       "overlap_alpha_max_abs_error": overlap_max,
                       "overlap_alpha_rms_error": overlap_rms,
                       "positive_redteam": fixtures, "counts": counts},
        "decomposition": {"shape": list(matrix.shape), "components": components,
                          "additive_raw_relative_l2": raw_relative,
                          "additive_centered_relative_l2": centered_relative,
                          "interaction_to_replay_rms_ratio": replay_ratio,
                          "interaction_singular_values": singular_values.tolist(),
                          "rank1_interaction_energy": rank1_energy,
                          "rank2_interaction_energy": rank2_energy},
        "permutation": {"count": NULLS, "seed": SEED,
                        "median_additive_raw_relative_l2": null_median,
                        "observed_additive_raw_relative_l2": raw_relative,
                        "median_minus_observed": null_median - raw_relative,
                        "observed_lower_tail_percentile": observed_percentile,
                        "errors": permutation_errors.tolist()},
        "context_means": dict(zip([name for name, _ in authority.TEMPLATES],
                                  (grand + context.ravel()).tolist())),
        "records": [{"row_id": row["row_id"], "subject_index": row["subject_index"],
                     "pair_index": row["pair_index"], "number": row["number"],
                     "subject": row["subject"], "attractor": row["attractor"],
                     "template": row["template_id"], "alpha": float(alpha[i]),
                     "subject_main": float(subject[row["subject_index"], 0]),
                     "context_main": float(context[0, row["context_index"]]),
                     "interaction": float(interaction[row["subject_index"], row["context_index"]])}
                    for i, row in enumerate(rows)],
        "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical(rows),
        "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": time.perf_counter() - started,
        "scope": ("Crossed opened-vocabulary localization of subject and context slots in "
                  "the L11H3 scalar; labels are decomposition indices, not circuit inputs."),
    }
    # Explicit Möbius double differences relative to the first subject/context.
    mobius = matrix - matrix[:, :1] - matrix[:1, :] + matrix[0, 0]
    result["decomposition"]["reference_mobius_rms"] = float(np.sqrt(np.mean(mobius ** 2)))
    result["decomposition"]["reference_mobius_max_abs"] = float(np.abs(mobius).max())
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in
                      ("terminal", "predictions", "instrument", "decomposition",
                       "permutation", "context_means")}, indent=2, sort_keys=True))
    assert instrument


if __name__ == "__main__":
    main()
