#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_corrected_instrument pred_b_additive_sufficient pred_c_interaction_material pred_d_rank1_candidate pred_e_qualitative_replication
"""Corrected opposite-number-attractor replication of the context decomposition."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_embedding_to_l11h3_context_mobius_v1 as crossed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = POLY / "SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json"
RANK1 = POLY / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
FLAWED = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_V1_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_CORRECTED_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_CORRECTED_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_EMBEDDING_TO_L11H3_CONTEXT_MOBIUS_CORRECTED_V1_RESULT.json"
LAYER, HEAD, HEAD_WIDTH = 11, 3, 128
NULLS, SEED = 256, 20260922
BARS = {"maximum_additive_raw_relative_l2": .15,
        "maximum_duplicate_alpha_error": 1e-4,
        "maximum_flawed_additive_error_drift": .05,
        "maximum_flawed_rank1_energy_drift": .15,
        "maximum_geometry_error": 1e-5,
        "maximum_synthetic_error": 1e-12,
        "minimum_interaction_rms": 5.,
        "minimum_interaction_to_duplicate_ratio": 1e5,
        "minimum_rank1_interaction_energy": .70}
PRICE = {"partial_forwards": 8, "sequences": 512, "permutations": 256,
         "behavior_logits": 0, "backwards": 0, "fits": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_corrected_instrument": None,
                       "pred_b_additive_sufficient": None,
                       "pred_c_interaction_material": None,
                       "pred_d_rank1_candidate": None,
                       "pred_e_qualitative_replication": None}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"authority": Path(authority.__file__), "decoder": DECODER,
             "rank1": RANK1, "flawed_result": FLAWED, "preregistration": PREREG}
    if binding["files"] != {name: sha(path) for name, path in paths.items()} \
            or binding["authority_sha256"] != authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["nulls"] != NULLS \
            or binding["null_seed"] != SEED or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, rank, flawed = (json.loads(path.read_text()) for path in (DECODER, RANK1, FLAWED))
    if decoder["terminal"] != "embedding_number_decoder_frozen" \
            or rank["terminal"] != "rank1_frozen_weights_only" \
            or flawed["terminal"] != "invalid":
        raise ValueError("parent status changed")
    rows = authority.build_rows()
    return binding, decoder, rank, flawed, rows


def plan():
    _, decoder, rank, flawed, rows = load_bound()
    return {"schema": "subject_number_embedding_to_l11h3_context_mobius_corrected_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "opposite_number_attractor_rows": 128,
            "batch_sizes": [64, 64, 64, 64], "rank": rank["rank"],
            "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"],
            "flawed_parent_terminal": flawed["terminal"],
            "authority_sha256": authority.canonical(rows), "bars": BARS,
            "price": PRICE, "binding_sha256": sha(BINDING)}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(600)
    binding, decoder, rank, flawed, rows = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"], dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"])
    unit_decoder = decoder_axis / decoder_axis.norm()
    writer_axis = torch.tensor(rank["axis"], dtype=torch.float32, device=device); writer_axis /= writer_axis.norm()
    head_slice = slice(HEAD * HEAD_WIDTH, (HEAD + 1) * HEAD_WIDTH)
    counts = {"partial_forwards": 0, "sequences": 0, "permutations": 0,
              "behavior_logits": 0, "backwards": 0, "fits": 0, "parameter_updates": 0}
    geometry_errors = []

    def batch_alpha(batch_rows):
        tokens = torch.tensor([row["token_ids"] for row in batch_rows], dtype=torch.long, device=device)
        positions = torch.tensor([row["subject_position"] for row in batch_rows], dtype=torch.long, device=device)
        batch = torch.arange(len(batch_rows), device=device)
        with torch.no_grad():
            base = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)).float()
        subject = base[batch, positions].double(); projection = subject @ unit_decoder
        target = threshold / float(decoder_axis.norm()); orthogonal = subject - projection[:, None] * unit_decoder
        scale = torch.sqrt((subject.square().sum(1) - target ** 2) / orthogonal.square().sum(1))
        removed_subject = target * unit_decoder + scale[:, None] * orthogonal
        removed = base.clone(); removed[batch, positions] = removed_subject.float()
        geometry_errors.extend([float((removed_subject @ decoder_axis - threshold).abs().max()),
            float(((removed_subject.norm(dim=1) - subject.norm(dim=1)).abs() / subject.norm(dim=1)).max())])

        def capture(initial):
            counts["partial_forwards"] += 1; counts["sequences"] += len(batch_rows); found = {}
            with torch.no_grad():
                x = initial; x0 = initial; first_value = None
                for layer, block in enumerate(model.transformer.h):
                    x = block.lambdas[0] * x + block.lambdas[1] * x0; handle = None
                    if layer == LAYER:
                        def hook(_module, arguments):
                            found["head"] = arguments[0][batch, positions, head_slice].detach().clone()
                        handle = block.attn.c_proj.register_forward_pre_hook(hook)
                    try: attention, first_value = block.attn(F.rms_norm(x, (x.shape[-1],)), first_value)
                    finally:
                        if handle is not None: handle.remove()
                    if layer == LAYER: break
                    x = x + attention; x = x + block.mlp(F.rms_norm(x, (x.shape[-1],)))
            return found["head"]
        base_head, removed_head = capture(base), capture(removed)
        weight = model.transformer.h[LAYER].attn.c_proj.weight[:, head_slice]
        return (F.linear(base_head - removed_head, weight) @ writer_axis).double().cpu().numpy()

    group_ids = [[i for i, row in enumerate(rows) if row["template_id"] in names]
                 for names in (("near", "behind"), ("under", "above"))]
    first_parts = [batch_alpha([rows[i] for i in ids]) for ids in group_ids]
    duplicate_parts = [batch_alpha([rows[i] for i in ids]) for ids in group_ids]
    alpha = np.empty(128); duplicate = np.empty(128)
    for ids, part, repeat in zip(group_ids, first_parts, duplicate_parts):
        alpha[ids] = part; duplicate[ids] = repeat
    duplicate_delta = alpha - duplicate
    duplicate_max = float(np.abs(duplicate_delta).max())
    duplicate_rms = float(np.sqrt(np.mean(duplicate_delta ** 2)))
    fixtures = crossed.positive_redteam(); matrix = alpha.reshape(32, 4)
    grand, subject, context, interaction = crossed.decompose(matrix)
    centered = matrix - grand; centered_ss = float(np.sum(centered ** 2))
    raw_relative = float(np.linalg.norm(interaction) / np.linalg.norm(matrix))
    centered_relative = float(np.linalg.norm(interaction) / np.linalg.norm(centered))
    singular_values = np.linalg.svd(interaction, compute_uv=False); energy = singular_values ** 2
    rank1_energy = float(energy[0] / energy.sum()); rank2_energy = float(energy[:2].sum() / energy.sum())
    interaction_rms = float(np.sqrt(np.mean(interaction ** 2)))
    interaction_ratio = interaction_rms / max(duplicate_rms, 1e-30)
    rng = np.random.default_rng(SEED); permutation_errors = []
    for _ in range(NULLS):
        permuted = np.stack([row[rng.permutation(4)] for row in matrix])
        permutation_errors.append(float(np.linalg.norm(crossed.decompose(permuted)[3]) / np.linalg.norm(permuted)))
        counts["permutations"] += 1
    null_median = float(np.median(permutation_errors))
    flawed_raw = flawed["decomposition"]["additive_raw_relative_l2"]
    flawed_rank1 = flawed["decomposition"]["rank1_interaction_energy"]
    additive_drift, rank1_drift = abs(raw_relative - flawed_raw), abs(rank1_energy - flawed_rank1)
    finite = bool(np.isfinite(np.asarray([*alpha, *singular_values, *permutation_errors])).all())
    pred_a = bool(finite and max(geometry_errors) <= BARS["maximum_geometry_error"]
                  and duplicate_max <= BARS["maximum_duplicate_alpha_error"]
                  and max(fixtures.values()) <= BARS["maximum_synthetic_error"]
                  and counts == PRICE and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    pred_b = bool(pred_a and raw_relative <= BARS["maximum_additive_raw_relative_l2"])
    pred_c = bool(pred_a and not pred_b and interaction_rms >= BARS["minimum_interaction_rms"]
                  and interaction_ratio >= BARS["minimum_interaction_to_duplicate_ratio"])
    pred_d = bool(pred_c and rank1_energy >= BARS["minimum_rank1_interaction_energy"])
    pred_e = bool(pred_a and additive_drift <= BARS["maximum_flawed_additive_error_drift"]
                  and rank1_drift <= BARS["maximum_flawed_rank1_energy_drift"])
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c, pred_d, pred_e)))
    terminal = "corrected_context_rank1_interaction_candidate" if pred_d else "corrected_context_additive_candidate" if pred_b else "corrected_context_interaction_null" if pred_a else "invalid"
    components = {"subject_main": {"rms": float(np.sqrt(np.mean(subject ** 2))),
                                    "centered_ss_fraction": float(4*np.sum(subject**2)/centered_ss)},
                  "context_main": {"rms": float(np.sqrt(np.mean(context ** 2))),
                                    "centered_ss_fraction": float(32*np.sum(context**2)/centered_ss)},
                  "interaction": {"rms": interaction_rms,
                                  "centered_ss_fraction": float(np.sum(interaction**2)/centered_ss)}}
    result = {"schema": "subject_number_embedding_to_l11h3_context_mobius_corrected_v1_result",
              "terminal": terminal, "predictions": predictions,
              "instrument": {"finite": finite, "maximum_geometry_error": max(geometry_errors),
                             "duplicate_alpha_max_abs_error": duplicate_max,
                             "duplicate_alpha_rms_error": duplicate_rms,
                             "positive_redteam": fixtures, "counts": counts},
              "decomposition": {"shape": [32,4], "grand": grand, "components": components,
                                "additive_raw_relative_l2": raw_relative,
                                "additive_centered_relative_l2": centered_relative,
                                "interaction_to_duplicate_rms_ratio": interaction_ratio,
                                "interaction_singular_values": singular_values.tolist(),
                                "rank1_interaction_energy": rank1_energy,
                                "rank2_interaction_energy": rank2_energy},
              "flawed_authority_comparison": {"additive_error_absolute_drift": additive_drift,
                                               "rank1_energy_absolute_drift": rank1_drift},
              "permutation": {"count": NULLS, "seed": SEED,
                              "median_additive_raw_relative_l2": null_median,
                              "observed_additive_raw_relative_l2": raw_relative,
                              "median_minus_observed": null_median-raw_relative,
                              "observed_lower_tail_percentile": float(np.mean(np.asarray(permutation_errors)<=raw_relative)),
                              "errors": permutation_errors},
              "context_means": dict(zip([name for name,_ in authority.TEMPLATES],
                                        (grand+context.ravel()).tolist())),
              "records": [{"row_id": row["row_id"], "subject_index": row["subject_index"],
                           "subject": row["subject"], "number": row["number"],
                           "attractor": row["attractor"], "attractor_number": row["attractor_number"],
                           "template": row["template_id"], "alpha": float(alpha[i]),
                           "interaction": float(interaction[row["subject_index"],row["context_index"]])}
                          for i,row in enumerate(rows)],
              "bars": BARS, "price": PRICE, "authority_sha256": authority.canonical(rows),
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
              "wall_seconds": time.perf_counter()-started,
              "scope": "Corrected opposite-number-attractor factorial localization; no OOD or behavioral claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({k:result[k] for k in ("terminal","predictions","instrument","decomposition","flawed_authority_comparison","permutation","context_means")}, indent=2, sort_keys=True))
    assert pred_a


if __name__ == "__main__": main()
