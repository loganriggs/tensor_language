#!/usr/bin/env python3
"""Regularize a rank-seven H3 student strictly inside the tensor rank-eight teacher."""

# BQGATE: EXPERIMENT pred_a_authority_manual_closure_finite_price pred_b_fit_improves_heldout_kl pred_c_discarded_normal_is_identifiable pred_d_student_is_crossbank_behaviorally_sufficient pred_e_student_complement_is_crossbank_selective pred_f_student_transports_crossbank pred_g_student_improves_v9_a2
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as scoring
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fit_v1
import circuit_candidate_temporal_auxiliary_fresh_cues_v2 as fit_v2
import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as test_v8
import circuit_candidate_temporal_auxiliary_fresh_cues_v9 as test_v9
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1 as instrument
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2.json"
SCREEN = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1_result.json"
RANK7 = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_rank7_v9_confirmation_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CAP8 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
CAP9 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v9_capability_v1_result.json"
V1 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
V2 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v2.py"
V8 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v8.py"
V9 = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v9.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
INSTRUMENT = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank2_downstream_v8_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_tensor_anchored_regularized_rank7_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_tensor_anchored_regularized_rank7_v2"
EXPECTED = {
    "prior": "ce484f2693d006c4c4f9963b9ad574971faf7df7c016be8da9929c9916583c1d",
    "screen": "adc318db1b08fd47c034cf4cd15b7234b16582b7ab134275f8c36265219254fc",
    "rank7": "d080cd9403321de37799c1c5088d3051e543df64aa632bf11bb257f1896014d3",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "cap8": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "cap9": "828d8b15d9bcf048de32d74384e2f4bc62972f20289515f8eb6c576302262392",
    "v1": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "v2": "adbfaf91ed2889cc42da85255edf9f5074f1002e9ad93dc1d4ff706de66d1144",
    "v8": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "v9": "9b771713c5803082c95a3566bc41120587e60f99c4c8bacc291602516bbe01a5",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "instrument": "936a7920b164e57473f7b7204352584b5a438dd0745afc27bfe7f0dd80354a66",
}
STEPS, LR, SIGMA, ENV_WEIGHT, ANCHOR_WEIGHT = 150, 0.02, 0.03, 0.5, 0.1
RESTARTS = ("weight_rank7", "random_101", "random_202")
CHECKPOINTS = tuple(range(0, STEPS + 1, 10))
ARMS = ("base_identity", "writer_live", "h3_full", "h3_regularized_rank7",
        "h3_regularized_rank7_orthogonal")
PRICE = {"model_forwards": 7692, "example_evaluations": 123852,
         "transformer_backward_forwards": 7200, "model_updates": 450,
         "fit_parameters": 8, "evaluation_records": 620}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def projector(q8, normal):
    normal = normal / normal.norm().clamp_min(1e-30)
    discarded = q8 @ normal
    return q8 @ q8.T - discarded @ discarded.T


def basis_from_normal(torch, q8, normal):
    normal = normal.detach() / normal.detach().norm().clamp_min(1e-30)
    _u, _s, vh = torch.linalg.svd(normal.T, full_matrices=True)
    return torch.linalg.qr(q8 @ vh[1:].T, mode="reduced").Q


def manual_logits(backend, batch, delta, projection, *, grad=False):
    """Exact model forward with a differentiable H3 response projector."""
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    n_head, width = int(model.config.n_head), int(model.config.n_embd // model.config.n_head)
    with torch.set_grad_enabled(grad):
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, v1 = x, None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0

            def patch(_module, arguments):
                flattened = arguments[0]
                changed = flattened.clone().view(n_head * 0 + len(batch.row_ids),
                                                   flattened.shape[1], n_head, width)
                component = delta @ projection
                changed[:, :, 3] = changed[:, :, 3] + component.to(changed)
                return (changed.reshape_as(flattened),) + tuple(arguments[1:])

            handle = block.attn.c_proj.register_forward_pre_hook(patch) if layer == 11 else None
            try:
                attention, v1 = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1)
            finally:
                if handle is not None:
                    handle.remove()
            x = live + attention
            x = x + block.mlp(F.rms_norm(x, (model.config.n_embd,)))
        logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0)
        index = torch.arange(len(lengths), device=backend.device)
        position = torch.tensor([length - 1 for length in lengths], device=backend.device)
        last = logits[index, position].float()
        answer = last[index, torch.tensor(batch.answer_ids, device=backend.device)]
        foil = last[index, torch.tensor(batch.foil_ids, device=backend.device)]
        pair = torch.stack((answer, foil), dim=1)
        if not grad:
            pair, last = pair.detach(), last.detach()
        return pair, last


def capture_context(backend, rows):
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base8 = attention_eval.capture_layer_attention(backend, base_batch, 8)
    donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
    base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
    destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
    writer_hook = mediation.fixed_source_delta_hook(
        backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
    handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
    try:
        writer_output, writer11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
    finally:
        handle.remove()
    raw_delta = writer11["head_output"][:, :, 3].float() - base11["head_output"][:, :, 3].float()
    delta = backend.torch.zeros_like(raw_delta)
    for index, query in enumerate(base_batch.semantic_positions):
        delta[index, :int(query)+1] = raw_delta[index, :int(query)+1]
    reconstruction = max(float(x["reconstruction_max_abs"])
        for x in (base8, donor8, base11, writer11))
    return {"rows": rows, "base_batch": base_batch, "donor_batch": donor_batch,
        "base_output": base_output, "donor_output": donor_output,
        "base11_output": base11_output, "writer_output": writer_output,
        "base11": base11, "writer11": writer11, "delta": delta,
        "writer_hook": writer_hook, "reconstruction": reconstruction}


def attach_targets(backend, context):
    torch, F = backend.torch, backend.F
    identity = torch.eye(128, device=backend.device)
    zero = torch.zeros((128, 128), device=backend.device)
    base_pair, base_logits = manual_logits(
        backend, context["base_batch"], context["delta"], zero)
    full_pair, full_logits = manual_logits(
        backend, context["base_batch"], context["delta"], identity)
    log_base = F.log_softmax(base_logits, dim=-1)
    log_full = F.log_softmax(full_logits, dim=-1)
    reference = F.kl_div(log_base, log_full, log_target=True, reduction="batchmean")
    if not math.isfinite(float(reference)) or float(reference) <= 1e-9:
        raise RuntimeError("full-H3 reference KL is ill-conditioned")
    native = torch.tensor(context["base_output"].answer_foil, device=backend.device)
    full_hook, _capture15, algebra = instrument.run_mode(
        backend, context["base_batch"], context["base11"], context["writer11"], identity, "full")
    full_hook_pair = torch.tensor(full_hook.answer_foil, device=backend.device)
    closure = {"manual_base_answer_foil_max_abs": float((base_pair-native).abs().max()),
               "manual_full_answer_foil_max_abs": float((full_pair-full_hook_pair).abs().max()),
               "full_projection_closure_max_abs": float(algebra)}
    context.update({"log_base": log_base, "log_full": log_full,
                    "reference_kl": reference.detach(), "target_closure": closure})
    return context


def environment_loss(backend, context, projection, *, grad):
    F = backend.F
    _pair, sub_logits = manual_logits(
        backend, context["base_batch"], context["delta"], projection, grad=grad)
    identity = backend.torch.eye(128, device=backend.device)
    _pair, comp_logits = manual_logits(
        backend, context["base_batch"], context["delta"], identity-projection, grad=grad)
    match = F.kl_div(F.log_softmax(sub_logits, dim=-1), context["log_full"],
                     log_target=True, reduction="batchmean") / context["reference_kl"]
    inert = F.kl_div(F.log_softmax(comp_logits, dim=-1), context["log_base"],
                     log_target=True, reduction="batchmean") / context["reference_kl"]
    return match + inert, match, inert


def balanced_loss(backend, contexts, q8, normal, a0, *, grad, antithetic=False, generator=None):
    torch = backend.torch
    unit = normal / normal.norm().clamp_min(1e-30)
    normals = (unit,)
    if antithetic:
        noise = torch.randn(unit.shape, generator=generator).to(backend.device)
        tangent = noise - unit * (unit.T @ noise)
        tangent = tangent / tangent.norm().clamp_min(1e-30)
        normals = tuple((unit + sign*SIGMA*tangent) /
                        (unit + sign*SIGMA*tangent).norm().clamp_min(1e-30)
                        for sign in (1.0, -1.0))
    by_environment = []
    details = []
    for context in contexts:
        variants = [environment_loss(backend, context, projector(q8, candidate), grad=grad)
                    for candidate in normals]
        values = [sum(item[index] for item in variants)/len(variants) for index in range(3)]
        by_environment.append(values[0]); details.append(values)
    stacked = torch.stack(by_environment)
    anchor = 2.0 * (1.0 - (unit.T @ a0).square().squeeze())
    joint = stacked.mean() + ENV_WEIGHT*stacked.var(unbiased=False) + ANCHOR_WEIGHT*anchor
    return joint, stacked, anchor, details


def fit_restart(backend, train, heldout, q8, a0, name):
    torch = backend.torch
    seed = 0 if name == "weight_rank7" else int(name.rsplit("_", 1)[1])
    generator = torch.Generator(device="cpu").manual_seed(20261102 + seed)
    initial = a0.clone() if name == "weight_rank7" else torch.randn(
        (8, 1), generator=generator).to(backend.device)
    raw = initial.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=LR)
    trace, best = [], None

    def checkpoint(step):
        nonlocal best
        with torch.no_grad():
            joint, values, anchor, _details = balanced_loss(
                backend, heldout, q8, raw, a0, grad=False)
            selection = float(values.max() + ANCHOR_WEIGHT*anchor)
        report = {"step": step, "joint": float(joint), "selection": selection,
                  "worst_environment": float(values.max()), "mean_environment": float(values.mean()),
                  "anchor_distance_squared": float(anchor)}
        trace.append(report)
        if best is None or selection < best[0]["selection"]:
            best = (report, (raw/raw.norm().clamp_min(1e-30)).detach().clone())

    checkpoint(0)
    for update in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss, _values, _anchor, _details = balanced_loss(
            backend, train, q8, raw, a0, grad=True, antithetic=True, generator=generator)
        loss.backward()
        optimizer.step()
        if update + 1 in CHECKPOINTS:
            checkpoint(update + 1)
    return {"name": name, "normal": best[1], "best": best[0], "trace": trace}


def output_from_pair(pair):
    return producer.BatchOutput(tuple(tuple(float(x) for x in row) for row in pair.detach().cpu()), {})


def evaluate_bank(backend, bank_name, rows, q, projection):
    torch, F = backend.torch, backend.F
    records, downstream, kl_reports = [], {}, {}
    reconstruction = identity_error = algebra = manual_error = 0.0
    forwards = evaluations = 0
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        context = capture_context(backend, panel_rows)
        base_batch = context["base_batch"]
        base15_output, base15 = attention_eval.capture_layer_attention(backend, base_batch, 15)
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(context["writer_hook"])
        try:
            writer15_output, writer15 = attention_eval.capture_layer_attention(backend, base_batch, 15)
        finally:
            handle.remove()
        identity_error = max(identity_error,
            instrument.pair_error(context["base_output"], context["base11_output"]),
            instrument.pair_error(context["base_output"], base15_output),
            instrument.pair_error(context["writer_output"], writer15_output))
        outputs = {"base_identity": context["base_output"], "writer_live": context["writer_output"]}
        captures = {"base_identity": base15, "writer_live": writer15}
        for arm, mode in (("h3_full", "full"), ("h3_regularized_rank7", "rank2"),
                          ("h3_regularized_rank7_orthogonal", "orthogonal")):
            outputs[arm], captures[arm], error = instrument.run_mode(
                backend, base_batch, context["base11"], context["writer11"], q, mode)
            algebra = max(algebra, error)
            reconstruction = max(reconstruction, float(captures[arm]["reconstruction_max_abs"]))
        zero = torch.zeros((128, 128), device=backend.device)
        identity = torch.eye(128, device=backend.device)
        manual = {}
        for arm, p in (("base_identity", zero), ("h3_full", identity),
                       ("h3_regularized_rank7", projection),
                       ("h3_regularized_rank7_orthogonal", identity-projection)):
            pair, logits = manual_logits(backend, base_batch, context["delta"], p)
            manual[arm] = (pair, logits)
            manual_error = max(manual_error, instrument.pair_error(outputs[arm], output_from_pair(pair)))
        log_base = F.log_softmax(manual["base_identity"][1], dim=-1)
        log_full = F.log_softmax(manual["h3_full"][1], dim=-1)
        reference = F.kl_div(log_base, log_full, log_target=True, reduction="batchmean")
        kl_reports[panel] = {
            "reference_full_vs_base": float(reference),
            "student_match_fraction": float(F.kl_div(
                F.log_softmax(manual["h3_regularized_rank7"][1], dim=-1), log_full,
                log_target=True, reduction="batchmean") / reference),
            "complement_inert_fraction": float(F.kl_div(
                F.log_softmax(manual["h3_regularized_rank7_orthogonal"][1], dim=-1), log_base,
                log_target=True, reduction="batchmean") / reference),
        }
        for arm in ARMS:
            records.extend(dict(record, bank=bank_name, panel=panel) for record in scoring.recovery_records(
                panel_rows, context["base_output"], context["donor_output"], outputs[arm], arm=arm))
        downstream[panel] = {arm: instrument.l15_pair_norms(backend, captures[arm], base15, base_batch)
            for arm in ("h3_full", "h3_regularized_rank7", "h3_regularized_rank7_orthogonal")}
        reconstruction = max(reconstruction, context["reconstruction"])
        forwards += 13; evaluations += 13*len(panel_rows)
    summaries = {panel: {arm: scoring.summarize([r for r in records
        if r["panel"] == panel and r["arm"] == arm]) for arm in ARMS} for panel in ("A1", "A2")}
    behavior = {panel: {arm: summaries[panel][arm]["mean_recovery"]
        / summaries[panel]["h3_full"]["mean_recovery"]
        for arm in ("h3_regularized_rank7", "h3_regularized_rank7_orthogonal")}
        for panel in ("A1", "A2")}
    downstream_means = {panel: {arm: sum(values)/len(values) for arm, values in by_arm.items()}
                        for panel, by_arm in downstream.items()}
    transport = {panel: downstream_means[panel]["h3_regularized_rank7"]
        / downstream_means[panel]["h3_full"] for panel in ("A1", "A2")}
    return {"summaries": summaries, "behavior_fraction_of_full_h3": behavior,
        "downstream_fraction_of_full_h3": transport, "full_vocabulary_kl": kl_reports,
        "records": records, "forwards": forwards, "evaluations": evaluations,
        "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "identity_max_abs": identity_error, "projection_closure_max_abs": algebra,
            "manual_vs_hook_answer_foil_max_abs": manual_error}}


def main():
    paths = {"prior": PRIOR, "screen": SCREEN, "rank7": RANK7, "subspace": SUBSPACE,
             "cap8": CAP8, "cap9": CAP9, "v1": V1, "v2": V2, "v8": V8, "v9": V9,
             "family_runner": FAMILY_RUNNER, "instrument": INSTRUMENT}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("tensor-anchored regularized DAS authority changed")
    prior, screen, rank7_result, subspace, cap8, cap9 = [json.loads(path.read_text())
        for path in (PRIOR, SCREEN, RANK7, SUBSPACE, CAP8, CAP9)]
    if (prior.get("candidate_id") != CANDIDATE_ID or screen.get("selected_rank") != 7
            or rank7_result.get("terminal") != "transfer_failure"
            or cap8.get("terminal") != "manifest" or cap9.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    fit_banks = {"v1": fit_v1.build_rows(), "v2": fit_v2.build_rows()}
    fit_groups = {(bank, panel): [row for row in rows if row["transform_id"] == panel]
                  for bank, rows in fit_banks.items() for panel in ("A1", "A2")}
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "teacher_rank": 8, "student_rank": 7,
        "fit_parameters": 8, "fit_environments": list(f"{b}_{p}" for b,p in fit_groups),
        "train_rows_per_environment": 16, "selection_rows_per_environment": 16,
        "restarts": list(RESTARTS), "steps_per_restart": STEPS, "checkpoints": list(CHECKPOINTS),
        "loss": {"full_vocabulary_sufficiency_kl": 1.0, "complement_inertness_kl": 1.0,
                 "environment_variance": ENV_WEIGHT, "tangent_noise_sigma": SIGMA,
                 "tensor_anchor": ANCHOR_WEIGHT}, "evaluation_banks": ["v8", "v9"],
        "arms": list(ARMS), **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    family, singular, energy = family_builder.build_family(backend, subspace)
    q7_weight, q8 = family[7], family[8]
    _u, _s, vh = backend.torch.linalg.svd((q8.T @ q7_weight).T, full_matrices=True)
    a0 = vh[-1:].T
    train, heldout = [], []
    for rows in fit_groups.values():
        train.append(attach_targets(backend, capture_context(backend, rows[0::2])))
        heldout.append(attach_targets(backend, capture_context(backend, rows[1::2])))
    target_closure = max(value for context in train+heldout
                         for value in context["target_closure"].values())
    with backend.torch.no_grad():
        baseline_train = float(balanced_loss(backend, train, q8, a0, a0, grad=False)[0])
        baseline_heldout = float(balanced_loss(backend, heldout, q8, a0, a0, grad=False)[0])
    fits = [fit_restart(backend, train, heldout, q8, a0, name) for name in RESTARTS]
    selected = min(fits, key=lambda item: item["best"]["selection"])
    normal = selected["normal"]
    q = basis_from_normal(backend.torch, q8, normal)
    projection = q @ q.T
    normal_cosines = [float((fits[i]["normal"].T @ fits[j]["normal"]).abs())
        for i in range(len(fits)) for j in range(i+1, len(fits))]
    normal_min_cosine = min(normal_cosines)
    projector_distance = float(2.0 * (1.0-(normal.T @ a0).square().squeeze()))
    allowed8 = {row_id for ids in cap8["jointly_capable_row_ids"].values() for row_id in ids}
    allowed9 = {row_id for ids in cap9["jointly_capable_row_ids"].values() for row_id in ids}
    test_rows = {
        "v8": [row for row in test_v8.build_rows() if row["transform_id"] in {"A1","A2"}
               and row["row_id"] in allowed8],
        "v9": [row for row in test_v9.build_rows() if row["transform_id"] in {"A1","A2"}
               and row["row_id"] in allowed9],
    }
    evaluations = {name: evaluate_bank(backend, name, rows, q, projection)
                   for name, rows in test_rows.items()}
    max_instrument = max(value for report in evaluations.values()
                         for value in report["instrument"].values())
    fit_values = [number for fit in fits for point in fit["trace"] for number in point.values()
                  if isinstance(number, (int,float))]
    pred_a = bool(target_closure <= 1e-4 and max_instrument <= 1e-4
        and all(math.isfinite(x) for x in fit_values) and PRICE == {
            "model_forwards": 7692, "example_evaluations": 123852,
            "transformer_backward_forwards": 7200, "model_updates": 450,
            "fit_parameters": 8, "evaluation_records": 620})
    pred_b = selected["best"]["joint"] < baseline_heldout
    pred_c = normal_min_cosine >= .80
    pred_d = all(evaluations[b]["behavior_fraction_of_full_h3"][p]["h3_regularized_rank7"] >= .90
                 for b in ("v8","v9") for p in ("A1","A2"))
    pred_e = all(abs(evaluations[b]["behavior_fraction_of_full_h3"][p][
        "h3_regularized_rank7_orthogonal"]) <= .12 for b in ("v8","v9") for p in ("A1","A2"))
    pred_f = all(evaluations[b]["downstream_fraction_of_full_h3"][p] >= .90
                 for b in ("v8","v9") for p in ("A1","A2"))
    old_v9_a2 = rank7_result["behavior_fraction_of_full_h3"]["A2"]["h3_weight_rank7"]
    new_v9_a2 = evaluations["v9"]["behavior_fraction_of_full_h3"]["A2"]["h3_regularized_rank7"]
    pred_g = new_v9_a2 > old_v9_a2
    predictions = {"pred_a_authority_manual_closure_finite_price": pred_a,
        "pred_b_fit_improves_heldout_kl": pred_b,
        "pred_c_discarded_normal_is_identifiable": pred_c,
        "pred_d_student_is_crossbank_behaviorally_sufficient": pred_d,
        "pred_e_student_complement_is_crossbank_selective": pred_e,
        "pred_f_student_transports_crossbank": pred_f,
        "pred_g_student_improves_v9_a2": pred_g}
    terminal = ("invalid" if not pred_a else "screen" if all(predictions.values())
        else "underidentified" if not pred_c else "regularized_but_insufficient" if pred_b
        else "regularization_null")
    result = {"schema": "temporal_auxiliary_h3_tensor_anchored_regularized_rank7_result_v2",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "teacher": {"rank": 8, "basis_sha256": tensor_sha(q8),
                    "static_reader_energy_fraction": energy[8],
                    "leading_orthogonal_reader_singular_values": [float(x) for x in singular[:8]]},
        "student": {"rank": 7, "basis_sha256": tensor_sha(q),
            "basis": q.detach().cpu().tolist(), "discarded_normal_teacher_coordinates": normal.cpu().flatten().tolist(),
            "projector_distance_squared_from_weight_rank7": projector_distance,
            "selected_restart": selected["name"], "restart_min_discarded_normal_cosine": normal_min_cosine},
        "fit": {"baseline_train_joint": baseline_train, "baseline_heldout_joint": baseline_heldout,
                "target_manual_closure_max_abs": target_closure,
                "restarts": [{key: value for key,value in fit.items() if key != "normal"} for fit in fits]},
        "evaluations": {name: {key:value for key,value in report.items()
            if key not in {"records","forwards","evaluations"}} for name, report in evaluations.items()},
        "v9_a2_behavior_comparison": {"fixed_weight_rank7": old_v9_a2,
            "regularized_rank7": new_v9_a2, "gain": new_v9_a2-old_v9_a2},
        "predictions": predictions, "terminal": terminal, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "student": result["student"],
        "fit": {k:v for k,v in result["fit"].items() if k != "restarts"},
        "evaluations": {b: {"behavior": r["behavior_fraction_of_full_h3"],
            "transport": r["downstream_fraction_of_full_h3"], "kl": r["full_vocabulary_kl"]}
            for b,r in result["evaluations"].items()},
        "v9_a2_behavior_comparison": result["v9_a2_behavior_comparison"],
        "predictions": predictions, "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
