"""RUNG 425 (Claude red-team lane) -- FRESH-ROW REPLICATION OF 424's JOINT EDGE BLOCK.

424 (ledger 2544) is now the program's most important positive: a coupled
continuous QK1xQK2xOV block (ranks 6/6/32, 23,310-value screen, native
generators retained, net saving 0) that preserves almost the entire
held-out downstream computation -- SELECT joint summed-edge relMSE
.009718, routed U16 R2 .98525, mean consumer R2 .99285, CE damage
+.000200 nat (CE added above native -- LOWER IS BETTER), against a
marginal PCA arm at .3168/.7079/+.004356 and a head-deranged control at
-.7181/+.550677.  Before the licensed factorization program builds on
it, this rung tests row-stability: refit the identical deterministic
pipeline on FIT (424's exact seeds 422/423) and evaluate all three arms
on 96 rows the arc has NEVER used -- mlp2_rank512_refit_v1
EVALUATION[96:192], sha-pinned, in-run zero row-hash overlap with
FIT+SELECT+FINAL (bar = 0; disjoint too from 421's EVALUATION[0:96]).

Arms (as in 424): marginal (affine PCA), joint (coupled fit), deranged
(joint parameters with score2 head h -> (h+4) mod 9).

Frozen predictions
------------------
pred_a (instrument + bridge): fresh receipt file sha matches; row
    overlap 0; payload fold <= 1e-10; model orthogonality <= 2e-5; and
    the refit reproduces 424's SELECT numbers -- joint summed relMSE
    within rel 5e-3 of .009718029422911471 AND joint routed R2 within
    abs 5e-3 of .9852544170556766 (CUDA-refit wobble observed ~1e-6).
pred_b (fresh replication): FRESH joint summed-edge relMSE <= .05 AND
    fresh joint routed R2 >= .90 AND fresh joint mean consumer R2 >= .90.
pred_c (controls keep ordering off-rows): fresh marginal summed relMSE
    >= 10x fresh joint AND fresh deranged routed R2 <= 0 AND fresh joint
    CE damage <= .002 nat AND <= fresh marginal CE damage (CE added
    above native -- LOWER IS BETTER).

Null: fresh joint summed relMSE > .10, or fresh joint routed R2 < .80,
or the deranged arm matches/beats joint on any registered metric =>
424's near-lossless coupling is row-specific and the factorization
program should pause for diagnosis.

Price: screen only; 424's 23,310-value screen restated; native
generators retained; net model saving 0; no compression or adoption
claim; no 424 bar is altered by any outcome here.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
QK = ROOT / "basis_aligned/qk_mdl"
OUT = BQ / "attention0_edge_block_fresh_rows_results.json"
EDGE = OPS / "attention0_realized_edge_block_term.py"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
FRESH_RECEIPT = BQ / "mlp2_rank512_refit_v1_rows_receipt.json"
FRESH_ROLE = "EVALUATION"
FRESH_SLICE = (96, 192)
D = 1152
DOC_BATCH = 4
POSITIONS = tuple(range(16, 241, 16))
U_RANK = 16

REF_SELECT_JOINT_SUMMED = 0.009718029422911471
REF_SELECT_JOINT_ROUTED = 0.9852544170556766


def _row_hashes(rows: torch.Tensor) -> set[str]:
    return {hashlib.sha256(row.contiguous().numpy().tobytes()).hexdigest()
            for row in rows.cpu()}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert FRESH_SLICE == (96, 192) and U_RANK == 16
        assert EDGE.exists() and ROWS_RECEIPT.exists() and FRESH_RECEIPT.exists()
        entries = json.loads(FRESH_RECEIPT.read_text())["entries"]
        assert FRESH_ROLE in entries and Path(entries[FRESH_ROLE]["path"]).exists()
        print("ATTENTION0 EDGE BLOCK FRESH ROWS | dry run: 424 joint block on never-used rows")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(QK))
    spec = importlib.util.spec_from_file_location("edge_mod", EDGE)
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    spec2 = importlib.util.spec_from_file_location("ov_base", em.BASE)
    base = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(base)
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring
    from tier2_model import rope_tables, apply_rot

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    final_rows = rows_parent.load_role(receipt["entries"]["FINAL"])

    fresh_entry = json.loads(FRESH_RECEIPT.read_text())["entries"][FRESH_ROLE]
    fresh_path = Path(fresh_entry["path"])
    file_sha = hashlib.sha256(fresh_path.read_bytes()).hexdigest()
    fresh_rows = torch.load(fresh_path, weights_only=True)[
        FRESH_SLICE[0]:FRESH_SLICE[1]].contiguous()
    used = (_row_hashes(fit_rows) | _row_hashes(select_rows)
            | _row_hashes(final_rows))
    overlap = sum(1 for h in _row_hashes(fresh_rows) if h in used)

    model, _ = facade.load_bilin18(device=device, dtype=torch.float32)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    block0 = model.transformer.h[0]

    with torch.no_grad():
        captured = base._capture_cproj_input(model, fit_rows, device).to(device)
        weight = block0.attn.c_proj.weight.detach().float()
        a_factor, _b = base._asvd(weight, captured)
        interface = torch.linalg.qr(
            a_factor[:, :U_RANK].float(), mode="reduced").Q.to(device)
        embedding = F.rms_norm(
            model.transformer.wte.weight.detach().float(), (D,))[:base.VOCAB]
        payload_fold_error = base._payload_exactness(model, embedding)
        all_payload = base._payload_codes(model, interface, embedding)
        write_samples = []
        for start in range(0, len(fit_rows), DOC_BATCH):
            tokens = fit_rows[start:start + DOC_BATCH, :-1].to(device)
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
            attention0, _v = block0.attn(F.rms_norm(token_base, (D,)), None)
            write_samples.append(attention0.float()[:, POSITIONS].reshape(-1, D))
        write_samples = torch.cat(write_samples)
        sigma = torch.sqrt(
            (write_samples @ interface).double().square().mean(0)).float()
        fit_gram, _norm, _live = base._response_metric(
            model, fit_rows, interface, sigma, None, device)
        metric_factor, _eig = base._metric_factor(fit_gram)
        fit_edges = em._collect_edges(model, fit_rows, rope_tables, apply_rot)

    initial = em._fit_initial(fit_edges, all_payload)
    marginal = {
        "mean1": initial["mean1"], "basis1": initial["basis1"],
        "mean2": initial["mean2"], "basis2": initial["basis2"],
        "meanv": initial["meanv"], "basisv": initial["basisv"],
    }
    restarts = [
        em._optimize(fit_edges, all_payload, initial, metric_factor, seed)
        for seed in (422, 423)]
    joint = min(restarts, key=lambda value: value["fit_objective"])
    models = {"marginal": marginal, "joint": joint, "deranged": joint}
    model_orth = max(
        em._orth_error(models[arm][key])
        for arm in models for key in ("basis1", "basis2", "basisv"))

    with torch.no_grad():
        select_edges = em._collect_edges(
            model, select_rows, rope_tables, apply_rot)
        bridge_joint = em._edge_metrics(
            select_edges, all_payload, models["joint"], metric_factor, "joint")
        bridge_transport = em._document_transport(
            model, select_rows, interface, all_payload, models,
            rope_tables, apply_rot, base, scoring)
        fresh_edges = em._collect_edges(
            model, fresh_rows, rope_tables, apply_rot)
        fresh_metrics = {
            arm: em._edge_metrics(
                fresh_edges, all_payload, models[arm], metric_factor, arm)
            for arm in ("marginal", "joint", "deranged")}
        fresh_transport = em._document_transport(
            model, fresh_rows, interface, all_payload, models,
            rope_tables, apply_rot, base, scoring)

    bridge_summed = bridge_joint["summed_relative_mse"]
    bridge_routed = bridge_transport["routed_u16_r2"]["joint"]
    fresh_joint_summed = fresh_metrics["joint"]["summed_relative_mse"]
    fresh_marginal_summed = fresh_metrics["marginal"]["summed_relative_mse"]
    fresh_deranged_summed = fresh_metrics["deranged"]["summed_relative_mse"]
    fresh_routed = fresh_transport["routed_u16_r2"]
    fresh_consumer = fresh_transport["mean_consumer_r2"]
    fresh_ce = fresh_transport["ce"]

    pred_a = (
        file_sha == fresh_entry["file_sha256"]
        and overlap == 0
        and payload_fold_error <= 1e-10
        and model_orth <= 2e-5
        and abs(bridge_summed - REF_SELECT_JOINT_SUMMED)
            / REF_SELECT_JOINT_SUMMED <= 5e-3
        and abs(bridge_routed - REF_SELECT_JOINT_ROUTED) <= 5e-3)
    pred_b = (
        fresh_joint_summed <= .05
        and fresh_routed["joint"] >= .90
        and fresh_consumer["joint"] >= .90)
    pred_c = (
        fresh_marginal_summed >= 10 * fresh_joint_summed
        and fresh_routed["deranged"] <= 0
        and fresh_ce["joint"]["damage"] <= .002
        and fresh_ce["joint"]["damage"] <= fresh_ce["marginal"]["damage"])
    null = (
        fresh_joint_summed > .10
        or fresh_routed["joint"] < .80
        or fresh_deranged_summed <= fresh_joint_summed
        or fresh_routed["deranged"] >= fresh_routed["joint"]
        or fresh_ce["deranged"]["damage"] <= fresh_ce["joint"]["damage"])

    result = {
        "status": "attention0_edge_block_fresh_rows_complete",
        "rung": 425,
        "claim_level": "fresh_row_replication_of_424_edge_block_screen_not_compression",
        "fresh_source": {"receipt": FRESH_RECEIPT.name, "role": FRESH_ROLE,
                         "slice": list(FRESH_SLICE), "file_sha256": file_sha,
                         "row_overlap_with_424_roles": overlap},
        "instrument": {"payload_fold_max_abs": payload_fold_error,
                       "model_orthogonality_max_abs": model_orth,
                       "select_bridge_joint_summed": bridge_summed,
                       "select_bridge_joint_routed": bridge_routed},
        "reference_424": {"select_joint_summed": REF_SELECT_JOINT_SUMMED,
                          "select_joint_routed": REF_SELECT_JOINT_ROUTED},
        "fresh_edge_metrics": fresh_metrics,
        "fresh_transport": {
            "routed_u16_r2": fresh_routed,
            "mean_consumer_r2": fresh_consumer,
            "consumer_r2": fresh_transport["consumer_r2"],
            "ce": fresh_ce,
        },
        'pred_a_instrument_and_select_bridge': bool(pred_a),
        'pred_b_fresh_row_block_replicates': bool(pred_b),
        'pred_c_controls_keep_ordering_off_rows': bool(pred_c),
        'null_424_block_is_row_specific': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": ("424_block_row_stable" if pred_a and pred_b and pred_c
                      and not null else "424_factorization_should_pause"),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
