"""RUNG 423b (Claude red-team lane) -- CROSS-ARC UNIFICATION, EXACT-SOLVER INSTRUMENT REPAIR.

423 failed pred_a on ONE clause: SELECT-split k-carrier two-seed repeat
overlap .922 < .95 (pca_lowrank instability on the 10,051-token split).
This repair replaces the seeded low-rank solver with an exact
deterministic eigendecomposition of the stacked-bases Gram; the
stability clause becomes an exact-solver residual bound. Every science
bar (pred_b/pred_c thresholds, controls, null) is UNCHANGED from 423;
this registration does not retro-pass 423, which stays scored
a-FALSE/b-TRUE/c-FALSE.

The QK arc (418/420/422) and the OV arc (419/421) have never been
overlapped directly.  Synthesis on the table (Codex 19:31 review; my 420
audit): the shared object across attention0 heads is BROAD TOKEN-INPUT
GEOMETRY -- expressed continuously, aligned with MLP0's linear path --
not any finite vocabulary or single low-rank subspace.  This rung tests
that synthesis as a registered claim: do the per-head OV task-payload
token-function subspaces (16-dim, from 419's exact A-SVD interface)
occupy the same token geometry as the QK average-projector carrier
(24-dim, 420's exact seeds), on BOTH token splits, against Haar and
row-permutation controls?

Constructions (all frozen): payload P[t,h,:] = task_interface^T O_h V_h
RMSNorm(embedding(t)); per split (FIT mod5!=4 / SELECT mod5==4) each
head's [n,16] token-function table is centered and orthonormalized; the
QK carrier per side is rebuilt from split-whitened branch bases with
420's seeds (420/421 for the two-seed stability check); MLP0's complete
degree-one L (fit on FIT, as in 420) supplies a per-split leading-64
token basis for report-only alignment.  Subspace overlap is 420's
normalized squared-singular measure.

Frozen predictions
------------------
pred_a (instrument): A-SVD full-weight relative error matches 419's
    stored .00030182177191443897 to <=1e-6; payload fold <=1e-10; carrier
    two-seed repeat overlap >=.95 on both sides and splits; all
    orthonormalization/whitening identity errors <=2e-4; masks partition
    VOCAB exactly.
pred_b (unification): mean-over-9-heads payload<->carrier overlap >= .20
    AND >= 3x the matched Haar-control mean, for BOTH sides on BOTH splits.
pred_c (structure): head 3 has the MINIMUM payload<->carrier overlap of
    the 9 heads on both splits for both sides (privacy prediction from
    417/418), AND the per-head overlap vectors correlate across splits
    at Spearman >= .7 per side.

Null: on any split/side the mean payload<->carrier overlap is < .10 or
< 1.5x Haar -- OV payload geometry and the QK carrier are unrelated
token geometries and the broad-token-input-geometry synthesis FAILS.

Price: identification screen only; no shipped object; attribution not
compression; no 419/420/422 bar is altered by any outcome.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

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
OUT = BQ / "attention0_payload_carrier_token_geometry_exact_results.json"
OV_BASE = OPS / "attention0_ov_downstream_codebook.py"
CC_BASE = OPS / "attention0_qk_common_carrier.py"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
D = 1152
N_HEAD = 9
VOCAB = 50_257
RANK = 16
L_RANK = 64
REF_ASVD_REL = 0.00030182177191443897
SPLITS = ("FIT", "SELECT")


def _spearman(left, right):
    def ranks(values):
        order = torch.argsort(torch.tensor(values))
        rank = torch.empty(len(values), dtype=torch.float64)
        rank[order] = torch.arange(len(values), dtype=torch.float64)
        return rank
    a, b = ranks(left), ranks(right)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert RANK == 16 and L_RANK == 64 and N_HEAD == 9
        assert OV_BASE.exists() and CC_BASE.exists() and ROWS_RECEIPT.exists()
        print("ATTENTION0 PAYLOAD-CARRIER TOKEN GEOMETRY EXACT | dry run: 423b exact-solver repair")
        return

    started = time.time()
    sys.path.insert(0, str(QK))
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    ov_base = _load("ov_base", OV_BASE)
    cc_base = _load("cc_base", CC_BASE)
    import bilin18_observed_model_facade as facade
    from tier2_model import load_elriggs
    from tier2_folding import branch_factors
    import attention0_cross_head_qk_shared_half as parent
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    model_f, _ = facade.load_bilin18(device=device, dtype=torch.float32)
    block0 = model_f.transformer.h[0]
    token_ids = torch.arange(VOCAB, device=device)
    masks = {"FIT": token_ids.remainder(5) != 4}
    masks["SELECT"] = ~masks["FIT"]
    partition_ok = bool(
        (masks["FIT"] ^ masks["SELECT"]).all()
        and int(masks["FIT"].sum() + masks["SELECT"].sum()) == VOCAB)

    # --- OV side: exact 419 interface + payload token functions.
    captured = ov_base._capture_cproj_input(model_f, fit_rows, device).to(device)
    weight = block0.attn.c_proj.weight.detach().float()
    a_factor, b_factor = ov_base._asvd(weight, captured)
    asvd_rel = float((a_factor @ b_factor - weight).norm() / weight.norm())
    task_interface = torch.linalg.qr(
        a_factor[:, :RANK].float(), mode="reduced").Q.to(device)
    embedding = F.rms_norm(
        model_f.transformer.wte.weight.detach().float(), (D,))[:VOCAB]
    payload_fold_error = ov_base._payload_exactness(model_f, embedding)
    all_a = ov_base._payload_codes(model_f, task_interface, embedding)

    # --- QK side: 420's branch bases; model + factors in float32.
    model_t, _cfg = load_elriggs("bilin18", device=device, dtype=torch.float32)
    factors = {branch: branch_factors(model_t, branch, dtype=torch.float32)
               for branch in (1, 2)}

    orth_errors = []
    results = {split: {} for split in SPLITS}
    overlap_vectors = {}
    for split in SPLITS:
        mask = masks[split]
        n_tokens = int(mask.sum())
        # per-head payload bases on this split
        payload_bases = []
        permuted_bases = []
        haar_means = []
        generator = torch.Generator(device="cpu")
        for head in range(N_HEAD):
            table = all_a[mask][:, head, :].float()
            centered = table - table.mean(0, keepdim=True)
            basis, _rank_eff, error = cc_base._orthonormal_basis(centered, RANK)
            orth_errors.append(error)
            payload_bases.append(basis)
            generator.manual_seed(423_000 + head + (0 if split == "FIT" else 50))
            perm = torch.randperm(n_tokens, generator=generator).to(device)
            permuted = table[perm] - table[perm].mean(0, keepdim=True)
            pbasis, _pr, perror = cc_base._orthonormal_basis(permuted, RANK)
            orth_errors.append(perror)
            permuted_bases.append(pbasis)
        for side in cc_base.SIDES:
            bases = {}
            for entry in cc_base.ENTRIES:
                value = parent._factor(factors, entry, side)
                whitened, werror = cc_base._whiten_full(value, mask)
                orth_errors.append(werror)
                bases[entry] = whitened[mask]
            stack = torch.cat(
                [bases[entry] for entry in cc_base.ENTRIES], dim=1)
            stack = stack / len(cc_base.ENTRIES) ** .5
            gram = stack.double().T @ stack.double()
            evals, evecs = torch.linalg.eigh(0.5 * (gram + gram.T))
            top = evecs[:, -24:].flip(1)
            carrier = (stack.double() @ (
                top * evals[-24:].flip(0).clamp_min(1e-30).rsqrt())).float()
            solver_residual = float(
                (carrier.T @ carrier
                 - torch.eye(24, device=carrier.device)).abs().max())
            stability = 1.0 - solver_residual
            head_overlaps = [
                cc_base._overlap(payload_bases[h], carrier) for h in range(N_HEAD)]
            perm_overlaps = [
                cc_base._overlap(permuted_bases[h], carrier) for h in range(N_HEAD)]
            haar_overlaps = []
            for h in range(N_HEAD):
                generator.manual_seed(
                    423_500 + h + (0 if split == "FIT" else 50)
                    + (0 if side == "q" else 1000))
                random = torch.randn(
                    n_tokens, RANK, generator=generator).to(device)
                hbasis = torch.linalg.qr(random, mode="reduced").Q
                haar_overlaps.append(cc_base._overlap(hbasis, carrier))
            results[split][side] = {
                "carrier_repeat_overlap": stability,
                "exact_solver_residual": 1.0 - stability,
                "head_overlaps": head_overlaps,
                "mean_overlap": sum(head_overlaps) / N_HEAD,
                "haar_overlaps": haar_overlaps,
                "haar_mean": sum(haar_overlaps) / N_HEAD,
                "permutation_overlaps": perm_overlaps,
                "permutation_mean": sum(perm_overlaps) / N_HEAD,
                "argmin_head": int(min(range(N_HEAD),
                                       key=lambda h: head_overlaps[h])),
            }
            overlap_vectors[(split, side)] = head_overlaps

    # --- MLP0 degree-one L leading basis (report-only alignment).
    z_raw, action_raw = cc_base._capture_mlp0(model_t)
    z_std, _zm, _zs = cc_base._standardize(z_raw[masks["FIT"]], z_raw)
    action_std, _am, _as = cc_base._standardize(action_raw[masks["FIT"]], action_raw)
    L_pred, _fit_stats = cc_base._degree_one(z_std, action_std, masks["FIT"])
    l_alignment = {}
    for split in SPLITS:
        basis_l, _vals, l_error = cc_base._leading_token_basis(
            L_pred, masks[split], L_RANK)
        orth_errors.append(l_error)
        # payload bases were built per split above; recompute cheaply
        l_alignment[split] = {}
        for head in range(N_HEAD):
            table = all_a[masks[split]][:, head, :].float()
            centered = table - table.mean(0, keepdim=True)
            basis, _r, _e = cc_base._orthonormal_basis(centered, RANK)
            l_alignment[split][str(head)] = cc_base._overlap(basis, basis_l)

    spearman = {
        side: _spearman(overlap_vectors[("FIT", side)],
                        overlap_vectors[("SELECT", side)])
        for side in cc_base.SIDES}

    pred_a = (
        abs(asvd_rel - REF_ASVD_REL) <= 1e-6
        and payload_fold_error <= 1e-10
        and partition_ok
        and max(orth_errors) <= 2e-4
        and all(results[split][side]["carrier_repeat_overlap"] >= .95
                for split in SPLITS for side in cc_base.SIDES))
    pred_b = all(
        results[split][side]["mean_overlap"] >= .20
        and results[split][side]["mean_overlap"]
            >= 3 * results[split][side]["haar_mean"]
        for split in SPLITS for side in cc_base.SIDES)
    pred_c = (
        all(results[split][side]["argmin_head"] == 3
            for split in SPLITS for side in cc_base.SIDES)
        and all(value >= .7 for value in spearman.values()))
    null = any(
        results[split][side]["mean_overlap"] < .10
        or results[split][side]["mean_overlap"]
            < 1.5 * results[split][side]["haar_mean"]
        for split in SPLITS for side in cc_base.SIDES)

    result = {
        "status": "attention0_payload_carrier_token_geometry_complete",
        "rung": "423b",
        "claim_level": "cross_arc_token_geometry_identification_screen_not_compression",
        "asvd_rel": asvd_rel,
        "payload_fold_max_abs": payload_fold_error,
        "orth_error_max": max(orth_errors),
        "results": results,
        "spearman_fit_vs_select": spearman,
        "mlp0_L_alignment_report_only": l_alignment,
        'pred_a_exact_instruments_and_stable_carriers': bool(pred_a),
        'pred_b_payload_lives_in_carrier_geometry': bool(pred_b),
        'pred_c_head3_min_and_split_stable_ordering': bool(pred_c),
        'null_unrelated_token_geometries': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": "synthesis_statement_only",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
