#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_class_mean_dominates_the_unembedding_class pred_c_pronoun_direction_aligns_with_class_contrast pred_d_final_state_contrast_is_shared_not_idiosyncratic pred_e_readers_write_the_shared_direction
"""Subspace fold (v390; Logan, 19 Sep: 'fold back from a subspace, not a direction'). Output target: the unembedding restricted to a token CLASS. Classes:
the 256 vocabulary plural nouns P and their singulars S (v287 spec). W_U rows U_P, U_S; class means m_P, m_S; the shared class contrast c = m_P - m_S; the
idiosyncratic residuals R = U_P - m_P (and U_S - m_S) with top-k right singular vectors (k = 8). Weight-level split: how much of the class's unembedding energy
is the mean vs the top-k residual. Fold back on the v76 rows at the FINAL token: the plural - singular difference of the final normalised state, and each
reader head's / MLP's final write, projected on c, on the pronoun direction u_they - u_he, and on the idiosyncratic top-k subspace. Question: does the number
circuit write the class-SHARED direction (a circuit for 'plural' that moves every plural noun's logit) or something pronoun-specific?
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_closure                              per-writer final-token writes sum to the final residual within relative 1e-4 (instrument)
    pred_b_class_mean_dominates_the_unembedding_class  ||m_P||^2 x n exceeds the top-8 residual singular energy of R_P (the class shares more than it differs in W_U)
    pred_c_pronoun_direction_aligns_with_class_contrast  cos(c, u_they - u_he) >= 0.30
    pred_d_final_state_contrast_is_shared_not_idiosyncratic  the plural - singular final-state difference has >= 3x more energy along c than in the whole top-8 idiosyncratic subspace (per unit direction: energy on c vs mean energy per idiosyncratic direction >= 3x)
    pred_e_readers_write_the_shared_direction  the readers 9.6 / 12.4 / 15.1 together carry >= 0.50 of the writers' summed |contrast| along c (as they do along u_they - u_he, v380)
PRICE (registered maximum): 3 row batches x 1 forward = 3 forwards + one SVD of a 256 x 1152 matrix; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/subspace_fold_number_v390_result.json"
CANDIDATE_ID = "subspace.number_class_v390"
K, N_HEAD, BATCH = 8, 9, 32
CLOSURE_TOL, COS_MIN, RATIO_MIN, READER_MIN = 1e-4, 0.30, 3.0, 0.50
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_class_mean_dominates_the_unembedding_class": "mean energy > top-8 residual energy", "pred_c_pronoun_direction_aligns_with_class_contrast": ">= 0.30", "pred_d_final_state_contrast_is_shared_not_idiosyncratic": ">= 3x per direction", "pred_e_readers_write_the_shared_direction": ">= 0.50"}
READERS = ("9.6", "12.4", "15.1")


def main() -> None:
    rows, he, she, agents, objects = g.build(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "pairs": len(pairs), "k": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "cos_min": COS_MIN, "ratio_min": RATIO_MIN, "reader_min": READER_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float().cpu()
    P = torch.tensor([b for _, b in pairs]); S = torch.tensor([a for a, _ in pairs]); UP, US = WU[P], WU[S]; mP, mS = UP.mean(0), US.mean(0); c = mP - mS
    RP = UP - mP; svP = torch.linalg.svd(RP, full_matrices=False); Vk = svP.Vh[:K]                       # idiosyncratic top-k directions (unit rows)
    mean_energy = float((mP.norm() ** 2) * len(P)); resid_topk_energy = float((svP.S[:K] ** 2).sum()); resid_total_energy = float((svP.S ** 2).sum())
    u = WU[L._single(" they")] - WU[L._single(" he")]; cos_cu = float(torch.nn.functional.cosine_similarity(c, u, dim=0))
    # fold back at the final token: per-writer writes (embedding, attention heads of blocks 9-15 per head, all MLPs, other attention blocks as totals)
    W = {}; XF = []
    def add(k, v): W.setdefault(k, []).append(v)
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pf = torch.tensor([r_.final for r_ in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; parts = {"embedding": x.clone()}
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                for k in parts: parts[k] = block.lambdas[0] * parts[k]
                parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                if 9 <= l <= 15:
                    captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                    try: attention, v1_ = block.attn(xin_a, v1_)
                    finally: hook.remove()
                    Wp = block.attn.c_proj.weight.detach().float(); y = captured["y"].float()
                    for h in range(N_HEAD): parts[f"{l}.{h}"] = y[:, :, h * hd:(h + 1) * hd] @ Wp[:, h * hd:(h + 1) * hd].T
                else:
                    attention, v1_ = block.attn(xin_a, v1_); parts[f"attn{l}"] = attention.clone()
                x = live + attention; m = block.mlp(F.rms_norm(x, (D,))); x = x + m; parts[f"mlp{l}"] = m.clone()
            for k, v in parts.items(): add(k, v[idx, pf].float().cpu())
            XF.append(x[idx, pf].float().cpu()); forwards += 1
    W = {k: torch.cat(v) for k, v in W.items()}; XF = torch.cat(XF); recon = sum(W.values()); closure = float(((recon - XF).norm(dim=1) / XF.norm(dim=1)).max())
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    diff = (XF[plural] - XF[sing]).mean(0)                                        # plural - singular final-state difference (pre-norm residual)
    ch = lambda v: (c / c.norm()); e_c = float((diff @ (c / c.norm())) ** 2); e_idio = [float((diff @ Vk[i]) ** 2) for i in range(K)]; e_u = float((diff @ (u / u.norm())) ** 2)
    ratio = e_c / (sum(e_idio) / K)
    contr = {k: float(((W[k][plural] - W[k][sing]).mean(0)) @ (c / c.norm())) for k in W}; total = sum(abs(v) for v in contr.values()); reader_share = sum(abs(contr[k]) for k in READERS) / total
    contr_u = {k: float(((W[k][plural] - W[k][sing]).mean(0)) @ (u / u.norm())) for k in W}
    top_c = sorted(contr, key=lambda k: -abs(contr[k]))[:8]; top_u = sorted(contr_u, key=lambda k: -abs(contr_u[k]))[:8]
    report = {"closure_max": closure, "class_unembedding": {"mean_energy": mean_energy, "residual_topk_energy": resid_topk_energy, "residual_total_energy": resid_total_energy, "topk_share_of_residual": resid_topk_energy / resid_total_energy, "singular_values_top8": svP.S[:K].tolist()},
              "cos_class_contrast_vs_pronoun_direction": cos_cu, "final_state_difference": {"energy_on_c": e_c, "energy_on_u": e_u, "energy_on_idiosyncratic_dirs": e_idio, "ratio_c_over_mean_idio": ratio, "total_energy": float(diff.norm() ** 2)},
              "writers_on_c_top8": {k: contr[k] for k in top_c}, "writers_on_u_top8": {k: contr_u[k] for k in top_u}, "reader_share_on_c": reader_share}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_class_mean_dominates_the_unembedding_class": mean_energy > resid_topk_energy, "pred_c_pronoun_direction_aligns_with_class_contrast": cos_cu >= COS_MIN, "pred_d_final_state_contrast_is_shared_not_idiosyncratic": ratio >= RATIO_MIN, "pred_e_readers_write_the_shared_direction": reader_share >= READER_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "subspace_fold_number_result_v390", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
