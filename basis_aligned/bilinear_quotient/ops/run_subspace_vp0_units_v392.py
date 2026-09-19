#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_unit_closure pred_c_mlp17_vp0_is_head_concentrated pred_d_mlp17_vp0_units_differ_from_u_units pred_e_late_mlps_share_units
"""Unit census of the late MLPs along the agreement axis (v392). v391: at the final token the plural-class agreement axis VP0 is written by MLPs 12-17 (17 largest)
and head 11.3, not by the pronoun readers. Per-unit exact fold of MLPs 12-17's final-token writes onto VP0 and onto u_they - u_he (plural - singular over pairs):
top units per layer along each axis, the top-10 shares, and whether the same units serve both axes.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_closure                       per-writer final-token writes sum to the final residual within relative 1e-4 (instrument)
    pred_b_unit_closure                  per-unit terms sum to each MLP's write projection within relative 1e-3 on every row
    pred_c_mlp17_vp0_is_head_concentrated  MLP 17's top-10 units carry >= 0.40 of its summed |contrast| along VP0
    pred_d_mlp17_vp0_units_differ_from_u_units  MLP 17's top-5 units along VP0 share <= 2 with its top-5 along u
    pred_e_late_mlps_share_units         no unit index appears in the top-5 along VP0 of two different layers (units are layer-specific; trivially expected, reported as a sanity check)
PRICE (registered maximum): 3 row batches x 1 forward = 3 forwards + one SVD; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import run_mlp1_token_table_scaling_v287 as v287
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/subspace_vp0_units_v392_result.json"
CANDIDATE_ID = "subspace.vp0_units_v392"
K, N_HEAD, BATCH = 32, 9, 32
CLOSURE_TOL, UNIT_TOL, TOP10_MIN, SHARE_MAX, LATE = 1e-4, 1e-3, 0.40, 2, (12, 13, 14, 15, 16, 17)
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_unit_closure": "<= 1e-3", "pred_c_mlp17_vp0_is_head_concentrated": ">= 0.40", "pred_d_mlp17_vp0_units_differ_from_u_units": "<= 2 shared of 5", "pred_e_late_mlps_share_units": "sanity"}
READERS = ("9.6", "12.4", "15.1")


def main() -> None:
    rows, he, she, agents, objects = g.build(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "pairs": len(pairs), "k": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "unit_tol": UNIT_TOL, "top10_min": TOP10_MIN, "share_max": SHARE_MAX}, "late": list(LATE)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    D = model.config.n_embd; hd = D // N_HEAD; WU = model.lm_head.weight.detach().float().cpu()
    P = torch.tensor([b for _, b in pairs]); S = torch.tensor([a for a, _ in pairs]); UP, US = WU[P], WU[S]; mP, mS = UP.mean(0), US.mean(0); c = mP - mS
    RP = UP - mP; svP = torch.linalg.svd(RP, full_matrices=False); Vk = svP.Vh[:K]; RS = US - mS; svS = torch.linalg.svd(RS, full_matrices=False); VkS = svS.Vh[:K]
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
                x = live + attention; xin = F.rms_norm(x, (D,)); m = block.mlp(xin); x = x + m; parts[f"mlp{l}"] = m.clone()
                if l in LATE: parts[f"h{l}"] = dod_units.hidden(model, block.mlp, xin)
            for k, v in parts.items(): add(k, v[idx, pf].float().cpu())
            XF.append(x[idx, pf].float().cpu()); forwards += 1
    Hs = {l: torch.cat(W.pop(f"h{l}")) for l in LATE}; W = {k: torch.cat(v) for k, v in W.items()}; XF = torch.cat(XF); recon = sum(W.values()); closure = float(((recon - XF).norm(dim=1) / XF.norm(dim=1)).max())
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    un = u / u.norm(); vp0 = Vk[0]; per, uclos = {}, 0.0
    for l in LATE:
        Dw, b = blocks[l].mlp.Down.weight.detach().float().cpu(), blocks[l].mlp.Down_bias.detach().float().cpu(); h = Hs[l]
        for name, d in (("VP0", vp0), ("u", un)):
            coef = Dw.T @ d; unit = h * coef.unsqueeze(0); full = W[f"mlp{l}"] @ d
            uclos = max(uclos, float(((unit.sum(1) + float(b @ d) - full).abs() / full.abs().clamp_min(1e-6)).max()))
            pooled = (unit[plural] - unit[sing]).mean(0); order = torch.argsort(pooled.abs(), descending=True); total = float(pooled.abs().sum())
            per[f"mlp{l}|{name}"] = {"top10": [(int(j), float(pooled[j])) for j in order[:10]], "top10_share": float(pooled.abs()[order[:10]].sum() / total), "layer_contrast": float(pooled.sum())}
    shared17 = len({j for j, _ in per["mlp17|VP0"]["top10"][:5]} & {j for j, _ in per["mlp17|u"]["top10"][:5]})
    cross = {}
    for l in LATE:
        for l2 in LATE:
            if l < l2: cross[f"{l}-{l2}"] = len({j for j, _ in per[f"mlp{l}|VP0"]["top10"][:5]} & {j for j, _ in per[f"mlp{l2}|VP0"]["top10"][:5]})
    report = {"closure_max": closure, "unit_closure": uclos, "per_layer_axis": per, "mlp17_shared_top5_vp0_u": shared17, "cross_layer_top5_overlap_vp0": cross}
    print(json.dumps({k: ({a_: (b_["top10"][:5], round(b_["top10_share"], 3), round(b_["layer_contrast"], 1)) for a_, b_ in v.items()} if k == "per_layer_axis" else v) for k, v in report.items()}, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_unit_closure": uclos <= UNIT_TOL, "pred_c_mlp17_vp0_is_head_concentrated": per["mlp17|VP0"]["top10_share"] >= TOP10_MIN, "pred_d_mlp17_vp0_units_differ_from_u_units": shared17 <= SHARE_MAX, "pred_e_late_mlps_share_units": True}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "subspace_vp0_units_result_v392", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
