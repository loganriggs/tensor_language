#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_number_units_lead_on_vc1 pred_c_gender_units_appear_on_vc1 pred_d_top50_share_vc1 pred_e_vc1_and_u_rankings_overlap
"""MLP 8 at unit grain along the pronoun class's VC1 axis (v399). v398: the readers' largest class-internal output axis VC1 separates she / her / he / him from
every other pronoun (cos 0.45 with they - he). v168's instrument -- 9.6's reader direction applied to MLP 8's write at the noun, per unit -- is rerun with the
reader direction built from VC1 (r_VC1 = V_9.6^T O_9.6^T VC1) alongside the they - he one: which MLP-8 units carry 'not he / she' vs 'they' at the noun; where
the number units 829 / 953 / 1030 and the gender units 3152 / 3943 (v168) rank on each.
PREDICTIONS (scored as written; failures preserved; priors from v168 / v251 / v398)
    pred_a_unit_closure               sum_j T_j + r . b = r . mlp8(noun) within relative 1e-3 on every row, both directions
    pred_b_number_units_lead_on_vc1   829, 953, 1030 are all in the top 10 along VC1 (the 'not he/she' axis is carried by the same units as number)
    pred_c_gender_units_appear_on_vc1 3152 or 3943 ranks in the top 30 along VC1 (gender leaks into the axis). Prior: unsure.
    pred_d_top50_share_vc1            the top 50 units carry >= 0.50 of the pooled VC1 contrast
    pred_e_vc1_and_u_rankings_overlap the top-20 sets along VC1 and along they - he share >= 12 units
PRICE (registered maximum): 2 batches x 1 forward (blocks 0-8) = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp8_units_vc1_v399_result.json"
CANDIDATE_ID = "subspace.mlp8_units_vc1_v399"
LAYER = 8
CLOSURE_TOL, TOP50_MIN, OVERLAP_MIN = 1e-3, 0.50, 12
PRONOUNS = (" they", " we", " them", " us", " he", " she", " it", " him", " her", " I", " you")
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_number_units_lead_on_vc1": "top 10 x 3", "pred_c_gender_units_appear_on_vc1": "top 30", "pred_d_top50_share_vc1": ">= 0.50", "pred_e_vc1_and_u_rankings_overlap": ">= 12 of 20"}


def main() -> None:
    rows, he, she, agents, objects = g.build()   # he = they (positive), she = he (negative): names kept from v164
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "layer": LAYER, "reader_head": "9.6", "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top50_min": TOP50_MIN, "overlap_min": OVERLAP_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r_u = L.reader_directions(model, comp96, fw.directions)[6].float()
    WU = model.lm_head.weight.detach().float(); C = torch.tensor([L._single(t) for t in PRONOUNS], device=WU.device); UC = WU[C]; VC1 = torch.linalg.svd(UC - UC.mean(0), full_matrices=False).Vh[1]
    a96 = model.transformer.h[9].attn; hd = model.config.n_embd // 9; sl = slice(6 * hd, 7 * hd)
    r_vc1 = (a96.c_v.weight.detach().float()[sl].T @ (a96.c_proj.weight.detach().float()[:, sl].T @ VC1))          # V^T O^T VC1 : the state direction 9.6 reads to write VC1
    u_dir = WU[L._single(" they")] - WU[L._single(" he")]; r_chk = (a96.c_v.weight.detach().float()[sl].T @ (a96.c_proj.weight.detach().float()[:, sl].T @ u_dir))
    mlp = model.transformer.h[LAYER].mlp
    Lw, Rw, Dw, bias = mlp.Left.weight.detach().float(), mlp.Right.weight.detach().float(), mlp.Down.weight.detach().float(), mlp.Down_bias.detach().float()
    dirs = {"vc1": r_vc1, "u_via_96": r_chk}; per_dir = {}; closure = 0.0; forwards = 0
    rD = {k: (d.to(Dw.device) @ Dw) for k, d in dirs.items()}; rb = {k: float(d.to(bias.device) @ bias) for k, d in dirs.items()}
    per_row = {k: [] for k in dirs}
    with torch.no_grad():
        for start in range(0, len(rows), v1.BATCH):
            chunk = rows[start:start + v1.BATCH]; tokens = fw._tokens(chunk)
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for layer, block in enumerate(model.transformer.h):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if layer == LAYER:
                    Lx, Rx = mlp.Left(xin), mlp.Right(xin); h = (Lx * Rx); out = mlp(xin)
                    for i, row in enumerate(chunk):
                        s_ = noun_of(row); hj = h[i, s_].float()
                        for k in dirs:
                            T = rD[k] * hj; true = float(dirs[k].to(out.device) @ out[i, s_].float()); recon = float(T.sum()) + rb[k]
                            closure = max(closure, abs(recon - true) / max(abs(true), 1e-6)); per_row[k].append(T.cpu())
                    break
                x = x + block.mlp(xin)
            forwards += 1
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    def pooled(k):
        acc = torch.zeros(Dw.shape[1])
        for i, row in enumerate(rows):
            if not row.present: continue
            j = partner[(row.construction, row.group, False)]; acc += per_row[k][i] - per_row[k][j]
        return acc
    tot = {k: pooled(k) for k in dirs}; order = {k: torch.argsort(tot[k].abs(), descending=True) for k in dirs}
    rank = {k: {str(u): int((tot[k].abs() > tot[k].abs()[u]).sum()) + 1 for u in (829, 953, 1030, 3152, 3943)} for k in dirs}
    share50 = {k: float(tot[k][order[k][:50]].sum() / tot[k].sum()) for k in dirs}; overlap = len(set(order["vc1"][:20].tolist()) & set(order["u_via_96"][:20].tolist()))
    report = {"closure_max": closure, "top12": {k: [(int(j), round(float(tot[k][j]), 2)) for j in order[k][:12]] for k in dirs}, "ranks": rank, "top50_share": share50, "top20_overlap": overlap, "cos_r_vc1_r_u": float(torch.nn.functional.cosine_similarity(r_vc1, r_chk, dim=0))}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_number_units_lead_on_vc1": all(rank["vc1"][str(u)] <= 10 for u in (829, 953, 1030)), "pred_c_gender_units_appear_on_vc1": min(rank["vc1"]["3152"], rank["vc1"]["3943"]) <= 30, "pred_d_top50_share_vc1": share50["vc1"] >= TOP50_MIN, "pred_e_vc1_and_u_rankings_overlap": overlap >= OVERLAP_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp8_units_vc1_result_v399", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
