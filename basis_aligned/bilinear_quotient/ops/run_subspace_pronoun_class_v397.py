#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_pronoun_class_mean_carries_little pred_c_pronoun_residual_top_direction_is_number pred_d_number_direction_carries_most pred_e_same_on_text
"""Subspace fold on the answer class itself: the pronouns (v397). v390-v396: the plural-NOUN class's mean and residual directions in W_U carry little of the number
circuit's output. The circuit's answers are pronouns, so fold from the pronoun class instead: C = {they, we, them, us, he, she, it, him, her, I, you} (11
rows of W_U); class mean m_C (pronoun-ness) and the residual R_C = U_C - m_C with its right singular vectors VC0..VC7 (the axes along which pronouns differ:
number, person, gender, case). Energy of the plural - singular final-state difference along m_C and each VCi, and their cosines with u_they - u_he, on the v76
rows and on the 128 natural rows.
PREDICTIONS (scored as written; failures preserved; priors from v390-v396)
    pred_a_closure                            per-writer final-token writes sum to the final residual within relative 1e-4 (instrument), panel and text
    pred_b_pronoun_class_mean_carries_little  energy along m_C is <= 0.10 of the energy along u on the panel (the circuit does not raise pronoun-ness)
    pred_c_pronoun_residual_top_direction_is_number  the VCi with the largest panel energy has |cos| >= 0.50 with u_they - u_he
    pred_d_number_direction_carries_most      that VCi carries >= 0.50 of the panel energy within VC0..VC7
    pred_e_same_on_text                       the same VCi carries the most energy within VC0..VC7 on the natural rows
PRICE (registered maximum): 2 natural batches + 3 panel batches (blocks 0-17) = 5 forwards + one small SVD; 0 backwards; 0 fits. Bar <= 7.
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
OUT = ROOT / "circuits/followups/subspace_pronoun_class_v397_result.json"
CANDIDATE_ID = "subspace.pronoun_class_v397"
K, N_HEAD, BATCH = 32, 9, 32
CLOSURE_TOL, MEAN_MAX, COS_MIN, CONC_MIN = 1e-4, 0.10, 0.50, 0.50
PRONOUNS = (" they", " we", " them", " us", " he", " she", " it", " him", " her", " I", " you")
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 7
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_pronoun_class_mean_carries_little": "<= 0.10 of u", "pred_c_pronoun_residual_top_direction_is_number": "|cos| >= 0.50", "pred_d_number_direction_carries_most": ">= 0.50 of VC0..7", "pred_e_same_on_text": "same index"}
READERS = ("9.6", "12.4", "15.1")


def main() -> None:
    rows, he, she, agents, objects = g.build(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "k": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "mean_max": MEAN_MAX, "cos_min": COS_MIN, "conc_min": CONC_MIN}, "pronouns": list(PRONOUNS), "natural_rows": len(recs)}
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
    def capture(batches):
        W = {}; XF = []
        def add(k, v): W.setdefault(k, []).append(v)
        with torch.no_grad():
            for tokens, pf in batches:
                idx = torch.arange(tokens.shape[0]); x = F.rms_norm(model.transformer.wte(tokens), (D,)); x0, v1_ = x, None; parts = {"embedding": x.clone()}
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
                XF.append(x[idx, pf].float().cpu())
        W = {k: torch.cat(v) for k, v in W.items()}; XF = torch.cat(XF); return W, XF, float(((sum(W.values()) - XF).norm(dim=1) / XF.norm(dim=1)).max())
    nat = torch.tensor([r_["ids"] for r_ in recs], device="cuda"); fin = [len(r_["ids"]) - 1 for r_ in recs]
    Wt, XFt, c1 = capture([(nat[s0:s0 + 64], torch.tensor(fin[s0:s0 + 64])) for s0 in range(0, len(recs), 64)]); forwards += 2
    Wp_, XFp, c2 = capture([(fw._tokens(rows[s0:s0 + BATCH]), torch.tensor([r_.final for r_ in rows[s0:s0 + BATCH]])) for s0 in range(0, len(rows), BATCH)]); forwards += 3; closure = max(c1, c2)
    tp = [i for i, r_ in enumerate(recs) if r_["label"] == "they"]; th = [i for i, r_ in enumerate(recs) if r_["label"] != "they"]
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; pp = [i for i, r_ in enumerate(rows) if r_.present]; ps = [partner[(rows[i].construction, rows[i].group, False)] for i in pp]
    un = u / u.norm(); C = torch.tensor([L._single(t) for t in PRONOUNS]); UC = WU[C]; mC = UC.mean(0); VC = torch.linalg.svd(UC - mC, full_matrices=False).Vh[:8]; mCn = mC / mC.norm()
    dt = XFt[tp].mean(0) - XFt[th].mean(0); dp = (XFp[pp] - XFp[ps]).mean(0)
    def spectrum(d): return {"mean": float((d @ mCn) ** 2), "u": float((d @ un) ** 2), "VC": [float((d @ VC[i]) ** 2) for i in range(8)]}
    sp, st = spectrum(dp), spectrum(dt); cosu = [float(VC[i] @ un) for i in range(8)]; ip = max(range(8), key=lambda i: sp["VC"][i]); it_ = max(range(8), key=lambda i: st["VC"][i])
    report = {"closure_max": closure, "panel": sp, "text": st, "cos_VC_u": cosu, "cos_mean_u": float(mCn @ un), "panel_top_vc": ip, "text_top_vc": it_, "panel_top_share": sp["VC"][ip] / sum(sp["VC"])}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_pronoun_class_mean_carries_little": sp["mean"] <= MEAN_MAX * sp["u"], "pred_c_pronoun_residual_top_direction_is_number": abs(cosu[ip]) >= COS_MIN, "pred_d_number_direction_carries_most": report["panel_top_share"] >= CONC_MIN, "pred_e_same_on_text": ip == it_}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "subspace_pronoun_class_result_v397", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
