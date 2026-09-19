#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_vp1_is_the_text_axis pred_c_vp1_writers_are_readers pred_d_vp1_on_panel_too pred_e_vp1_energy_small_vs_u
"""The class-residual direction that does carry on text: VP1 (v396). v395: on the 128 natural rows the labelled number contrast has 594k of energy along the
plural class's second residual direction VP1 (abstract plurals: efforts / actions / attempts vs agent plurals: astronauts / bees / architects) against 24k on VP0
and 9.8M on u. Here VP1's writers on text and on the panel, and the panel's energy along VP1 (v390 gave 0.23M), to judge whether VP1 is a real second axis of
the number state on both, or another artefact.
PREDICTIONS (scored as written; failures preserved; priors from v390 / v395)
    pred_a_closure               per-writer final-token writes sum to the final residual within relative 1e-4 (instrument), text and panel
    pred_b_vp1_is_the_text_axis  on text VP1 carries >= 0.50 of the energy within the plural class's top-32 residual directions
    pred_c_vp1_writers_are_readers  on text the readers 9.6 / 12.4 / 15.1 carry >= 0.35 of the writers' summed |contrast| along VP1 (VP1 rides with the pronoun circuit). Prior: unsure.
    pred_d_vp1_on_panel_too      on the panel VP1 carries >= 0.05 of the class-residual top-32 energy (v390: 0.23M of ~5.4M)
    pred_e_vp1_energy_small_vs_u on text VP1 carries <= 0.10 of u's energy (it is a side axis, not a rival)
PRICE (registered maximum): 2 natural batches + 3 panel batches (blocks 0-17) = 5 forwards + one SVD; 0 backwards; 0 fits. Bar <= 7.
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
OUT = ROOT / "circuits/followups/subspace_vp1_v396_result.json"
CANDIDATE_ID = "subspace.vp1_v396"
K, N_HEAD, BATCH = 32, 9, 32
CLOSURE_TOL, CONC_MIN, READER_MIN, PANEL_MIN, SIDE_MAX = 1e-4, 0.50, 0.35, 0.05, 0.10
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 7
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_vp1_is_the_text_axis": ">= 0.50 of top-32", "pred_c_vp1_writers_are_readers": ">= 0.35", "pred_d_vp1_on_panel_too": ">= 0.05", "pred_e_vp1_energy_small_vs_u": "<= 0.10 of u"}
READERS = ("9.6", "12.4", "15.1")


def main() -> None:
    rows, he, she, agents, objects = g.build(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "k": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "conc_min": CONC_MIN, "reader_min": READER_MIN, "panel_min": PANEL_MIN, "side_max": SIDE_MAX}, "natural_rows": len(recs)}
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
    un = u / u.norm(); vp1 = Vk[1]
    dt = XFt[tp].mean(0) - XFt[th].mean(0); dp = (XFp[pp] - XFp[ps]).mean(0)
    eT = [float((dt @ Vk[i]) ** 2) for i in range(K)]; eP = [float((dp @ Vk[i]) ** 2) for i in range(K)]
    contr_t = {k: float((Wt[k][tp].mean(0) - Wt[k][th].mean(0)) @ vp1) for k in Wt}; tot_t = sum(abs(v) for v in contr_t.values()); reader_t = sum(abs(contr_t[k]) for k in READERS) / tot_t
    contr_p = {k: float(((Wp_[k][pp] - Wp_[k][ps]).mean(0)) @ vp1) for k in Wp_}
    report = {"closure_max": closure, "text": {"energy_vp1": eT[1], "energy_u": float((dt @ un) ** 2), "vp1_share_top32": eT[1] / sum(eT), "writers_vp1_top8": {k: contr_t[k] for k in sorted(contr_t, key=lambda k: -abs(contr_t[k]))[:8]}, "reader_share_vp1": reader_t},
              "panel": {"energy_vp1": eP[1], "vp1_share_top32": eP[1] / sum(eP), "writers_vp1_top8": {k: contr_p[k] for k in sorted(contr_p, key=lambda k: -abs(contr_p[k]))[:8]}}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_vp1_is_the_text_axis": report["text"]["vp1_share_top32"] >= CONC_MIN, "pred_c_vp1_writers_are_readers": reader_t >= READER_MIN, "pred_d_vp1_on_panel_too": report["panel"]["vp1_share_top32"] >= PANEL_MIN, "pred_e_vp1_energy_small_vs_u": eT[1] <= SIDE_MAX * report["text"]["energy_u"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "subspace_vp1_result_v396", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
