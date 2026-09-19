#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_vp0_carries_on_text pred_c_vp0_concentrated_on_text pred_d_late_mlps_write_vp0_on_text pred_e_readers_write_u_on_text
"""The agreement axis on natural text (v395). v390 / v391 (panel): the plural-class residual's top unembedding direction VP0 carries as much of the final-state
number contrast as the pronoun direction and is written by MLPs 12-17 and head 11.3. OOD on the 128 natural rows (labelled they / he; final-state difference =
they-labelled rows minus he-labelled rows, unpaired means): energy along VP0, along u, and over the class's top-32 residual directions; writers along VP0 and u.
PREDICTIONS (scored as written; failures preserved; priors from v390 / v391)
    pred_a_closure                   per-writer final-token writes sum to the final residual within relative 1e-4 (instrument)
    pred_b_vp0_carries_on_text       energy along VP0 >= 0.5x energy along u on text
    pred_c_vp0_concentrated_on_text  VP0 holds >= 0.50 of the final-state difference's energy within the plural class's top-32 residual directions
    pred_d_late_mlps_write_vp0_on_text  MLPs 12-17 together carry >= 0.50 of the writers' summed |contrast| along VP0
    pred_e_readers_write_u_on_text   readers 9.6 / 12.4 / 15.1 together carry >= 0.35 of the summed |contrast| along u (panel 0.44)
PRICE (registered maximum): 2 natural batches (blocks 0-17) = 2 forwards + one SVD; 0 backwards; 0 fits. Bar <= 4.
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
OUT = ROOT / "circuits/followups/subspace_vp0_natural_v395_result.json"
CANDIDATE_ID = "subspace.vp0_natural_v395"
K, N_HEAD, BATCH = 32, 9, 32
CLOSURE_TOL, RATIO_MIN, CONC_MIN, LATE_MIN, READER_MIN, LATE = 1e-4, 0.5, 0.50, 0.50, 0.35, (12, 13, 14, 15, 16, 17)
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_vp0_carries_on_text": ">= 0.5x u", "pred_c_vp0_concentrated_on_text": ">= 0.50 of top-32", "pred_d_late_mlps_write_vp0_on_text": ">= 0.50", "pred_e_readers_write_u_on_text": ">= 0.35"}
READERS = ("9.6", "12.4", "15.1")


def main() -> None:
    rows, he, she, agents, objects = g.build(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "k": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "ratio_min": RATIO_MIN, "conc_min": CONC_MIN, "late_min": LATE_MIN, "reader_min": READER_MIN}, "natural_rows": len(recs)}
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
        nat = torch.tensor([r_["ids"] for r_ in recs], device="cuda"); fin = [len(r_["ids"]) - 1 for r_ in recs]
        for start in range(0, len(recs), 64):
            tokens = nat[start:start + 64]; idx = torch.arange(tokens.shape[0]); pf = torch.tensor(fin[start:start + 64])
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
    plural = [i for i, r_ in enumerate(recs) if r_["label"] == "they"]; sing = [i for i, r_ in enumerate(recs) if r_["label"] != "they"]
    diff = XF[plural].mean(0) - XF[sing].mean(0); un = u / u.norm()
    eP = [float((diff @ Vk[i]) ** 2) for i in range(K)]; eU = float((diff @ un) ** 2); conc = eP[0] / sum(eP)
    contr = {name: {k: float((W[k][plural].mean(0) - W[k][sing].mean(0)) @ d) for k in W} for name, d in (("VP0", Vk[0]), ("u", un))}
    tot = {name: sum(abs(v) for v in c_.values()) for name, c_ in contr.items()}
    late_share = sum(abs(contr["VP0"][f"mlp{l}"]) for l in LATE) / tot["VP0"]; reader_share = sum(abs(contr["u"][k]) for k in READERS) / tot["u"]
    tops = {name: sorted(c_, key=lambda k: -abs(c_[k]))[:8] for name, c_ in contr.items()}
    report = {"closure_max": closure, "energy_vp0": eP[0], "energy_u": eU, "energy_top32_plural_dirs": eP, "vp0_concentration": conc, "late_mlp_share_vp0": late_share, "reader_share_u": reader_share, "writers_top8": {name: {k: contr[name][k] for k in tops[name]} for name in contr}}
    print(json.dumps({k: v for k, v in report.items() if k != "energy_top32_plural_dirs"}, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_vp0_carries_on_text": eP[0] >= RATIO_MIN * eU, "pred_c_vp0_concentrated_on_text": conc >= CONC_MIN, "pred_d_late_mlps_write_vp0_on_text": late_share >= LATE_MIN, "pred_e_readers_write_u_on_text": reader_share >= READER_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "subspace_vp0_natural_result_v395", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
