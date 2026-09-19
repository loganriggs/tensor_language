#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_vp0_energy_concentrated pred_c_vp0_writers_are_the_readers pred_d_vp0_writer_ranking_differs_from_u pred_e_singular_class_top_direction_also_carries
"""Fold back from the plural class's top idiosyncratic direction (v391). v390: the plural - singular final-state difference on the v76 rows carries 5.0M of energy
along VP0 -- the top right singular vector of the plural nouns' unembedding residual (plural nouns vs is / has / goes) -- against 4.8M along u_they - u_he and 10k
along the class mean contrast. Here: the energy spectrum of the final-state difference over the plural class's residual directions VP0..VP31 and the singular
class's VS0..VS31; the writers (embedding, heads of blocks 9-15, MLPs, other attention totals) along VP0, VP1, VS0 and u; whether the same readers carry VP0 as
carry u, and how the rankings differ.
PREDICTIONS (scored as written; failures preserved; priors from v390)
    pred_a_closure                          per-writer final-token writes sum to the final residual within relative 1e-4 (instrument)
    pred_b_vp0_energy_concentrated          VP0 carries >= 0.60 of the final-state difference's energy within the plural class's top-32 residual directions
    pred_c_vp0_writers_are_the_readers      the readers 9.6 / 12.4 / 15.1 together carry >= 0.50 of the writers' summed |contrast| along VP0
    pred_d_vp0_writer_ranking_differs_from_u  the top-3 writers along VP0 are not the same set as the top-3 along u. Prior: unsure.
    pred_e_singular_class_top_direction_also_carries  VS0 carries >= 0.5x VP0's energy of the final-state difference
PRICE (registered maximum): 3 row batches x 1 forward = 3 forwards + two SVDs; 0 backwards; 0 fits. Bar <= 5.
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
OUT = ROOT / "circuits/followups/subspace_vp0_writers_v391_result.json"
CANDIDATE_ID = "subspace.vp0_writers_v391"
K, N_HEAD, BATCH = 32, 9, 32
CLOSURE_TOL, CONC_MIN, READER_MIN, VS_MIN = 1e-4, 0.60, 0.50, 0.5
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_vp0_energy_concentrated": ">= 0.60 of top-32", "pred_c_vp0_writers_are_the_readers": ">= 0.50", "pred_d_vp0_writer_ranking_differs_from_u": "top-3 sets differ", "pred_e_singular_class_top_direction_also_carries": ">= 0.5x"}
READERS = ("9.6", "12.4", "15.1")


def main() -> None:
    rows, he, she, agents, objects = g.build(); pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "pairs": len(pairs), "k": K, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "conc_min": CONC_MIN, "reader_min": READER_MIN, "vs_min": VS_MIN}}
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
                x = live + attention; m = block.mlp(F.rms_norm(x, (D,))); x = x + m; parts[f"mlp{l}"] = m.clone()
            for k, v in parts.items(): add(k, v[idx, pf].float().cpu())
            XF.append(x[idx, pf].float().cpu()); forwards += 1
    W = {k: torch.cat(v) for k, v in W.items()}; XF = torch.cat(XF); recon = sum(W.values()); closure = float(((recon - XF).norm(dim=1) / XF.norm(dim=1)).max())
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r_ in enumerate(rows) if r_.present]; sing = [partner[(rows[i].construction, rows[i].group, False)] for i in plural]
    diff = (XF[plural] - XF[sing]).mean(0); un = u / u.norm()
    eP = [float((diff @ Vk[i]) ** 2) for i in range(K)]; eS = [float((diff @ VkS[i]) ** 2) for i in range(K)]; conc = eP[0] / sum(eP)
    dirs = {"VP0": Vk[0], "VP1": Vk[1], "VS0": VkS[0], "u": un}
    contr = {name: {k: float(((W[k][plural] - W[k][sing]).mean(0)) @ d) for k in W} for name, d in dirs.items()}
    tops = {name: sorted(c_, key=lambda k: -abs(c_[k]))[:8] for name, c_ in contr.items()}
    reader_share = {name: sum(abs(c_[k]) for k in READERS) / sum(abs(v) for v in c_.values()) for name, c_ in contr.items()}
    report = {"closure_max": closure, "energy_plural_dirs_top32": eP, "energy_singular_dirs_top32": eS, "vp0_concentration": conc, "energy_on_u": float((diff @ un) ** 2), "total_energy": float(diff.norm() ** 2), "cos_VP0_VS0": float(Vk[0] @ VkS[0]), "cos_VP0_u": float(Vk[0] @ un),
              "writers_top8": {name: {k: contr[name][k] for k in tops[name]} for name in dirs}, "reader_share": reader_share}
    print(json.dumps({k: v for k, v in report.items() if k not in ("energy_plural_dirs_top32", "energy_singular_dirs_top32")}, indent=1)); print("eP top8", [round(x) for x in eP[:8]], "eS top8", [round(x) for x in eS[:8]])
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_vp0_energy_concentrated": conc >= CONC_MIN, "pred_c_vp0_writers_are_the_readers": reader_share["VP0"] >= READER_MIN, "pred_d_vp0_writer_ranking_differs_from_u": set(tops["VP0"][:3]) != set(tops["u"][:3]), "pred_e_singular_class_top_direction_also_carries": eS[0] >= VS_MIN * eP[0]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "subspace_vp0_writers_result_v391", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
