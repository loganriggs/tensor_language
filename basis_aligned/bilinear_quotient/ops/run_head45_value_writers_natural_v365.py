#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_writer_closure pred_b_mlps_1_to_3_carry_on_text pred_c_mlp3_largest_on_text pred_d_attention_small_on_text pred_e_shares_within_020_of_panel
"""Writers of head 4.5's number value at natural cue nouns (v365). v360 (frames "The X"): MLP 3 0.52, MLP 2 0.18, embedding 0.16, MLP 1 0.11, attention 0.02 of
4.5's own-key value contrast on 9.6's reader. v364: 4.5 leads at real cue nouns. Same exact writer fold at the cue noun of the 128 natural rows (plural-cue
minus singular-cue means), own-key term.
PREDICTIONS (scored as written; failures preserved; priors from v360)
    pred_a_writer_closure           the writer terms reproduce the own-key term within relative 1e-3 on every row
    pred_b_mlps_1_to_3_carry_on_text  MLP 1 + MLP 2 + MLP 3 carry >= 0.60 of the own-key contrast
    pred_c_mlp3_largest_on_text     MLP 3 is the single largest writer
    pred_d_attention_small_on_text  attention 0-3 together <= 0.25 of the summed |contrast|
    pred_e_shares_within_020_of_panel  each of MLP 1 / 2 / 3's shares is within 0.20 of its frame value (0.11 / 0.18 / 0.52). Prior: unsure.
PRICE (registered maximum): 2 natural batches (blocks 0-4) = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
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
OUT = ROOT / "circuits/followups/head45_value_writers_natural_v365_result.json"
CANDIDATE_ID = "pronoun_number.head45_value_writers_natural_v365"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, MLP_MIN, ATTN_MAX, NEAR, HEAD, BATCH = 1e-3, 0.60, 0.25, 0.20, 5, 64
PANEL = {"mlp1": 0.109, "mlp2": 0.181, "mlp3": 0.52}
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_writer_closure": "<= 1e-3", "pred_b_mlps_1_to_3_carry_on_text": ">= 0.60", "pred_c_mlp3_largest_on_text": "largest", "pred_d_attention_small_on_text": "<= 0.25", "pred_e_shares_within_020_of_panel": "within 0.20 x 3"}


def main() -> None:
    rows, he, she, agents, objects = g.build(); recs = [r_ for p_ in NATURAL for r_ in json.loads(p_.read_text())["rows"]]
    plan = {"candidate_id": CANDIDATE_ID, "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "mlp_min": MLP_MIN, "attn_max": ATTN_MAX, "near": NEAR}, "head": HEAD, "natural_rows": len(recs)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    attn = blocks[LAYER].attn; D = model.config.n_embd; hd = D // N_HEAD; Wp = attn.c_proj.weight.detach().float()
    closure = 0.0; nat = torch.tensor([r_["ids"] for r_ in recs], device="cuda"); cue = [int(r_["cue_offset"]) for r_ in recs]
    Wo_h = Wp[:, HEAD * hd:(HEAD + 1) * hd]; Wv_h = attn.c_v.weight.detach().float()[HEAD * hd:(HEAD + 1) * hd]; store = {}
    def wrap(orig):
        def f(q, k, v, q2, k2):
            B, T, H, Dh = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dh) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dh)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0); store["pat"] = pat[:, HEAD].float(); store["v"] = v[:, :, HEAD].float(); return orig(q, k, v, q2, k2)
        return f
    W = {"embedding": [], "attn0": [], "mlp0": [], "attn1": [], "mlp1": [], "attn2": [], "mlp2": [], "attn3": [], "mlp3": []}; terms = {k: [] for k in W}; own_all = []
    with torch.no_grad():
        for s0 in range(0, len(recs), BATCH):
            ids = nat[s0:s0 + BATCH]; idx = torch.arange(ids.shape[0], device=ids.device); pn = torch.tensor(cue[s0:s0 + BATCH], device=ids.device)
            x = F.rms_norm(model.transformer.wte(ids), (D,)); x0, v1_ = x, None; parts = {k: torch.zeros_like(x) for k in W}; parts["embedding"] = x.clone()
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                for k in parts: parts[k] = block.lambdas[0] * parts[k]
                parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                xin_a = F.rms_norm(live, (D,))
                if l == LAYER:
                    orig = block.attn.squared_attention; block.attn.squared_attention = wrap(orig)
                    try: attention, v1_ = block.attn(xin_a, v1_)
                    finally: block.attn.squared_attention = orig
                    pat, v = store["pat"], store["v"]; pXX = pat[idx, pn, pn]; rms = live[idx, pn].float().pow(2).mean(-1, keepdim=True).sqrt(); lam = float(block.attn.lamb)
                    own = (pXX.unsqueeze(1) * v[idx, pn]) @ Wo_h.T; recon = torch.zeros_like(own)
                    for k, part in parts.items():
                        vpart = (1 - lam) * (part[idx, pn].float() / rms) @ Wv_h.T
                        if k == "embedding": vpart = vpart + lam * v1_[idx, pn, HEAD].float()
                        t_ = (pXX.unsqueeze(1) * vpart) @ Wo_h.T; recon = recon + t_; terms[k].append((t_ @ r.to(t_.device)).cpu())
                    closure = max(closure, float(((recon - own).norm(dim=1) / own.norm(dim=1)).max())); own_all.append((own @ r.to(own.device)).cpu()); break
                attention, v1_ = block.attn(xin_a, v1_); x = live + attention; parts[f"attn{l}"] = parts[f"attn{l}"] + attention
                m = block.mlp(F.rms_norm(x, (D,))); x = x + m; parts[f"mlp{l}"] = parts[f"mlp{l}"] + m
            forwards += 1
    pi = torch.tensor([i for i, r_ in enumerate(recs) if r_["cue"] == "plural"]); si = torch.tensor([i for i, r_ in enumerate(recs) if r_["cue"] != "plural"])
    own_all = torch.cat(own_all); c_own = float(own_all[pi].mean() - own_all[si].mean()); per = {k: float(torch.cat(v)[pi].mean() - torch.cat(v)[si].mean()) for k, v in terms.items()}
    total = sum(abs(v) for v in per.values()); share = {k: v / c_own for k, v in per.items()}; abs_share = {k: abs(v) / total for k, v in per.items()}
    report = {"closure_max": closure, "own_key_contrast": c_own, "writer_contrast": per, "writer_share_of_own": share, "writer_abs_share": abs_share, "mlp123_share": share["mlp1"] + share["mlp2"] + share["mlp3"], "attn_abs_share": sum(abs_share[f"attn{i}"] for i in range(4)),
              "largest_positive_writer": max(per, key=per.get)}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_writer_closure": closure <= CLOSURE_TOL, "pred_b_mlps_1_to_3_carry_on_text": report["mlp123_share"] >= MLP_MIN, "pred_c_mlp3_largest_on_text": report["largest_positive_writer"] == "mlp3",
                   "pred_d_attention_small_on_text": report["attn_abs_share"] <= ATTN_MAX, "pred_e_shares_within_020_of_panel": all(abs(share[k] - PANEL[k]) <= NEAR for k in PANEL)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "head45_value_writers_natural_result_v365", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
