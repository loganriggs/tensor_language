#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_unit_closure pred_b_agreement_units_in_top10 pred_c_top10_carry_half pred_d_3465_sign_positive_on_r pred_e_mismatch_leaders_not_in_top10
"""Which MLP-3 units write the state head 4.5 copies? (v361). v360: MLP 3 supplies 52% of head 4.5's flipped number value at X after "The". MLP 3's write is
Down h + bias, so its term in 4.5's own-key value splits exactly by unit: (1 - lamb) p(X,X) . O_h V_h (lambda-chain . Down_j h_j / rms). Per unit, the plural -
singular contrast on r over the 256 pairs; top-k shares; ranks of the agreement units 3465 / 493 and of the mismatch leaders 3040 / 114 / 565 (which fire
at the verb, not at a noun after "The").
PREDICTIONS (scored as written; failures preserved; priors from v329-v354)
    pred_a_unit_closure                the per-unit terms sum to MLP 3's term within relative 1e-3 on every row
    pred_b_agreement_units_in_top10    3465 and 493 are both in the top 10 by |pooled contrast|
    pred_c_top10_carry_half            the top 10 units carry >= 0.50 of the summed |contrast|
    pred_d_3465_sign_positive_on_r     3465's pooled contribution is positive on r (its plural write, negative in its own units, maps to 'they' through 4.5)
    pred_e_mismatch_leaders_not_in_top10  3040, 114 and 565 are all outside the top 10 (they are verb-site units)
PRICE (registered maximum): 512 rows / 256 = 2 forwards (blocks 0-4); 0 backwards; 0 fits. Bar <= 4.
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
OUT = ROOT / "circuits/followups/head45_mlp3_units_v361_result.json"
CANDIDATE_ID = "pronoun_number.head45_mlp3_units_v361"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, TOP10_MIN, HEAD = 1e-3, 0.50, 5
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_unit_closure": "<= 1e-3", "pred_b_agreement_units_in_top10": "3465 and 493", "pred_c_top10_carry_half": ">= 0.50", "pred_d_3465_sign_positive_on_r": "positive", "pred_e_mismatch_leaders_not_in_top10": "outside top-10 x 3"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; tokens = sorted({t for p in pairs for t in p})
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "top10_min": TOP10_MIN}, "frame": "The X", "head": HEAD}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    attn = blocks[LAYER].attn; D = model.config.n_embd; hd = D // N_HEAD; Wp = attn.c_proj.weight.detach().float()
    closure = 0.0; tindex = {t: i for i, t in enumerate(tokens)}; si = torch.tensor([tindex[a] for a, _ in pairs]); pi = torch.tensor([tindex[b] for _, b in pairs])
    the = L._single("The"); Wo_h = Wp[:, HEAD * hd:(HEAD + 1) * hd]; Wv_h = attn.c_v.weight.detach().float()[HEAD * hd:(HEAD + 1) * hd]; lamb = float(attn.lamb) if hasattr(attn, "lamb") else None
    Wv0_h = blocks[0].attn.c_v.weight.detach().float()[HEAD * hd:(HEAD + 1) * hd]; store = {}
    def wrap(orig):
        def f(q, k, v, q2, k2):
            B, T, H, Dh = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dh) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dh)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0); store["pat"] = pat[:, HEAD].float(); store["v"] = v[:, :, HEAD].float(); return orig(q, k, v, q2, k2)
        return f
    mlp3 = blocks[3].mlp; Dw3 = mlp3.Down.weight.detach().float(); b3 = mlp3.Down_bias.detach().float(); unit_terms, mlp3_terms = [], []; closure = 0.0
    with torch.no_grad():
        for s0 in range(0, len(tokens), 256):
            ids = torch.tensor([[the, t] for t in tokens[s0:s0 + 256]], device="cuda"); x = F.rms_norm(model.transformer.wte(ids), (D,)); x0, v1_ = x, None; h3 = None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                if l == LAYER:
                    orig = block.attn.squared_attention; block.attn.squared_attention = wrap(orig)
                    try: attention, v1_ = block.attn(xin_a, v1_)
                    finally: block.attn.squared_attention = orig
                    pat = store["pat"]; pXX = pat[:, 1, 1]; rms = live[:, 1].float().pow(2).mean(-1, keepdim=True).sqrt(); lam = float(block.attn.lamb)
                    scale = float(block.lambdas[0])                                                                    # MLP 3's write reaches block 4's live through one lambda0
                    m3 = scale * (h3 @ Dw3.T + b3)                                                                      # [B, D] MLP 3's write at X (scaled)
                    term_total = (pXX.unsqueeze(1) * ((1 - lam) * (m3 / rms) @ Wv_h.T)) @ Wo_h.T                          # MLP 3's term in the own-key value
                    per_unit = (h3.unsqueeze(2) * Dw3.T.unsqueeze(0))                                                    # [B, 4608, D] unit writes (unscaled)
                    per_unit_r = ((pXX.unsqueeze(1) * (1 - lam) * scale / rms) * ((per_unit @ Wv_h.T) @ Wo_h.T @ r.to(per_unit.device)))   # [B, 4608] on r
                    bias_r = float(((pXX.unsqueeze(1) * (1 - lam) * scale * (b3.unsqueeze(0) / rms) @ Wv_h.T) @ Wo_h.T @ r.to(m3.device)).mean())
                    recon = per_unit_r.sum(1) + ((pXX * (1 - lam) * scale / rms.squeeze(1)) * float(((b3 @ Wv_h.T) @ Wo_h.T) @ r.to(m3.device)))
                    closure = max(closure, float(((recon - (term_total @ r.to(term_total.device))).abs() / (term_total @ r.to(term_total.device)).abs().clamp_min(1e-6)).max()))
                    unit_terms.append(per_unit_r.cpu()); break
                attention, v1_ = block.attn(xin_a, v1_); x = live + attention; xin = F.rms_norm(x, (D,))
                if l == 3: h3 = dod_units.hidden(model, block.mlp, xin)[:, 1].float()
                x = x + block.mlp(xin)
            forwards += 1
    U = torch.cat(unit_terms); tindex = {t: i for i, t in enumerate(tokens)}; si = torch.tensor([tindex[a] for a, _ in pairs]); pi = torch.tensor([tindex[b] for _, b in pairs])
    pooled = (U[pi] - U[si]).sum(0); order = torch.argsort(pooled.abs(), descending=True); total = float(pooled.abs().sum())
    rank = {str(u): int((pooled.abs() > pooled.abs()[u]).sum()) + 1 for u in (3465, 493, 3040, 114, 565)}
    report = {"closure_max": closure, "top12": [(int(j), float(pooled[j])) for j in order[:12]], "top10_share": float(pooled.abs()[order[:10]].sum() / total), "ranks": rank, "contrast_3465": float(pooled[3465]), "contrast_493": float(pooled[493])}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_unit_closure": closure <= CLOSURE_TOL, "pred_b_agreement_units_in_top10": rank["3465"] <= 10 and rank["493"] <= 10, "pred_c_top10_carry_half": report["top10_share"] >= TOP10_MIN, "pred_d_3465_sign_positive_on_r": report["contrast_3465"] > 0,
                   "pred_e_mismatch_leaders_not_in_top10": all(rank[u] > 10 for u in ("3040", "114", "565"))}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "head45_mlp3_units_result_v361", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
