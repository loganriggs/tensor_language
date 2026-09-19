#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_writer_closure pred_b_mlps_1_to_3_carry_the_flip pred_c_embedding_keeps_the_alone_sign pred_d_mlp3_is_the_largest_writer pred_e_attention_writers_small
"""Which earlier writer flips head 4.5's own-key value after "The"? (v360). v359: 4.5's write at X after "The" is 99% its own-key term, which has the
opposite sign to the same term for X alone. The head's value is linear in block 4's normalised live input: v = W_v rms(live) (lambda-mixed with the block-0 value
v1, which is itself linear in x0), so the own-key term p(X,X) . O_h V_h v splits exactly by residual writer -- embedding (x0 and its lambda re-injections),
attention 0-3 and MLP 0-3 totals -- with the pattern weight and rms as per-row scalars. Plural - singular contrast on r per writer, for X after "The".
PREDICTIONS (scored as written; failures preserved; priors from v340 / v359)
    pred_a_writer_closure             the writer terms reproduce the own-key term within relative 1e-3 on every row
    pred_b_mlps_1_to_3_carry_the_flip MLP 1 + MLP 2 + MLP 3 together carry >= 0.60 of the own-key term's pooled positive contrast
    pred_c_embedding_keeps_the_alone_sign  the embedding terms' pooled contrast is negative (the single-token sign) or within 0.10 of zero in share
    pred_d_mlp3_is_the_largest_writer MLP 3 (the agreement units' block) is the single largest positive writer. Prior: unsure -- MLP 1's conditioned write may lead.
    pred_e_attention_writers_small    attention 0-3 together carry <= 0.25 of the summed |contrast|
PRICE (registered maximum): 512 rows / 256 = 2 forwards (blocks 0-4); 0 backwards; 0 fits. Bar <= 4.
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
OUT = ROOT / "circuits/followups/head45_value_writers_v360_result.json"
CANDIDATE_ID = "pronoun_number.head45_value_writers_v360"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, MLP_MIN, EMB_TOL, ATTN_MAX, HEAD = 1e-3, 0.60, 0.10, 0.25, 5
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_writer_closure": "<= 1e-3", "pred_b_mlps_1_to_3_carry_the_flip": ">= 0.60", "pred_c_embedding_keeps_the_alone_sign": "negative or |share| <= 0.10", "pred_d_mlp3_is_the_largest_writer": "largest positive", "pred_e_attention_writers_small": "<= 0.25"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; tokens = sorted({t for p in pairs for t in p})
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "mlp_min": MLP_MIN, "emb_tol": EMB_TOL, "attn_max": ATTN_MAX}, "frame": "The X", "head": HEAD}
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
    W = {"embedding": [], "attn0": [], "mlp0": [], "attn1": [], "mlp1": [], "attn2": [], "mlp2": [], "attn3": [], "mlp3": []}; terms = {k: [] for k in W}; own_all = []
    with torch.no_grad():
        for s0 in range(0, len(tokens), 256):
            ids = torch.tensor([[the, t] for t in tokens[s0:s0 + 256]], device="cuda"); x = F.rms_norm(model.transformer.wte(ids), (D,)); x0, v1_ = x, None
            parts = {k: torch.zeros_like(x) for k in W}; parts["embedding"] = x.clone()
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                for k in parts: parts[k] = block.lambdas[0] * parts[k]
                parts["embedding"] = parts["embedding"] + block.lambdas[1] * x0
                xin_a = F.rms_norm(live, (D,))
                if l == LAYER:
                    orig = block.attn.squared_attention; block.attn.squared_attention = wrap(orig)
                    try: attention, v1_ = block.attn(xin_a, v1_)
                    finally: block.attn.squared_attention = orig
                    pat, v = store["pat"], store["v"]; pXX = pat[:, 1, 1]; rms = live[:, 1].float().pow(2).mean(-1, keepdim=True).sqrt()
                    own = (pXX.unsqueeze(1) * v[:, 1]) @ Wo_h.T                                                   # exact own-key term
                    # writer split of the block-4 value: v = (1 - lamb) W_v (live/rms) + lamb v1 ; v1 = W_v0 rms(x0 stream at block 0) is embedding-only
                    lam = float(block.attn.lamb) if torch.is_tensor(block.attn.lamb) and block.attn.lamb.numel() == 1 else None
                    recon = torch.zeros_like(own)
                    for k, part in parts.items():
                        vpart = (1 - lam) * (part[:, 1].float() / rms) @ Wv_h.T if lam is not None else (part[:, 1].float() / rms) @ Wv_h.T
                        if k == "embedding" and lam is not None: vpart = vpart + lam * v1_[:, 1, HEAD * hd:(HEAD + 1) * hd].float() if v1_.dim() == 3 else vpart + lam * v1_[:, 1, HEAD].float()
                        t_ = (pXX.unsqueeze(1) * vpart) @ Wo_h.T; recon = recon + t_; terms[k].append((t_ @ r.to(t_.device)).cpu())
                    closure = max(closure, float(((recon - own).norm(dim=1) / own.norm(dim=1)).max())); own_all.append((own @ r.to(own.device)).cpu()); break
                attention, v1_ = block.attn(xin_a, v1_); x = live + attention; parts[f"attn{l}"] = parts[f"attn{l}"] + attention
                m = block.mlp(F.rms_norm(x, (D,))); x = x + m; parts[f"mlp{l}"] = parts[f"mlp{l}"] + m
            forwards += 1
    own_all = torch.cat(own_all); c_own = float((own_all[pi] - own_all[si]).sum()); per = {k: float((torch.cat(v)[pi] - torch.cat(v)[si]).sum()) for k, v in terms.items()}
    total = sum(abs(v) for v in per.values()); share = {k: v / c_own for k, v in per.items()}; abs_share = {k: abs(v) / total for k, v in per.items()}
    report = {"closure_max": closure, "own_key_contrast": c_own, "writer_contrast": per, "writer_share_of_own": share, "writer_abs_share": abs_share, "mlp123_share": share["mlp1"] + share["mlp2"] + share["mlp3"], "attn_abs_share": sum(abs_share[f"attn{i}"] for i in range(4)),
              "largest_positive_writer": max(per, key=per.get)}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_writer_closure": closure <= CLOSURE_TOL, "pred_b_mlps_1_to_3_carry_the_flip": report["mlp123_share"] >= MLP_MIN, "pred_c_embedding_keeps_the_alone_sign": per["embedding"] < 0 or abs(share["embedding"]) <= EMB_TOL,
                   "pred_d_mlp3_is_the_largest_writer": report["largest_positive_writer"] == "mlp3", "pred_e_attention_writers_small": report["attn_abs_share"] <= ATTN_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "head45_value_writers_result_v360", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
