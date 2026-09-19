#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_key_split_closure pred_b_own_key_term_replays_alone_sign pred_c_The_key_term_carries_the_flip pred_d_The_key_term_dominates pred_e_pattern_weight_on_The_differs_by_number
"""What flips head 4.5: the value it copies from "The", or its own-key value? (v359). v358: 4.5's number write on 9.6's reader is negative for plural
nouns alone and strongly positive after "The". A head's write at X is sum over keys j of p(X, j) . O V x_j; with the frame ["The", X] there are two keys.
Split 4.5's write at X into the own-key term p(X,X) . OV x_X and the "The"-key term p(X,The) . OV x_The, each projected on r, plural - singular over the
256 pairs; also the pattern weights p(X,X), p(X,The) by noun number. Exact: the two terms sum to the head's write.
PREDICTIONS (scored as written; failures preserved; priors from v358)
    pred_a_key_split_closure               own-key term + "The"-key term = head 4.5's write within relative 1e-4 on every row
    pred_b_own_key_term_replays_alone_sign the own-key term's pooled plural - singular contrast is negative (the single-token sign)
    pred_c_The_key_term_carries_the_flip   the "The"-key term's pooled contrast is positive and larger in magnitude than the own-key term's
    pred_d_The_key_term_dominates          the "The"-key term carries >= 0.60 of the summed |contrast| of the two terms
    pred_e_pattern_weight_on_The_differs_by_number  the median pattern weight p(X, The) differs between plural and singular X by >= 0.5 pooled std (the flip is pattern-borne: plural nouns attend to "The" differently). Prior: unsure -- the alternative is a value-borne flip through x_The's own value being read with a number-dependent query.
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
OUT = ROOT / "circuits/followups/head45_key_split_v359_result.json"
CANDIDATE_ID = "pronoun_number.head45_key_split_v359"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, DOM_MIN, PAT_STD, HEAD = 1e-4, 0.60, 0.5, 5
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_key_split_closure": "<= 1e-4", "pred_b_own_key_term_replays_alone_sign": "negative", "pred_c_The_key_term_carries_the_flip": "positive, larger", "pred_d_The_key_term_dominates": ">= 0.60", "pred_e_pattern_weight_on_The_differs_by_number": ">= 0.5 std"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; tokens = sorted({t for p in pairs for t in p})
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "dom_min": DOM_MIN, "pat_std": PAT_STD}, "frame": "The X", "head": HEAD}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    attn = blocks[LAYER].attn; D = model.config.n_embd; hd = D // N_HEAD; Wp = attn.c_proj.weight.detach().float()
    closure = 0.0; tindex = {t: i for i, t in enumerate(tokens)}; si = torch.tensor([tindex[a] for a, _ in pairs]); pi = torch.tensor([tindex[b] for _, b in pairs])
    the = L._single("The"); own_t, the_t, p_own, p_the = [], [], [], []
    Wv = attn.c_v.weight.detach().float(); Wo_h = Wp[:, HEAD * hd:(HEAD + 1) * hd]; store = {}
    def wrap(orig):
        def f(q, k, v, q2, k2):
            B, T, H, Dh = q.shape; pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dh) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dh)
            pat = pat.masked_fill(torch.tril(torch.ones(T, T, device=pat.device, dtype=torch.bool)).logical_not(), 0.0); store["pat"] = pat[:, HEAD].float(); store["v"] = v[:, :, HEAD].float(); return orig(q, k, v, q2, k2)
        return f
    with torch.no_grad():
        for s0 in range(0, len(tokens), 256):
            ids = torch.tensor([[the, t] for t in tokens[s0:s0 + 256]], device="cuda"); x = F.rms_norm(model.transformer.wte(ids), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                if l == LAYER:
                    orig = block.attn.squared_attention; block.attn.squared_attention = wrap(orig)
                    captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                    try: attention, v1_ = block.attn(xin_a, v1_)
                    finally: hook.remove(); block.attn.squared_attention = orig
                    y = captured["y"][:, 1].float(); head_write = y[:, HEAD * hd:(HEAD + 1) * hd] @ Wo_h.T             # the head's total write at X
                    pat, v = store["pat"], store["v"]                                                                   # pat [B, T, T]; v [B, T, hd] (already lambda-mixed with v1)
                    own = (pat[:, 1, 1].unsqueeze(1) * v[:, 1]) @ Wo_h.T; frm = (pat[:, 1, 0].unsqueeze(1) * v[:, 0]) @ Wo_h.T
                    closure = max(closure, float(((own + frm - head_write).norm(dim=1) / head_write.norm(dim=1)).max()))
                    own_t.append((own @ r.to(own.device)).cpu()); the_t.append((frm @ r.to(frm.device)).cpu()); p_own.append(pat[:, 1, 1].cpu()); p_the.append(pat[:, 1, 0].cpu()); break
                attention, v1_ = block.attn(xin_a, v1_); x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
    own_t, the_t, p_own, p_the = torch.cat(own_t), torch.cat(the_t), torch.cat(p_own), torch.cat(p_the)
    c_own, c_the = float((own_t[pi] - own_t[si]).sum()), float((the_t[pi] - the_t[si]).sum()); std_p = float(torch.cat([p_the[pi] - p_the[pi].mean(), p_the[si] - p_the[si].mean()]).std())
    report = {"closure_max": closure, "own_key_contrast": c_own, "The_key_contrast": c_the, "The_key_share": abs(c_the) / (abs(c_own) + abs(c_the)), "pattern_The_median": {"plural": float(p_the[pi].median()), "singular": float(p_the[si].median())},
              "pattern_own_median": {"plural": float(p_own[pi].median()), "singular": float(p_own[si].median())}, "pattern_The_gap_std": float((p_the[pi].median() - p_the[si].median()) / std_p)}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_key_split_closure": closure <= CLOSURE_TOL, "pred_b_own_key_term_replays_alone_sign": c_own < 0, "pred_c_The_key_term_carries_the_flip": c_the > 0 and abs(c_the) > abs(c_own), "pred_d_The_key_term_dominates": report["The_key_share"] >= DOM_MIN, "pred_e_pattern_weight_on_The_differs_by_number": abs(report["pattern_The_gap_std"]) >= PAT_STD}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "head45_key_split_result_v359", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
