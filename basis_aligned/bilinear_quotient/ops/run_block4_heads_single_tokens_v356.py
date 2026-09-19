#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_head_sum_closure pred_b_head_4_5_carries_the_largest_number_contrast pred_c_head_4_5_separates_pairs pred_d_number_is_concentrated_in_few_heads pred_e_sign_matches_panel
"""Block-4 heads from single tokens (v356; the verb-copy stage of §4.8 with the single-token instrument). §4.8 named head 4.5 as the shared subject-feature
copier whose number part is one 128-d axis (v205-v215). Each of the 256 vocabulary pairs' nouns alone through blocks 0-4: the per-head write of block 4
(c_proj applied to each head's slice) projected on head 9.6's they - he reader direction (v168's instrument: r = V^T O^T (u_they - u_he) for 9.6, applied
to block 4's write as the state 9.6 reads), plural - singular contrast per head in pooled std, and head shares of the summed |contrast|.
PREDICTIONS (scored as written; failures preserved; priors from v205-v215)
    pred_a_head_sum_closure                        the nine per-head writes sum to block 4's attention output within relative 1e-4
    pred_b_head_4_5_carries_the_largest_number_contrast  head 4.5 has the largest |pooled plural - singular contrast| of the nine
    pred_c_head_4_5_separates_pairs                head 4.5's projection separates plural from singular nouns by >= 1 pooled std
    pred_d_number_is_concentrated_in_few_heads     the top 2 heads carry >= 0.60 of the summed |contrast|
    pred_e_sign_matches_panel                      head 4.5's contrast has the sign that favours "they" for plural nouns (positive on the they - he reader). Prior: unsure about the sign convention of r; reported.
PRICE (registered maximum): 512 single tokens / 256 = 2 forwards (blocks 0-4); 0 backwards; 0 fits. Bar <= 4.
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
OUT = ROOT / "circuits/followups/block4_heads_single_tokens_v356_result.json"
CANDIDATE_ID = "pronoun_number.block4_heads_single_tokens_v356"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, GAP_STD, TOP2_MIN = 1e-4, 1.0, 0.60
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_head_sum_closure": "<= 1e-4", "pred_b_head_4_5_carries_the_largest_number_contrast": "rank 1 of 9", "pred_c_head_4_5_separates_pairs": ">= 1 std", "pred_d_number_is_concentrated_in_few_heads": "top-2 >= 0.60", "pred_e_sign_matches_panel": "positive"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; tokens = sorted({t for p in pairs for t in p})
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gap_std": GAP_STD, "top2_min": TOP2_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    attn = blocks[LAYER].attn; D = model.config.n_embd; hd = D // N_HEAD; Wp = attn.c_proj.weight.detach().float()
    proj, closure = [], 0.0
    with torch.no_grad():
        for s0 in range(0, len(tokens), 256):
            ids = torch.tensor(tokens[s0:s0 + 256], device="cuda").unsqueeze(1); x = F.rms_norm(model.transformer.wte(ids), (D,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                if l == LAYER:
                    captured = {}
                    hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                    try: attention, v1_ = block.attn(xin_a, v1_)
                    finally: hook.remove()
                    y = captured["y"][:, 0].float()                                   # [B, D] concatenated heads
                    per_head = torch.stack([(y[:, h * hd:(h + 1) * hd] @ Wp[:, h * hd:(h + 1) * hd].T) for h in range(N_HEAD)], 1)   # [B, 9, D]
                    closure = max(closure, float(((per_head.sum(1) - attention[:, 0].float()).norm(dim=1) / attention[:, 0].float().norm(dim=1)).max()))
                    proj.append((per_head @ r.to(per_head.device)).cpu()); break
                attention, v1_ = block.attn(xin_a, v1_); x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
            forwards += 1
    P = torch.cat(proj); tindex = {t: i for i, t in enumerate(tokens)}; si = torch.tensor([tindex[a] for a, _ in pairs]); pi = torch.tensor([tindex[b] for _, b in pairs])
    per = {}
    for h in range(N_HEAD):
        s_, p_ = P[si, h], P[pi, h]; std = float(torch.cat([s_ - s_.mean(), p_ - p_.mean()]).std()); per[f"4.{h}"] = {"pooled_contrast": float((p_ - s_).sum()), "gap_over_std": float((p_.median() - s_.median()) / std), "plural_side_fraction": float(((p_ - s_) > 0).float().mean())}
    total = sum(abs(v["pooled_contrast"]) for v in per.values()); order = sorted(per, key=lambda k: -abs(per[k]["pooled_contrast"])); shares = {k: abs(per[k]["pooled_contrast"]) / total for k in order}
    report = {"closure_max": closure, "per_head": per, "order": order, "shares": shares, "top2_share": sum(list(shares.values())[:2])}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_head_sum_closure": closure <= CLOSURE_TOL, "pred_b_head_4_5_carries_the_largest_number_contrast": order[0] == "4.5", "pred_c_head_4_5_separates_pairs": abs(per["4.5"]["gap_over_std"]) >= GAP_STD,
                   "pred_d_number_is_concentrated_in_few_heads": report["top2_share"] >= TOP2_MIN, "pred_e_sign_matches_panel": per["4.5"]["pooled_contrast"] > 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "block4_heads_single_tokens_result_v356", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
