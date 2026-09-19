#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_head_sum_closure pred_b_4_5_flips_sign_after_The pred_c_4_5_dominates_after_The pred_d_4_8_keeps_its_sign pred_e_flip_also_after_a
"""Head 4.5's sign flip: alone vs after a determiner (v358). v356: from the token alone 4.5's number write on 9.6's they - he reader is negative for
plural (31% of block 4's contrast, behind 4.8); v357: on the pronoun rows (nouns after "The") 4.5 carries ~90% with the POSITIVE, they-for-plural sign.
Frames on the 256 vocabulary pairs: X alone; "The X"; " a X" (a singular-licensing determiner); read at X; per-head projections on r, plural - singular.
PREDICTIONS (scored as written; failures preserved; priors from v356 / v357)
    pred_a_head_sum_closure     the nine per-head writes sum to block 4's attention output within relative 1e-4 in every frame
    pred_b_4_5_flips_sign_after_The  4.5's pooled contrast is negative alone and positive after "The"
    pred_c_4_5_dominates_after_The   after "The", 4.5 carries >= 0.60 of the summed |contrast|
    pred_d_4_8_keeps_its_sign   4.8's pooled contrast is negative in all three frames
    pred_e_flip_also_after_a    after " a", 4.5's pooled contrast is positive too (the flip is 'a determiner precedes', not 'a plural-licensing one'). Prior: unsure.
PRICE (registered maximum): 3 frames x 512 rows / 256 = 6 forwards (blocks 0-4); 0 backwards; 0 fits. Bar <= 8.
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
OUT = ROOT / "circuits/followups/head45_sign_flip_v358_result.json"
CANDIDATE_ID = "pronoun_number.head45_sign_flip_v358"
LAYER, N_HEAD = 4, 9
CLOSURE_TOL, DOM_MIN = 1e-4, 0.60
FRAMES = {"alone": [], "The": ["The"], "a": [" a"]}
FORWARDS_MAX = 8
PREDICTIONS = {"pred_a_head_sum_closure": "<= 1e-4", "pred_b_4_5_flips_sign_after_The": "neg alone, pos after The", "pred_c_4_5_dominates_after_The": ">= 0.60", "pred_d_4_8_keeps_its_sign": "negative x 3", "pred_e_flip_also_after_a": "positive after a"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    pairs = [(a, b) for _, a, b in v287.SPEC["noun_pairs_vocab"]][:256]; tokens = sorted({t for p in pairs for t in p})
    plan = {"candidate_id": CANDIDATE_ID, "pairs": len(pairs), "layer": LAYER, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "dom_min": DOM_MIN}, "frames": {k: v for k, v in FRAMES.items()}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    fw = L.ManualForward(backend)
    comp96 = next(c for c in dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, he, she, ((9, 6),)).set_components())
    fw.directions = L.readout_directions(model, (comp96,), he, she); r = L.reader_directions(model, comp96, fw.directions)[6].float().cpu()
    attn = blocks[LAYER].attn; D = model.config.n_embd; hd = D // N_HEAD; Wp = attn.c_proj.weight.detach().float()
    closure, report = 0.0, {}
    tindex = {t: i for i, t in enumerate(tokens)}; si = torch.tensor([tindex[a] for a, _ in pairs]); pi = torch.tensor([tindex[b] for _, b in pairs])
    for fname, pre_txt in FRAMES.items():
        pre = [L._single(t) for t in pre_txt]; k = len(pre); proj = []
        with torch.no_grad():
            for s0 in range(0, len(tokens), 256):
                ids = torch.tensor([pre + [t] for t in tokens[s0:s0 + 256]], device="cuda"); x = F.rms_norm(model.transformer.wte(ids), (D,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; xin_a = F.rms_norm(live, (D,))
                    if l == LAYER:
                        captured = {}; hook = block.attn.c_proj.register_forward_pre_hook(lambda m, args: captured.setdefault("y", args[0]))
                        try: attention, v1_ = block.attn(xin_a, v1_)
                        finally: hook.remove()
                        y = captured["y"][:, k].float(); per_head = torch.stack([(y[:, h * hd:(h + 1) * hd] @ Wp[:, h * hd:(h + 1) * hd].T) for h in range(N_HEAD)], 1)
                        closure = max(closure, float(((per_head.sum(1) - attention[:, k].float()).norm(dim=1) / attention[:, k].float().norm(dim=1)).max()))
                        proj.append((per_head @ r.to(per_head.device)).cpu()); break
                    attention, v1_ = block.attn(xin_a, v1_); x = live + attention; x = x + block.mlp(F.rms_norm(x, (D,)))
                forwards += 1
        P = torch.cat(proj); per = {f"4.{h}": float((P[pi, h] - P[si, h]).sum()) for h in range(N_HEAD)}; total = sum(abs(v) for v in per.values()); order = sorted(per, key=lambda kk: -abs(per[kk]))
        report[fname] = {"pooled_contrast": per, "order": order, "shares": {kk: abs(per[kk]) / total for kk in order}}
    report["closure_max"] = closure
    print(json.dumps({f_: {"order": v["order"][:3], "4.5": round(v["pooled_contrast"]["4.5"]), "4.8": round(v["pooled_contrast"]["4.8"]), "share_4.5": round(v["shares"]["4.5"], 3)} for f_, v in report.items() if f_ != "closure_max"}, indent=1))
    c = lambda f_, h: report[f_]["pooled_contrast"][h]
    predictions = {"pred_a_head_sum_closure": closure <= CLOSURE_TOL, "pred_b_4_5_flips_sign_after_The": c("alone", "4.5") < 0 < c("The", "4.5"), "pred_c_4_5_dominates_after_The": report["The"]["shares"]["4.5"] >= DOM_MIN,
                   "pred_d_4_8_keeps_its_sign": all(c(f_, "4.8") < 0 for f_ in FRAMES), "pred_e_flip_also_after_a": c("a", "4.5") > 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "head45_sign_flip_result_v358", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
