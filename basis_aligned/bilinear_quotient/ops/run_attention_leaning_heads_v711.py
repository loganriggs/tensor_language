#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_single_head_inflates pred_c_inflating_head_is_layer8 pred_d_55_shares_cause pred_e_rank64_band_fixes
"""Attention lane, v711: inside band 6-8, WHICH head's program inflates 8.3 and 5.5 — and does rank 64 for the band fix it?

v709: with the frozen program applied to band 6-8 alone, 8.3's mean-ablation value is 3.2x native and 5.5's 2.5x; leaving the band native
removes both. This rung: (i) leave-one-head-native inside the band-6-8-alone configuration — for each of the 27 heads, restore it to native
and re-measure the band cost and the 8.3 / 5.5 ratios; (ii) the fix arm: the full program with every band-6-8 head at CLOSED-FORM rank 64
(v702 kernel kappa0, maps SVD-truncated at 64, kappa_r recomputed; no fit) — cost and the five ratios. Response only. CE ADDED, lower is
better; ratio = in-program value / native value (v701).
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_single_head_inflates    some single band-6-8 head restored to native lowers 8.3's ratio by >= 1.0 from the band-alone 3.21. Prior: unsure
    pred_c_inflating_head_is_layer8 the head whose restoration lowers 8.3's ratio most is in layer 8 (a layer-mate). Prior: unsure
    pred_d_55_shares_cause         that same head is also the one whose restoration lowers 5.5's ratio most. Prior: unsure
    pred_e_rank64_band_fixes       with band 6-8 at closed-form rank 64, the full-program cost <= 0.084 and 8.3's ratio <= 2.0. Prior: likely
PRICE (registered maximum): native 6; kappa_r 2 + 2; band-alone base 6 + 2 x 6 = 18; 27 x 18 = 486; rank-64 arm 6 + 5 x 6 = 36; total 556 forwards;
0 backwards; 0 fits. Bar <= 575.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_leaning_heads_v711_result.json"
PROGS702 = ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt"
PROGS706 = ROOT / "circuits/followups/attention_mixed16_manip_v706_programs.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.leaning_heads_v711"
FORWARDS_MAX = 575
BAND = (6, 7, 8); R64 = 64
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
FIVE = ((8, 3), (5, 5), (1, 4), (3, 5), (1, 1))
BANDS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14), (15, 16, 17))
REPLAY_TOL, DROP_MIN, COST_MAX, RATIO_MAX = 0.002, 1.0, 0.084, 2.0
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_single_head_inflates": "some head drops 8.3 ratio by >= 1.0", "pred_c_inflating_head_is_layer8": "argmin head in layer 8",
               "pred_d_55_shares_cause": "same head is argmin for 5.5", "pred_e_rank64_band_fixes": "cost <= 0.084 and 8.3 ratio <= 2.0"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bands": [list(b) for b in BANDS], "heads": [list(k) for k in FIVE],
            "bars": {"replay_tol": REPLAY_TOL, "drop_min": DROP_MIN, "cost_max": COST_MAX, "ratio_max": RATIO_MAX}, "band": list(BAND)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long(); fit64 = torch.load(FIT_ROWS, map_location="cpu").long()[:64]
    saved = torch.load(PROGS706, map_location=dev)["mixed16"]
    means = torch.load(MEANS, map_location=dev); v701 = json.load(open(V701))["report"]["mean_ablation_cost"]
    ALL = [(l, h) for l in LAYERS for h in range(H)]; simplified = [k for k in ALL if k not in FIVE]
    state = {"idx": None, "programs": {}, "n": {}, "ablate": set()}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}
    nhooks = [blocks[l].attn.register_forward_pre_hook(lambda m, a, l=l: state["n"].__setitem__(l, a[0])) for l in LAYERS]

    def make_patched(l):
        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat = pat.masked_fill(~causal, 0.0)
            pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            ctx = {"dmat": dmat, "off": off, "pos": pos, "idx": state["idx"], "attn": blocks[l].attn, "n": state["n"][l], "hd": hd, "causal": causal}
            cols = []
            for h in range(Hn):
                prog = state["programs"].get((l, h))
                cols.append(pat[:, h] if prog is None else AP.apply_program(pat[:, h], prog, ctx))
            pat = torch.stack(cols, 1)
            z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
            for (al, h) in state["ablate"]:
                if al == l:
                    z = z.clone(); z[:, h] = means[f"mean_{l}_{h}"]
            return z
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(progs):
        state["programs"] = progs; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["programs"] = {}
        return total / n, fw

    maps = {k: {n_: saved[f"{k[0]}.{k[1]}"][n_].to(dev).float() for n_ in ("c_q", "c_k", "c_q2", "c_k2")} for k in simplified}
    kappa = {k: saved[f"{k[0]}.{k[1]}"]["kappa"].to(dev).float() for k in simplified}
    acc = {k: 0 for k in simplified}; nb = 0
    with torch.no_grad():
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); state["programs"] = {}; model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
            Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            for k in simplified:
                acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps[k], hd, causal), dmat, off, Tn)
            nb += 1
    kr = {k: v / nb for k, v in acc.items()}
    full = {k: {"kind": "lowrank", "kappa": kappa[k], "kappa_r": kr[k], "maps": maps[k]} for k in simplified}

    def values_under(progs, heads):
        base, fw = ce(progs); out = {}
        for k in heads:
            state["ablate"] = {k}; c_, f_ = ce(progs); fw += f_; state["ablate"] = set()
            key = f"{k[0]}.{k[1]}"; out[key] = {"in_program_value": c_ - base, "ratio": (c_ - base) / v701[key]}
        return base, out, fw

    TWO = ((8, 3), (5, 5))
    native, fw = ce({}); forwards += fw
    band_heads = [k for k in simplified if k[0] in BAND]
    alone = {k: full[k] for k in band_heads}
    c_alone, r_alone, fw = values_under(alone, TWO); forwards += fw
    report = {"native": native, "band_alone": {"cost": c_alone - native, "heads": r_alone}, "leave_head": {}}
    print(f"native {native:.5f} | band 6-8 alone cost {c_alone - native:+.4f}; ratios 8.3 {r_alone['8.3']['ratio']:.2f} 5.5 {r_alone['5.5']['ratio']:.2f}")
    for k in band_heads:
        progs = {kk: p for kk, p in alone.items() if kk != k}
        c_, r_, fw = values_under(progs, TWO); forwards += fw
        key = f"{k[0]}.{k[1]}"; report["leave_head"][key] = {"cost": c_ - native, "heads": r_}
        print(f"band 6-8 alone, {key} native: cost {c_ - native:+.4f} | 8.3 ratio {r_['8.3']['ratio']:.2f} | 5.5 ratio {r_['5.5']['ratio']:.2f}")
    # ---- fix arm: full program, band 6-8 at closed-form rank 64 -------------------------------------------------------------------------
    progs702 = torch.load(PROGS702, map_location=dev)
    maps64 = {k: AP.truncated_maps(blocks[k[0]].attn, k[1], R64, hd) for k in band_heads}
    acc = {k: 0 for k in band_heads}; nb = 0
    with torch.no_grad():
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); state["programs"] = {}; model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
            Tn = idx.shape[1]; causal = torch.tril(torch.ones(Tn, Tn, device=dev, dtype=torch.bool)); pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
            for k in band_heads:
                acc[k] = acc[k] + AP.offset_mean(AP.lowrank_pattern(blocks[k[0]].attn, state["n"][k[0]], maps64[k], hd, causal), dmat, off, Tn)
            nb += 1
    fix = dict(full)
    for k in band_heads:
        fix[k] = {"kind": "lowrank", "kappa": progs702[f"kappa_{k[0]}_{k[1]}"].to(dev).float(), "kappa_r": acc[k] / nb, "maps": maps64[k]}
    c_fix, r_fix, fw = values_under(fix, FIVE); forwards += fw
    report["fix_rank64_band"] = {"cost": c_fix - native, "heads": r_fix}
    print(f"full program with band 6-8 at closed-form rank 64: cost {c_fix - native:+.4f}; ratios " + " ".join(f"{k}:{r_fix[k]['ratio']:.2f}" for k in r_fix))
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    lh = report["leave_head"]
    arg83 = min(lh, key=lambda key: lh[key]["heads"]["8.3"]["ratio"]); arg55 = min(lh, key=lambda key: lh[key]["heads"]["5.5"]["ratio"])
    drop83 = r_alone["8.3"]["ratio"] - lh[arg83]["heads"]["8.3"]["ratio"]
    print(f"8.3: argmin head {arg83} (ratio {lh[arg83]['heads']['8.3']['ratio']:.2f}, drop {drop83:.2f}); 5.5: argmin head {arg55} (ratio {lh[arg55]['heads']['5.5']['ratio']:.2f})")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_single_head_inflates": drop83 >= DROP_MIN, "pred_c_inflating_head_is_layer8": arg83.startswith("8."),
                   "pred_d_55_shares_cause": arg55 == arg83, "pred_e_rank64_band_fixes": (c_fix - native) <= COST_MAX and r_fix["8.3"]["ratio"] <= RATIO_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    report["argmin_83"] = arg83; report["argmin_55"] = arg55; report["drop_83"] = drop83
    OUT.write_text(json.dumps({"schema": "attention_leaning_heads_result_v711", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
