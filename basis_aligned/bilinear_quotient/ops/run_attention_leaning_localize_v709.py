#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_single_band_inflates pred_c_inflating_band_is_early pred_d_leave_one_band_consistent pred_e_five_heads_share_cause
"""Attention lane, v709: WHICH approximated heads make the model lean on 8.3 / 5.5 / 1.4 / 3.5 / 1.1? Leave-one-band / only-one-band, no fit.

v707: under the all-hybrid pattern program the mean-ablation value of 8.3 is 4.4x its native value (5.5 2.5x, 1.4 2.0x, 3.5 1.5x, 1.1 1.3x)
and restoring those heads to native does not change it — the approximated CONTEXT leans on them. This rung localizes the context: the
frozen program (v706 endpoint maps, the five heads native) applied ONLY to one layer band, and applied to all bands EXCEPT one, for the six
bands 0-2, 3-5, 6-8, 9-11, 12-14, 15-17. Under each of the 12 configurations: cost, and the in-program mean-ablation value of the five heads.
Response only (edits are the fixed v706 programs). CE ADDED, lower is better; ratio = in-program value / native value (v701).
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_single_band_inflates    some band applied ALONE gives 8.3 a ratio >= 2.5 (more than half the 4.4x excess sits in one band). Prior: unsure
    pred_c_inflating_band_is_early the band that inflates 8.3 most when applied alone is 0-2 or 3-5 (upstream feature writers). Prior: likely
    pred_d_leave_one_band_consistent leaving that same band native (all others programmed) lowers 8.3's ratio by >= 1.0 from the full-program 4.4. Prior: unsure
    pred_e_five_heads_share_cause  that band is also the alone-band with the largest ratio for at least 3 of the other four heads. Prior: unsure
PRICE (registered maximum): native 6; kappa_r 2; full program base 6 + 5 x 6 = 36; 12 configurations x (6 + 30) = 432; total 482 forwards;
0 backwards; 0 fits. Bar <= 500.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery
import attention_program_lib as AP

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_leaning_localize_v709_result.json"
PROGS706 = ROOT / "circuits/followups/attention_mixed16_manip_v706_programs.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.leaning_localize_v709"
FORWARDS_MAX = 500
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
FIVE = ((8, 3), (5, 5), (1, 4), (3, 5), (1, 1))
BANDS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13, 14), (15, 16, 17))
REPLAY_TOL, ALONE_MIN, DROP_MIN, SHARE_MIN = 0.002, 2.5, 1.0, 3
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_single_band_inflates": "max alone ratio(8.3) >= 2.5", "pred_c_inflating_band_is_early": "argmax band in {0-2, 3-5}",
               "pred_d_leave_one_band_consistent": "full ratio - leave-band ratio >= 1.0", "pred_e_five_heads_share_cause": ">= 3 of 4 other heads share the argmax band"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bands": [list(b) for b in BANDS], "heads": [list(k) for k in FIVE],
            "bars": {"replay_tol": REPLAY_TOL, "alone_min": ALONE_MIN, "drop_min": DROP_MIN, "share_min": SHARE_MIN}}
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

    def values_under(progs):
        base, fw = ce(progs); out = {}
        for k in FIVE:
            state["ablate"] = {k}; c_, f_ = ce(progs); fw += f_; state["ablate"] = set()
            key = f"{k[0]}.{k[1]}"; out[key] = {"in_program_value": c_ - base, "ratio": (c_ - base) / v701[key]}
        return base, out, fw

    native, fw = ce({}); forwards += fw
    c_full, r_full, fw = values_under(full); forwards += fw
    report = {"native": native, "full": {"cost": c_full - native, "heads": r_full}, "alone": {}, "leave": {}}
    print(f"native {native:.5f} | full program cost {c_full - native:+.4f}; ratios " + " ".join(f"{k}:{r_full[k]['ratio']:.2f}" for k in r_full))
    for band in BANDS:
        name = f"{band[0]}-{band[-1]}"
        alone = {k: p for k, p in full.items() if k[0] in band}; leave = {k: p for k, p in full.items() if k[0] not in band}
        c_a, r_a, fw = values_under(alone); forwards += fw; c_l, r_l, fw = values_under(leave); forwards += fw
        report["alone"][name] = {"cost": c_a - native, "heads": r_a}; report["leave"][name] = {"cost": c_l - native, "heads": r_l}
        print(f"band {name}: ALONE cost {c_a - native:+.4f} ratios " + " ".join(f"{k}:{r_a[k]['ratio']:.2f}" for k in r_a) + f" | LEAVE native cost {c_l - native:+.4f} ratios " + " ".join(f"{k}:{r_l[k]['ratio']:.2f}" for k in r_l))
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    for hh in nhooks:
        hh.remove()
    pre.remove()
    names = [f"{b[0]}-{b[-1]}" for b in BANDS]
    argmax = {key: max(names, key=lambda nm: report["alone"][nm]["heads"][key]["ratio"]) for key in r_full}
    top83 = argmax["8.3"]; alone83 = report["alone"][top83]["heads"]["8.3"]["ratio"]; leave83 = report["leave"][top83]["heads"]["8.3"]["ratio"]
    shared = sum(1 for key in r_full if key != "8.3" and argmax[key] == top83)
    print(f"8.3: alone-band argmax {top83} (ratio {alone83:.2f}); leave-{top83}-native ratio {leave83:.2f} vs full {r_full['8.3']['ratio']:.2f}; argmax bands {argmax}; shared with 8.3: {shared}/4")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_single_band_inflates": alone83 >= ALONE_MIN, "pred_c_inflating_band_is_early": top83 in ("0-2", "3-5"),
                   "pred_d_leave_one_band_consistent": (r_full["8.3"]["ratio"] - leave83) >= DROP_MIN, "pred_e_five_heads_share_cause": shared >= SHARE_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    report["argmax_band"] = argmax; report["shared_with_83"] = shared
    OUT.write_text(json.dumps({"schema": "attention_leaning_localize_result_v709", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
