#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_induction_indicator_55 pred_c_same_token_heads pred_d_induction_heads pred_e_prev_token_head_gains_nothing
"""Attention lane, v730: one-number content programs for the 38 valuable heads — same-token and induction indicators on top of the kernel.

v729: head 1.4's content is one number (kappa(d) + 0.05 [tok_i == tok_j] recovers 0.70 of its value, as much as rank 16). This rung asks
the same of every head with mean-ablation value >= 0.005 (38 heads, v701), with two indicator families: SAME [tok_i == tok_j] (duplicate
token) and INDUCTION [tok_i == tok_{j-1}] (the key's previous token equals the query token — the induction read), beta in {0.02, 0.05, 0.1},
plus kernel-only (beta 0). One head edited at a time inside the native model, closed-form v702 kernels. CE ADDED; recovery = 1 - cost /
value; gain = kernel-only cost - best indicator cost.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native within 0.002 of 3.13241 (instrument)
    pred_b_induction_indicator_55  head 5.5's best INDUCTION program recovers >= 0.5 of its value. Prior: likely
    pred_c_same_token_heads        >= 5 heads gain >= 0.001 from the SAME indicator. Prior: unsure
    pred_d_induction_heads         >= 3 heads gain >= 0.001 from the INDUCTION indicator. Prior: unsure
    pred_e_prev_token_head_gains_nothing head 0.3 gains <= 0.0005 from either indicator. Prior: likely
PRICE (registered maximum): native 6; 38 heads x 7 programs x 6 = 1596; total 1602 forwards; 0 backwards; 0 fits. Bar <= 1620.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_indicator_atlas_v730_result.json"
PROGS702 = ROOT / "circuits/followups/attention_program_ladder_v702_programs.pt"
V701 = ROOT / "circuits/followups/attention_mean_ablation_v701_result.json"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.indicator_atlas_v730"
FORWARDS_MAX = 1620
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
VALUE_FLOOR = 0.005
BETAS = (0.02, 0.05, 0.1)
REPLAY_TOL, REC55, GAIN, N_SAME, N_IND, GAIN03 = 0.002, 0.5, 0.001, 5, 3, 0.0005
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_induction_indicator_55": "5.5 induction recovery >= 0.5", "pred_c_same_token_heads": ">= 5 heads gain >= 0.001 (same)",
               "pred_d_induction_heads": ">= 3 heads gain >= 0.001 (induction)", "pred_e_prev_token_head_gains_nothing": "0.3 gain <= 0.0005"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "betas": list(BETAS), "value_floor": VALUE_FLOOR,
            "bars": {"replay_tol": REPLAY_TOL, "rec55": REC55, "gain": GAIN, "n_same": N_SAME, "n_ind": N_IND, "gain03": GAIN03}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    values = json.load(open(V701))["report"]["mean_ablation_cost"]; progs = torch.load(PROGS702, map_location=dev)
    heads = [(l, h) for l in LAYERS for h in range(H) if values[f"{l}.{h}"] >= VALUE_FLOOR]
    state = {"idx": None, "edit": None}
    pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
    natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

    def make_patched(l):
        def patched(q, k, v, q2, k2):
            Bn, Tn, Hn, Dn = q.shape
            pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
            causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat = pat.masked_fill(~causal, 0.0)
            e = state["edit"]
            if e is not None and e[0] == l:
                _, h, fam, beta, kappa = e
                pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]
                prog = kappa[dmat][None].expand(Bn, Tn, Tn)
                if fam == "same":
                    prog = prog + beta * (state["idx"][:, :, None] == state["idx"][:, None, :]).float()
                elif fam == "induction":
                    prev = torch.cat([torch.full_like(state["idx"][:, :1], -1), state["idx"][:, :-1]], 1)          # token before each key
                    prog = prog + beta * (state["idx"][:, :, None] == prev[:, None, :]).float()
                pat = pat.clone(); pat[:, h] = torch.where(off, prog, pat[:, h])
            return torch.einsum("bhqk,bkhd->bhqd", pat, v)
        return patched

    for l in LAYERS:
        blocks[l].attn.squared_attention = make_patched(l)

    def ce(edit):
        state["edit"] = edit; total = 0.0; n = 0; fw = 0
        with torch.no_grad():
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
        state["edit"] = None
        return total / n, fw

    native, fw = ce(None); forwards += fw
    report = {}
    for (l, h) in heads:
        key = f"{l}.{h}"; kappa = progs[f"kappa_{l}_{h}"].float(); val = values[key]
        c0, fw = ce((l, h, "kernel", 0.0, kappa)); forwards += fw; c0 -= native
        fams = {}
        for fam in ("same", "induction"):
            cs = {}
            for b in BETAS:
                c_, fw = ce((l, h, fam, b, kappa)); forwards += fw; cs[b] = c_ - native
            bb = min(cs, key=cs.get); fams[fam] = {"costs": {str(b): c for b, c in cs.items()}, "best_beta": bb, "best_cost": cs[bb], "gain": c0 - cs[bb], "recovery": 1 - cs[bb] / val}
        report[key] = {"value": val, "kernel_cost": c0, "kernel_recovery": 1 - c0 / val, **fams}
        print(f"{key}: value {val:.4f} | kernel {c0:+.4f} ({1 - c0 / val:.2f}) | same best beta {fams['same']['best_beta']} {fams['same']['best_cost']:+.4f} gain {fams['same']['gain']:+.4f} | induction best beta {fams['induction']['best_beta']} {fams['induction']['best_cost']:+.4f} gain {fams['induction']['gain']:+.4f}")
    for l in LAYERS:
        blocks[l].attn.squared_attention = natives[l]
    pre.remove()
    same_heads = [k for k in report if report[k]["same"]["gain"] >= GAIN]; ind_heads = [k for k in report if report[k]["induction"]["gain"] >= GAIN]
    print(f"native {native:.5f} | same-token gainers ({len(same_heads)}): {same_heads} | induction gainers ({len(ind_heads)}): {ind_heads} | 5.5 induction recovery {report['5.5']['induction']['recovery']:.2f} | 0.3 gains {report['0.3']['same']['gain']:+.4f} / {report['0.3']['induction']['gain']:+.4f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_induction_indicator_55": report["5.5"]["induction"]["recovery"] >= REC55,
                   "pred_c_same_token_heads": len(same_heads) >= N_SAME, "pred_d_induction_heads": len(ind_heads) >= N_IND,
                   "pred_e_prev_token_head_gains_nothing": max(report["0.3"]["same"]["gain"], report["0.3"]["induction"]["gain"]) <= GAIN03}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_indicator_atlas_result_v730", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "heads": report, "same_gainers": same_heads, "induction_gainers": ind_heads},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
