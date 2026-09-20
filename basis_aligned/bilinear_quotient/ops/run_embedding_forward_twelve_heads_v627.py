#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_uniform_mean_cheap pred_c_unit_mass_cheap pred_d_twelve_heads_compose pred_e_twelve_under_bar
"""Embedding-forward folding, rung 19 (v627): head 1.8 as a running-mean subtractor, and the full twelve-head program for blocks 0-1.

v626: head 1.8's real pattern is a nearly uniform negative weight over the context with per-entry size ~ 1/(positions) and signed row sum
~ -0.9 at every query position; a kernel-only pattern costs +0.003. This rung installs the simplest possible form — pattern(i, j) := -m / i for
every j < i (a uniform running mean; m = the mean real row sum measured on 64 fit rows, and m = 1 exactly) — and then prices the whole block-0/1
attention program at once: five layer-0 gated filters (v623 tables) + six layer-1 gated filters (v624 tables) + head 1.8 as -m/i. Heads 0.0,
0.1, 0.2, 0.5 (content, layer 0) and 1.2, 1.4 (content, layer 1) stay native. CE on 192 x 512 skip7000; CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays      native CE within 0.002 of 3.13241 (instrument)
    pred_b_uniform_mean_cheap  1.8 := -m / i (measured m) costs <= 0.01. Prior: likely
    pred_c_unit_mass_cheap     1.8 := -1 / i costs <= 0.015. Prior: unsure
    pred_d_twelve_heads_compose the twelve-head program costs <= 1.5 x (eleven-head program re-measured here + the -m/i edit alone). Prior: likely
    pred_e_twelve_under_bar    the twelve-head program costs <= 0.05. Prior: likely (v625 eleven: 0.029)
PRICE (registered maximum): 2 capture forwards (64 rows) + 5 configs x 6 eval batches = 32 forwards; 0 backwards; 0 fits (m is one measured mean).
Bar <= 36.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_twelve_heads_v627_result.json"
T0 = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.twelve_heads_v627"
FORWARDS_MAX = 36
EBATCH, N_CAP, Q_MIN = 32, 64, 8
L0_SET, L1_SET, MEAN_HEAD = (3, 4, 6, 7, 8), (0, 1, 3, 5, 6, 7), 8
REPLAY_TOL, UNI_MAX, UNIT_MAX, COMP_RATIO, TWELVE_MAX = 0.002, 0.01, 0.015, 1.5, 0.05
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_uniform_mean_cheap": "<= 0.01", "pred_c_unit_mass_cheap": "<= 0.015",
               "pred_d_twelve_heads_compose": "<= 1.5 x (eleven + mean)", "pred_e_twelve_under_bar": "<= 0.05"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "l0_set": list(L0_SET), "l1_set": list(L1_SET), "mean_head": MEAN_HEAD,
            "bars": {"replay_tol": REPLAY_TOL, "uni_max": UNI_MAX, "unit_max": UNIT_MAX, "comp_ratio": COMP_RATIO, "twelve_max": TWELVE_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    t0_ = torch.load(T0, map_location=dev); t1_ = torch.load(T1, map_location=dev)
    tables = {0: {h: (t0_[f"head{h}_A"], t0_[f"head{h}_B"], t0_[f"head{h}_kappa"]) for h in L0_SET}, 1: {h: (t1_[f"head{h}_A"], t1_[f"head{h}_B"], t1_[f"head{h}_kappa"]) for h in L1_SET}}
    with torch.no_grad():
        state = {"idx": None, "heads": {0: (), 1: ()}, "mean_m": None, "capture": None}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        natives = {l: blocks[l].attn.squared_attention for l in (0, 1)}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = (causal & (dmat > 0))[None]
                if l == 1 and state["capture"] is not None:
                    state["capture"].append(pat[:, MEAN_HEAD].detach().clone())
                if state["heads"][l]:
                    idx = state["idx"]
                    for h in state["heads"][l]:
                        A, Bt, kappa = tables[l][h]
                        prog = kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
                        pat[:, h] = torch.where(off, prog, pat[:, h])
                if l == 1 and state["mean_m"] is not None:
                    prog = (-state["mean_m"] / pos.clamp_min(1).float())[None, :, None].expand(Bn, Tn, Tn)
                    pat[:, MEAN_HEAD] = torch.where(off, prog, pat[:, MEAN_HEAD])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in (0, 1):
            blocks[l].attn.squared_attention = make_patched(l)
        # ---- measure m: mean signed off-diagonal row sum of head 1.8 on 64 fit rows, queries >= 8 ---------------------------------
        state["capture"] = []
        for s in range(0, N_CAP, EBATCH):
            idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); forwards += 1
        reals = torch.cat(state["capture"]); state["capture"] = None
        Tn = reals.shape[-1]; pos = torch.arange(Tn, device=dev); off = (pos[:, None] > pos[None, :])
        row_sum = (reals * off[None]).sum(-1)[:, Q_MIN:]; m = float(-row_sum.mean())
        print(f"measured m (mean signed off-diagonal row sum of 1.8, negated) = {m:.4f}; by position quartile: {[round(float(-row_sum[:, a:b].mean()), 3) for a, b in ((0, 120), (120, 250), (250, 380), (380, 504))]}")

        def ce(h0, h1, mean_m):
            state["heads"] = {0: h0, 1: h1}; state["mean_m"] = mean_m; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        native, fw = ce((), (), None); forwards += fw
        e = {}
        e["mean_m"], fw = ce((), (), m); forwards += fw; e["mean_m"] -= native
        e["mean_1"], fw = ce((), (), 1.0); forwards += fw; e["mean_1"] -= native
        e["eleven"], fw = ce(L0_SET, L1_SET, None); forwards += fw; e["eleven"] -= native
        e["twelve"], fw = ce(L0_SET, L1_SET, m); forwards += fw; e["twelve"] -= native
        for l in (0, 1):
            blocks[l].attn.squared_attention = natives[l]
        pre.remove()
        print(f"native {native:.5f} | CE added: 1.8 := -m/i {e['mean_m']:+.4f}; -1/i {e['mean_1']:+.4f}; eleven {e['eleven']:+.4f}; twelve {e['twelve']:+.4f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_uniform_mean_cheap": e["mean_m"] <= UNI_MAX, "pred_c_unit_mass_cheap": e["mean_1"] <= UNIT_MAX,
                   "pred_d_twelve_heads_compose": e["twelve"] <= COMP_RATIO * (e["eleven"] + e["mean_m"]), "pred_e_twelve_under_bar": e["twelve"] <= TWELVE_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_twelve_heads_result_v627", "candidate_id": CANDIDATE_ID, "plan": plan, "report": {"native": native, "m": m, "edits": e},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
