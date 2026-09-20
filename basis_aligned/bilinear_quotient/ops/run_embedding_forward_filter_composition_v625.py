#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_six_set_cheap pred_c_composes_across_layers pred_d_head18_is_sink pred_e_program_not_sink
"""Embedding-forward folding, rung 17 (v625): do the gated-filter programs COMPOSE across layers, and what is head 1.8 really doing?

v623 installed five layer-0 heads' patterns as kappa(d) A(t) B(s) at +0.0094; v624 six layer-1 heads at ~+0.009 summed — and found head 1.8
uninstallable (+0.558) despite a near-perfect single-token fit. Two questions, reusing the saved tables (no new table forwards):
 (i) COMPOSITION: v619 showed MLP projections compound across layers. Here: the clean six-head layer-1 set, the five-head layer-0 set, and all
     eleven together, on skip7000. If eleven <= 1.5 x (five + six), pattern programs compose where projections did not.
 (ii) HEAD 1.8: capture its REAL pattern on 16 held-out rows and compare with the token program's prediction on the same rows: share of the
     off-diagonal pattern mass (|pattern|) on position 0 for query positions >= 8. A sink head attends to position 0 by position, which no
     token x offset table can express (position 0 is only "offset i", and the program spends that kappa on every offset).
CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays        native CE within 0.002 of 3.13241 (instrument)
    pred_b_six_set_cheap         the six-head layer-1 set (1.0, 1.1, 1.3, 1.5, 1.6, 1.7) costs <= 0.02. Prior: likely (singles sum to 0.009)
    pred_c_composes_across_layers all eleven heads <= 1.5 x (five-set + six-set), each re-measured here. Prior: unsure — v619's lesson
    pred_d_head18_is_sink        head 1.8's real pattern puts >= 0.5 of its off-diagonal |mass| on position 0 (query positions >= 8, 16 rows). Prior: unsure
    pred_e_program_not_sink      the token program for 1.8 puts <= 0.2 of its |mass| on position 0 on the same rows. Prior: likely
PRICE (registered maximum): 4 configs x 6 batches = 24 forwards + 1 capture forward (16 rows) = 25; 0 backwards; 0 fits. Bar <= 30.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_filter_composition_v625_result.json"
T0 = ROOT / "circuits/followups/embedding_forward_gated_filter_edit_v623_tables.pt"
T1 = ROOT / "circuits/followups/embedding_forward_layer1_filters_v624_tables.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.filter_composition_v625"
FORWARDS_MAX = 30
EBATCH = 32
L0_SET, L1_SET, SINK = (3, 4, 6, 7, 8), (0, 1, 3, 5, 6, 7), 8
N_DIAG, Q_MIN = 16, 8
REPLAY_TOL, SIX_MAX, COMP_RATIO, SINK_MIN, PROG_MAX = 0.002, 0.02, 1.5, 0.5, 0.2
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_six_set_cheap": "<= 0.02", "pred_c_composes_across_layers": "<= 1.5 x (five + six)",
               "pred_d_head18_is_sink": ">= 0.5 mass on position 0", "pred_e_program_not_sink": "<= 0.2 mass on position 0"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "l0_set": list(L0_SET), "l1_set": list(L1_SET), "sink": SINK,
            "bars": {"replay_tol": REPLAY_TOL, "six_max": SIX_MAX, "comp_ratio": COMP_RATIO, "sink_min": SINK_MIN, "prog_max": PROG_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    t0_ = torch.load(T0, map_location=dev); t1_ = torch.load(T1, map_location=dev)
    tables = {0: {h: (t0_[f"head{h}_A"], t0_[f"head{h}_B"], t0_[f"head{h}_kappa"]) for h in L0_SET},
              1: {h: (t1_[f"head{h}_A"], t1_[f"head{h}_B"], t1_[f"head{h}_kappa"]) for h in L1_SET + (SINK,)}}
    with torch.no_grad():
        state = {"idx": None, "heads": {0: (), 1: ()}, "capture": None}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        natives = {l: blocks[l].attn.squared_attention for l in (0, 1)}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                if state["capture"] is not None and l == 1:
                    state["capture"]["real"] = pat[:, SINK].detach().clone()
                if state["heads"][l]:
                    idx = state["idx"]; pos = torch.arange(Tn, device=pat.device); dmat = (pos[:, None] - pos[None, :]).clamp_min(0); off = causal & (dmat > 0)
                    for h in state["heads"][l]:
                        A, Bt, kappa = tables[l][h]
                        prog = kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
                        pat[:, h] = torch.where(off[None], prog, pat[:, h])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in (0, 1):
            blocks[l].attn.squared_attention = make_patched(l)

        def ce(h0, h1):
            state["heads"] = {0: h0, 1: h1}; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            return total / n, fw

        native, fw = ce((), ()); forwards += fw
        e = {}
        e["five_l0"], fw = ce(L0_SET, ()); forwards += fw; e["five_l0"] -= native
        e["six_l1"], fw = ce((), L1_SET); forwards += fw; e["six_l1"] -= native
        e["eleven"], fw = ce(L0_SET, L1_SET); forwards += fw; e["eleven"] -= native
        print(f"native {native:.5f} | CE added: five_l0 {e['five_l0']:+.4f} six_l1 {e['six_l1']:+.4f} eleven {e['eleven']:+.4f} (sum {e['five_l0'] + e['six_l1']:+.4f})")

        # ---- head 1.8 diagnosis on 16 rows: real pattern vs token program, mass on position 0 -----------------------------------
        state["heads"] = {0: (), 1: ()}; state["capture"] = {}
        idx = ev[:N_DIAG, :-1].to(dev); model(idx, ev[:N_DIAG, 1:].to(dev)); forwards += 1
        real = state["capture"]["real"]; state["capture"] = None                                   # [B, T, T]
        Tn = real.shape[-1]; pos = torch.arange(Tn, device=dev); dmat = (pos[:, None] - pos[None, :]).clamp_min(0)
        A, Bt, kappa = tables[1][SINK]; prog = kappa[dmat][None] * A[idx][:, :, None] * Bt[idx][:, None, :]
        off = (dmat > 0)[None].expand_as(real)
        real_off = real.abs() * off; prog_off = prog.abs() * off
        qmask = pos >= Q_MIN
        real_share0 = float(real_off[:, qmask, 0].sum() / real_off[:, qmask].sum()); prog_share0 = float(prog_off[:, qmask, 0].sum() / prog_off[:, qmask].sum())
        real_prev = float(real_off[:, qmask][:, :, :].gather(2, (pos[qmask] - 1)[None, :, None].expand(N_DIAG, -1, 1)).sum() / real_off[:, qmask].sum())
        argmax_col = real_off[:, qmask].argmax(-1); frac_argmax0 = float((argmax_col == 0).float().mean())
        sign0 = float(real[:, qmask, 0].mean()); mean_off = float(real_off[:, qmask].sum(-1).mean())
        print(f"head 1.8 (query pos >= {Q_MIN}): real |mass| share on position 0 = {real_share0:.3f} (previous token {real_prev:.3f}; argmax at 0 in {frac_argmax0:.2f} of queries; mean signed pattern at 0 {sign0:+.4f}); token program share on position 0 = {prog_share0:.3f}")
        for l in (0, 1):
            blocks[l].attn.squared_attention = natives[l]
        pre.remove()
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL,
                   "pred_b_six_set_cheap": e["six_l1"] <= SIX_MAX,
                   "pred_c_composes_across_layers": e["eleven"] <= COMP_RATIO * (e["five_l0"] + e["six_l1"]),
                   "pred_d_head18_is_sink": real_share0 >= SINK_MIN,
                   "pred_e_program_not_sink": prog_share0 <= PROG_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_filter_composition_result_v625", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": e, "head18": {"real_share_pos0": real_share0, "real_share_prev": real_prev, "frac_argmax_pos0": frac_argmax0,
                                                                                    "mean_signed_at_pos0": sign0, "mean_offdiag_abs_mass": mean_off, "program_share_pos0": prog_share0}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
