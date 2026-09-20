#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_replays pred_b_four_taps_cut_halves_induction pred_c_ten_taps_cut_kills_induction pred_d_four_taps_cost pred_e_ten_taps_cost
"""Embedding-forward folding, rung 29 (v637): which previous-token taps feed the induction head 5.5? Joint cuts.

v636: head 5.5 is the induction head (9.6x enrichment on keys whose predecessor is the current token), yet zeroing head 0.3's off-diagonal — the
sharpest previous-token tap — leaves its induction share unchanged (0.100 vs 0.093). The atlas (v622 / v624 / v628) lists several taps that write
"the previous token" into the residual: sharp taps 0.3, 1.1, 1.3, 2.6 and short windows 0.6, 0.8, 1.5, 1.7, 2.2, 2.8. Joint cuts (off-diagonal
pattern zeroed, diagonal and values native): {0.3}; FOUR = the four sharp taps; TEN = FOUR + the six short windows. For each: 5.5's induction share
on 64 skip80 rows (response, queries >= 8) and, for FOUR and TEN, the CE on 192 x 512 skip7000 (edit). CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays               native CE within 0.002 of 3.13241 (instrument)
    pred_b_four_taps_cut_halves_induction under FOUR, 5.5's induction share <= 0.5 x native. Prior: unsure
    pred_c_ten_taps_cut_kills_induction under TEN, 5.5's induction share <= 0.3 x native. Prior: likely if (b)
    pred_d_four_taps_cost               FOUR costs >= 0.10 nats (the previous-token feature is load-bearing far beyond induction). Prior: likely
    pred_e_ten_taps_cost                TEN costs >= 2 x FOUR. Prior: unsure
PRICE (registered maximum): 4 cut conditions x 2 capture forwards (64 rows) = 8 + 3 configs x 6 eval batches = 18; total 26 forwards; 0 backwards;
0 fits. Bar <= 30.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_prev_taps_v637_result.json"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.prev_taps_v637"
FORWARDS_MAX = 30
EBATCH, N_CAP, Q_MIN = 32, 64, 8
L_IND, H_IND = 5, 5
CUTS = {"03": ((0, 3),), "four": ((0, 3), (1, 1), (1, 3), (2, 6)), "ten": ((0, 3), (1, 1), (1, 3), (2, 6), (0, 6), (0, 8), (1, 5), (1, 7), (2, 2), (2, 8))}
REPLAY_TOL, HALF, KILL, FOUR_MIN, TEN_RATIO = 0.002, 0.5, 0.3, 0.10, 2.0
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_four_taps_cut_halves_induction": "<= 0.5 x native", "pred_c_ten_taps_cut_kills_induction": "<= 0.3 x native",
               "pred_d_four_taps_cost": ">= 0.10", "pred_e_ten_taps_cost": ">= 2 x FOUR"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "cuts": {k: [list(x) for x in v] for k, v in CUTS.items()},
            "bars": {"replay_tol": REPLAY_TOL, "half": HALF, "kill": KILL, "four_min": FOUR_MIN, "ten_ratio": TEN_RATIO}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    layers = sorted({l for v in CUTS.values() for (l, h) in v} | {L_IND})
    with torch.no_grad():
        state = {"idx": None, "cut": (), "capture": None}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        natives = {l: blocks[l].attn.squared_attention for l in layers}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                off = (causal & ~torch.eye(Tn, device=pat.device, dtype=torch.bool))[None]
                for (cl, ch) in state["cut"]:
                    if cl == l:
                        pat[:, ch] = torch.where(off, torch.zeros_like(pat[:, ch]), pat[:, ch])
                if l == L_IND and state["capture"] is not None:
                    state["capture"].append(pat[:, H_IND].detach().clone())
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in layers:
            blocks[l].attn.squared_attention = make_patched(l)

        def induction_share(cut):
            state["cut"] = cut; state["capture"] = []; idxs = []
            for s in range(0, N_CAP, EBATCH):
                idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); idxs.append(idx)
            P = torch.cat(state["capture"]); idx = torch.cat(idxs); state["capture"] = None; state["cut"] = ()
            Tn = P.shape[-1]; pos = torch.arange(Tn, device=dev); off = (pos[:, None] > pos[None, :])[None].expand(P.shape[0], -1, -1)
            prev_tok = torch.cat([torch.full_like(idx[:, :1], -1), idx[:, :-1]], 1); ind = (idx[:, :, None] == prev_tok[:, None, :]) & off
            qm = pos >= Q_MIN; m = P[:, qm].abs() * off[:, qm]
            return float((m * ind[:, qm]).sum() / m.sum()), float(ind[:, qm].sum() / off[:, qm].sum()), 2 * (N_CAP // EBATCH)

        shares = {}
        nat_share, base, fw = induction_share(()); forwards += fw; shares["native"] = nat_share
        for name, cut in CUTS.items():
            shares[name], _, fw = induction_share(cut); forwards += fw
        print(f"5.5 induction share (base {base:.4f}): native {nat_share:.3f} | " + " ".join(f"cut {k}: {v:.3f} ({v / nat_share:.2f}x)" for k, v in shares.items() if k != "native"))

        def ce(cut):
            state["cut"] = cut; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            state["cut"] = ()
            return total / n, fw

        native, fw = ce(()); forwards += fw; e = {}
        for name in ("four", "ten"):
            v_, fw = ce(CUTS[name]); forwards += fw; e[name] = v_ - native
        for l in layers:
            blocks[l].attn.squared_attention = natives[l]
        pre.remove()
        print(f"native {native:.5f} | CE added: cut four {e['four']:+.4f}, cut ten {e['ten']:+.4f}")
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_four_taps_cut_halves_induction": shares["four"] <= HALF * nat_share,
                   "pred_c_ten_taps_cut_kills_induction": shares["ten"] <= KILL * nat_share, "pred_d_four_taps_cost": e["four"] >= FOUR_MIN, "pred_e_ten_taps_cost": e["ten"] >= TEN_RATIO * e["four"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_prev_taps_result_v637", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": e, "induction_share": shares, "base_rate": base},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
