"""Embedding-forward folding, rung 31 (v639): the last bisection — taps and windows each sufficient, both needed?

v637 / v638: head 5.5's induction key feature ("my predecessor was X") is written in layers 0-2 (all patterns cut -> 0.14x native share) but neither
the ten previous-token taps (0.86x) nor the eight content heads (1.02x) are needed alone. Remaining sets: WINDOWS = the nine window / running-mean
heads {0.4, 0.7, 1.0, 1.6, 1.8, 2.0, 2.3, 2.4, 2.7}; POSITIONAL19 = TAPS + WINDOWS; WINDOWS + CONTENT (everything but the taps). Response only:
5.5's induction share on 64 skip80 rows (queries >= 8).
PREDICTIONS (scored as written; failures preserved)
    pred_a_windows_alone_keep       under WINDOWS alone the share >= 0.6 x native (the taps suffice). Prior: unsure
    pred_b_positional19_kill        under POSITIONAL19 the share <= 0.3 x native (taps + windows are the writers). Prior: likely
    pred_c_taps_replay              the TAPS cut replays v637 within 0.01 (0.081) (instrument)
    pred_d_windows_plus_content_keep under WINDOWS + CONTENT the share >= 0.5 x native (the taps alone still carry it). Prior: unsure
    pred_e_monotone                 POSITIONAL19 share <= min(TAPS share, WINDOWS share) (cutting more never restores). Prior: likely
PRICE (registered maximum): 5 capture conditions x 2 forwards (64 rows) = 10 forwards; 0 backwards; 0 fits. Bar <= 12.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_prev_feature_sets_v639_result.json"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.prev_feature_sets_v639"
FORWARDS_MAX = 12
EBATCH, N_CAP, Q_MIN = 32, 64, 8
L_IND, H_IND = 5, 5
ALL = lambda ls: tuple((l, h) for l in ls for h in range(9))
TAPS = ((0, 3), (1, 1), (1, 3), (2, 6), (0, 6), (0, 8), (1, 5), (1, 7), (2, 2), (2, 8))
WINDOWS = ((0, 4), (0, 7), (1, 0), (1, 6), (1, 8), (2, 0), (2, 3), (2, 4), (2, 7))
CONTENT = ((0, 0), (0, 1), (0, 2), (0, 5), (1, 2), (1, 4), (2, 1), (2, 5))
CUTS = {"taps": TAPS, "windows": WINDOWS, "positional19": TAPS + WINDOWS, "windows_content": WINDOWS + CONTENT}
KEEP_A, KILL, REPLAY_TOL, KEEP_D, V637_TAPS = 0.6, 0.3, 0.01, 0.5, 0.081
PREDICTIONS = {"pred_a_windows_alone_keep": ">= 0.6 x native", "pred_b_positional19_kill": "<= 0.3 x native", "pred_c_taps_replay": "+-0.01 of 0.081",
               "pred_d_windows_plus_content_keep": ">= 0.5 x native", "pred_e_monotone": "positional19 <= min(taps, windows)"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "cuts": {k: [list(x) for x in v] for k, v in CUTS.items()},
            "bars": {"keep_a": KEEP_A, "kill": KILL, "replay_tol": REPLAY_TOL, "keep_d": KEEP_D}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long()
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
            return float((m * ind[:, qm]).sum() / m.sum()), float(ind[:, qm].sum() / off[:, qm].sum()), N_CAP // EBATCH

        shares = {}
        nat_share, base, fw = induction_share(()); forwards += fw; shares["native"] = nat_share
        for name, cut in CUTS.items():
            shares[name], _, fw = induction_share(cut); forwards += fw
        print(f"5.5 induction share (base {base:.4f}): native {nat_share:.3f} | " + " ".join(f"cut {k}: {v:.3f} ({v / nat_share:.2f}x)" for k, v in shares.items() if k != "native"))
        for l in layers:
            blocks[l].attn.squared_attention = natives[l]
        pre.remove()
    predictions = {"pred_a_windows_alone_keep": shares["windows"] >= KEEP_A * nat_share, "pred_b_positional19_kill": shares["positional19"] <= KILL * nat_share,
                   "pred_c_taps_replay": abs(shares["taps"] - V637_TAPS) <= REPLAY_TOL, "pred_d_windows_plus_content_keep": shares["windows_content"] >= KEEP_D * nat_share,
                   "pred_e_monotone": shares["positional19"] <= min(shares["taps"], shares["windows"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_prev_feature_sets_result_v639", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"induction_share": shares, "base_rate": base},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
