"""Embedding-forward folding, rung 28 (v636): head 5.5 as the induction head (paired with 0.3), and what head 5.7 attends to.

v635 (with its base-rate instrument corrected: the true rate of induction-eligible keys is ~0.0098) found head 5.5 enriched 9.5x on keys whose
predecessor equals the current token, and head 5.7 to be the model's costliest head (+0.84 when its off-diagonal is zeroed) with no induction
structure. This rung: (i) corrected base rates; (ii) 5.5's pairing — its induction share with head 0.3's off-diagonal (the previous-token tap)
zeroed; (iii) 5.5's key-restricted edits (zero on induction-eligible keys vs on all other keys); (iv) 5.7's off-diagonal |mass| share on
position 0, on positions 0-7, and its per-entry weight for queries in [8, 64) vs [256, 512). Rows: shares on 64 skip80 rows (queries >= 8); CE
on 192 x 512 skip7000. CE ADDED, lower is better.
PREDICTIONS (scored as written; failures preserved)
    pred_a_native_replays          native CE within 0.002 of 3.13241 (instrument)
    pred_b_55_induction_enriched   5.5's induction share >= 5 x the (corrected) base rate. Prior: likely (9.5x in v635)
    pred_c_55_pairs_with_03        with 0.3's off-diagonal zeroed, 5.5's induction share falls by >= 30%. Prior: unsure
    pred_d_55_induction_keys_carry_cost zeroing 5.5 on induction-eligible keys costs >= 2 x zeroing it on the other keys. Prior: unsure
    pred_e_57_first_positions      5.7 puts >= 0.5 of its off-diagonal |mass| on positions 0-7 (a sink / prefix aggregator). Prior: unsure
PRICE (registered maximum): 4 capture forwards (native and 0.3-cut, 64 rows) + 3 configs x 6 eval batches = 22 forwards; 0 backwards; 0 fits. Bar <= 26.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_induction_v636_result.json"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.induction_v636"
FORWARDS_MAX = 26
EBATCH, N_CAP, Q_MIN = 32, 64, 8
L_IND, H_IND, L_PREV, H_PREV, H_SINK = 5, 5, 0, 3, 7
REPLAY_TOL, ENRICH_MIN, COST_RATIO, DROP, FIRST_MIN = 0.002, 5.0, 2.0, 0.7, 0.5
PREDICTIONS = {"pred_a_native_replays": "+-0.002", "pred_b_55_induction_enriched": ">= 5 x base", "pred_c_55_pairs_with_03": "share falls >= 30%",
               "pred_d_55_induction_keys_carry_cost": ">= 2 x other keys", "pred_e_57_first_positions": ">= 0.5 on positions 0-7"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "heads": {"induction": [L_IND, H_IND], "prev": [L_PREV, H_PREV], "sink": [L_IND, H_SINK]},
            "bars": {"replay_tol": REPLAY_TOL, "enrich_min": ENRICH_MIN, "cost_ratio": COST_RATIO, "drop": DROP, "first_min": FIRST_MIN}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    H = model.config.n_head; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    fit = torch.load(FIT_ROWS, map_location="cpu").long(); ev = torch.load(EVAL_ROWS, map_location="cpu").long()
    with torch.no_grad():
        state = {"idx": None, "capture": None, "cut_prev": False, "mode": None}
        pre = model.register_forward_pre_hook(lambda m, args: state.__setitem__("idx", args[0]))
        natives = {l: blocks[l].attn.squared_attention for l in (L_PREV, L_IND)}

        def key_masks(idx, Tn):
            """[B, T, T] boolean masks: induction-eligible keys (tok_{j-1} == tok_i, j >= 1), duplicate keys (tok_j == tok_i); causal off-diagonal only."""
            pos = torch.arange(Tn, device=idx.device); off = (pos[:, None] > pos[None, :])[None]
            prev_tok = torch.cat([torch.full_like(idx[:, :1], -1), idx[:, :-1]], 1)                                   # tok_{j-1}
            ind = (idx[:, :, None] == prev_tok[:, None, :]) & off
            dup = (idx[:, :, None] == idx[:, None, :]) & off
            return ind, dup, off

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                if l == L_PREV and state["cut_prev"]:
                    off = (causal & ~torch.eye(Tn, device=pat.device, dtype=torch.bool))[None]
                    pat[:, H_PREV] = torch.where(off, torch.zeros_like(pat[:, H_PREV]), pat[:, H_PREV])
                if l == L_IND:
                    if state["capture"] is not None:
                        state["capture"].append(pat.detach().clone())
                    if state["mode"]:
                        ind, dup, off = key_masks(state["idx"], Tn)
                        kill = ind if state["mode"] == "induction" else (off & ~ind)
                        pat[:, H_IND] = torch.where(kill, torch.zeros_like(pat[:, H_IND]), pat[:, H_IND])
                return torch.einsum("bhqk,bkhd->bhqd", pat, v)
            return patched

        for l in (L_PREV, L_IND):
            blocks[l].attn.squared_attention = make_patched(l)

        def shares(cut_prev):
            state["cut_prev"] = cut_prev; state["capture"] = []; idxs = []
            for s in range(0, N_CAP, EBATCH):
                idx = fit[s:s + EBATCH, :-1].to(dev); model(idx, fit[s:s + EBATCH, 1:].to(dev)); idxs.append(idx)
            pats = torch.cat(state["capture"]); idx_all = torch.cat(idxs); state["capture"] = None; state["cut_prev"] = False
            Tn = pats.shape[-1]; ind, dup, off = key_masks(idx_all, Tn); qm = (torch.arange(Tn, device=dev) >= Q_MIN)
            ind, dup, off = ind[:, qm], dup[:, qm], off[:, qm].expand(ind.shape[0], -1, -1)
            base_ind = float(ind.sum() / off.sum()); base_dup = float(dup.sum() / off.sum())
            pos = torch.arange(Tn, device=dev); first8 = (pos[None, :] < 8).expand(int(qm.sum()), -1)[None] & off
            out = {}
            for h in range(H):
                P = pats[:, h][:, qm]; m = P.abs() * off; tot = float(m.sum())
                per_entry = m.sum(-1) / off.sum(-1).clamp_min(1)
                out[h] = {"induction_share": float((m * ind).sum() / tot), "duplicate_share": float((m * dup).sum() / tot), "row_sum": float((P * off).sum(-1).mean()),
                          "share_pos0": float(m[:, :, 0].sum() / tot), "share_first8": float((m * first8).sum() / tot),
                          "per_entry_early": float(per_entry[:, :56].mean()), "per_entry_late": float(per_entry[:, 248:].mean())}
            return out, base_ind, base_dup, 2 * (N_CAP // EBATCH)

        nat, base_ind, base_dup, fw = shares(False); forwards += fw
        print(f"base rates: induction-eligible keys {base_ind:.4f}, duplicate keys {base_dup:.4f}")
        for h in range(H):
            print(f"head 5.{h}: induction share {nat[h]['induction_share']:.3f} ({nat[h]['induction_share'] / base_ind:.1f}x), duplicate share {nat[h]['duplicate_share']:.3f} ({nat[h]['duplicate_share'] / base_dup:.1f}x), row sum {nat[h]['row_sum']:+.2f}")
        cut, _, _, fw = shares(True); forwards += fw
        print(f"with head 0.3's off-diagonal zeroed: 5.{H_IND} induction share {cut[H_IND]['induction_share']:.3f} (native {nat[H_IND]['induction_share']:.3f})")
        sk = nat[H_SINK]; print(f"head 5.{H_SINK}: share on position 0 {sk['share_pos0']:.3f}, on positions 0-7 {sk['share_first8']:.3f}; per-entry |weight| queries 8-64 {sk['per_entry_early']:.4f} vs 256-512 {sk['per_entry_late']:.4f}")

        def ce(mode):
            state["mode"] = mode; total = 0.0; n = 0; fw = 0
            for s in range(0, ev.shape[0], EBATCH):
                idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); fw += 1
            state["mode"] = None
            return total / n, fw

        native, fw = ce(None); forwards += fw
        e_ind, fw = ce("induction"); forwards += fw; e_oth, fw = ce("other"); forwards += fw
        e = {"zero_induction_keys": e_ind - native, "zero_other_keys": e_oth - native}
        for l in (L_PREV, L_IND):
            blocks[l].attn.squared_attention = natives[l]
        pre.remove()
        print(f"native {native:.5f} | CE added: 5.{H_IND} zeroed on induction keys {e['zero_induction_keys']:+.4f}, on other keys {e['zero_other_keys']:+.4f}")
    enrich = {h: nat[h]["induction_share"] / base_ind for h in range(H)}
    predictions = {"pred_a_native_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_55_induction_enriched": enrich[H_IND] >= ENRICH_MIN,
                   "pred_c_55_pairs_with_03": cut[H_IND]["induction_share"] <= DROP * nat[H_IND]["induction_share"],
                   "pred_d_55_induction_keys_carry_cost": e["zero_induction_keys"] >= COST_RATIO * e["zero_other_keys"],
                   "pred_e_57_first_positions": nat[H_SINK]["share_first8"] >= FIRST_MIN}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_induction_result_v636", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "edits": e, "base_rates": {"induction": base_ind, "duplicate": base_dup}, "layer5_native": {str(h): v for h, v in nat.items()},
                                          "layer5_prev_cut": {str(h): v for h, v in cut.items()}, "enrichment": {str(h): v for h, v in enrich.items()}},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
