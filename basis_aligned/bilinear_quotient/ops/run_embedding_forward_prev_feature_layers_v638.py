"""Embedding-forward folding, rung 30 (v638): where is the induction head's previous-token key feature written? Layer-wise cuts.

v637: zeroing the off-diagonal patterns of all ten previous-token taps in blocks 0-2 leaves head 5.5's induction share at 0.86x native. This rung
localises the feature by cutting whole sets of patterns (values, projections and diagonals native) and measuring 5.5's induction share on 64
skip80 rows (queries >= 8): ALL04 = every head of layers 0-4 (the anchor: position j's residual then depends on tok_j alone, so the share must
fall to the base rate ~0.0098 — if it does not, the statistic is suspect); L34 = every head of layers 3-4; L012 = every head of layers 0-2;
CONTENT012 = the eight content heads of layers 0-2 (0.0 0.1 0.2 0.5 1.2 1.4 2.1 2.5). Response only; no CE.
PREDICTIONS (scored as written; failures preserved)
    pred_a_anchor_vanishes   under ALL04 the induction share <= 2 x the base rate (instrument anchor). Prior: must hold
    pred_b_layers34_carry    under L34 the share <= 0.5 x native. Prior: unsure
    pred_c_layers012_carry   under L012 the share <= 0.5 x native. Prior: unsure
    pred_d_content_heads_do_not under CONTENT012 the share >= 0.8 x native. Prior: likely
    pred_e_shares_consistent L34 and L012 shares are each >= the ALL04 share (cutting more never restores the feature). Prior: likely
PRICE (registered maximum): 5 capture conditions x 2 forwards (64 rows) = 10 forwards; 0 backwards; 0 fits. Bar <= 12.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/embedding_forward_prev_feature_layers_v638_result.json"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "embedding_forward.prev_feature_layers_v638"
FORWARDS_MAX = 12
EBATCH, N_CAP, Q_MIN = 32, 64, 8
L_IND, H_IND = 5, 5
ALL = lambda ls: tuple((l, h) for l in ls for h in range(9))
CUTS = {"all04": ALL((0, 1, 2, 3, 4)), "l34": ALL((3, 4)), "l012": ALL((0, 1, 2)), "content012": ((0, 0), (0, 1), (0, 2), (0, 5), (1, 2), (1, 4), (2, 1), (2, 5))}
ANCHOR_MULT, HALF, KEEP = 2.0, 0.5, 0.8
PREDICTIONS = {"pred_a_anchor_vanishes": "<= 2 x base", "pred_b_layers34_carry": "<= 0.5 x native", "pred_c_layers012_carry": "<= 0.5 x native",
               "pred_d_content_heads_do_not": ">= 0.8 x native", "pred_e_shares_consistent": "l34, l012 >= all04"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "cuts": {k: [list(x) for x in v] for k, v in CUTS.items()},
            "bars": {"anchor_mult": ANCHOR_MULT, "half": HALF, "keep": KEEP}}
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
    predictions = {"pred_a_anchor_vanishes": shares["all04"] <= ANCHOR_MULT * base, "pred_b_layers34_carry": shares["l34"] <= HALF * nat_share,
                   "pred_c_layers012_carry": shares["l012"] <= HALF * nat_share, "pred_d_content_heads_do_not": shares["content012"] >= KEEP * nat_share,
                   "pred_e_shares_consistent": shares["l34"] >= shares["all04"] and shares["l012"] >= shares["all04"]}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "embedding_forward_prev_feature_layers_result_v638", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"induction_share": shares, "base_rate": base},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
