#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_capture_replays pred_b_energy_rank_small pred_c_energy_does_not_predict_ce pred_d_sink_energy_rank pred_e_wide_heads_have_more_energy
"""Attention lane, v722: is the write side 'wide' in energy or only in importance? The write covariance spectrum of every head against the CE
rank it needs (v710 single-head recoveries at rank 8 / 32).

v710-v716: a head's write needs ~64 of its 128 directions for CE (rank 32 recovers a median 0.85 singly). This rung folds the write
covariance G_h = W_o,h Cov(z_h - mean_h) W_o,h^T (64 fit rows, mean from v701) and reports its 90% / 99% energy ranks per head, the
energy captured by the top-8 and top-32 directions, and their relation to v710's CE recoveries: if energy rank is small while CE needs
more, the write is wide in LOW-energy, HIGH-importance directions — the token-metric lesson of v615 again, on the OV side. FOLD only.
PREDICTIONS (scored as written; failures preserved)
    pred_a_capture_replays             native CE on skip7000 within 0.002 of 3.13241 (instrument)
    pred_b_energy_rank_small           the median 90%-energy rank of G_h over the 162 heads <= 32. Prior: unsure
    pred_c_energy_does_not_predict_ce  over heads with value >= 0.005 (38), |Spearman(top-32 energy share, rank-32 CE recovery)| <= 0.4. Prior: unsure
    pred_d_sink_energy_rank            head 5.7's centered write has 90%-energy rank <= 8 (v710: its rank-8 CE recovery was only 0.58). Prior: likely
    pred_e_wide_heads_have_more_energy heads with rank-32 recovery < 0.7 have a larger median 90%-energy rank than heads with recovery >= 0.9. Prior: unsure
PRICE (registered maximum): native 6 + covariance capture 2 = 8 forwards; 0 backwards; 0 fits. Bar <= 10.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import circuit_fast_screen_producer as producer
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_write_energy_v722_result.json"
FIT_ROWS = ROOT / ".rowcache/fineweb_n480_skip80.pt"
EVAL_ROWS = ROOT / ".rowcache/fineweb_n192_skip7000.pt"
MEANS = ROOT / "circuits/followups/attention_mean_ablation_v701_means.pt"
V710 = ROOT / "circuits/followups/attention_write_rank_centered_v710_result.json"
NATIVE_V615 = 3.13241
CANDIDATE_ID = "attention.write_energy_v722"
FORWARDS_MAX = 10
EBATCH = 32
LAYERS = tuple(range(18)); H = 9
REPLAY_TOL, MED_RANK, RHO_MAX, SINK_RANK, VALUE_FLOOR = 0.002, 32, 0.4, 8, 0.005
PREDICTIONS = {"pred_a_capture_replays": "+-0.002", "pred_b_energy_rank_small": "median rank90 <= 32", "pred_c_energy_does_not_predict_ce": "|Spearman| <= 0.4",
               "pred_d_sink_energy_rank": "5.7 rank90 <= 8", "pred_e_wide_heads_have_more_energy": "median rank90 (rec32 < 0.7) > median rank90 (rec32 >= 0.9)"}


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "med_rank": MED_RANK, "rho_max": RHO_MAX, "sink_rank": SINK_RANK, "value_floor": VALUE_FLOOR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, model = backend.torch, backend.model
    D = model.config.n_embd; hd = D // H; dev = "cuda"; blocks = model.transformer.h; forwards = 0
    ev = torch.load(EVAL_ROWS, map_location="cpu").long(); fit64 = torch.load(FIT_ROWS, map_location="cpu").long()[:64]
    means = torch.load(MEANS, map_location=dev); v710 = json.load(open(V710))["report"]["heads"]
    ALL = [(l, h) for l in LAYERS for h in range(H)]
    with torch.no_grad():
        cov = {k: torch.zeros(hd, hd, dtype=torch.float64, device=dev) for k in ALL}; n_tok = [0]; state = {"capture": False}
        natives = {l: blocks[l].attn.squared_attention for l in LAYERS}

        def make_patched(l):
            def patched(q, k, v, q2, k2):
                Bn, Tn, Hn, Dn = q.shape
                pat = (torch.einsum("bqhd,bkhd->bhqk", q, k) / Dn) * (torch.einsum("bqhd,bkhd->bhqk", q2, k2) / Dn)
                causal = torch.tril(torch.ones(Tn, Tn, device=pat.device, dtype=torch.bool)); pat.masked_fill_(~causal, 0.0)
                z = torch.einsum("bhqk,bkhd->bhqd", pat, v)
                if state["capture"]:
                    for h in range(Hn):
                        zc = (z[:, h].float() - means[f"mean_{l}_{h}"]).reshape(-1, Dn).double(); cov[(l, h)] += zc.T @ zc
                    if l == 0:
                        n_tok[0] += Bn * Tn
                return z
            return patched

        for l in LAYERS:
            blocks[l].attn.squared_attention = make_patched(l)
        total = 0.0; n = 0
        for s in range(0, ev.shape[0], EBATCH):
            idx = ev[s:s + EBATCH, :-1].to(dev); total += float(model(idx, ev[s:s + EBATCH, 1:].to(dev))) * idx.numel(); n += idx.numel(); forwards += 1
        native = total / n
        state["capture"] = True
        for s in range(0, 64, EBATCH):
            idx = fit64[s:s + EBATCH, :-1].to(dev); model(idx, fit64[s:s + EBATCH, 1:].to(dev)); forwards += 1
        state["capture"] = False
        for l in LAYERS:
            blocks[l].attn.squared_attention = natives[l]
        heads = {}
        for (l, h) in ALL:
            Wo = blocks[l].attn.c_proj.weight[:, h * hd:(h + 1) * hd].detach().double(); G = Wo @ (cov[(l, h)] / n_tok[0]) @ Wo.T
            ev_ = torch.linalg.eigvalsh(G).flip(0).clamp_min(0); cum = ev_.cumsum(0) / ev_.sum()
            key = f"{l}.{h}"; info = v710[key]
            heads[key] = {"rank90": int((cum < 0.9).sum()) + 1, "rank99": int((cum < 0.99).sum()) + 1, "share_top8": float(cum[7]), "share_top32": float(cum[31]), "total_energy": float(ev_.sum()),
                          "value": info["value"], "rec8": info["recovery_8"], "rec32": info["recovery_32"]}
    ranks90 = sorted(v["rank90"] for v in heads.values()); med90 = ranks90[len(ranks90) // 2]
    big = [k for k in heads if heads[k]["value"] >= VALUE_FLOOR and heads[k]["rec32"] is not None]

    def spearman(x, y):
        rx = torch.tensor(x).argsort().argsort().double(); ry = torch.tensor(y).argsort().argsort().double(); rx -= rx.mean(); ry -= ry.mean(); return float((rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12))

    rho = spearman([heads[k]["share_top32"] for k in big], [heads[k]["rec32"] for k in big])
    wide = [heads[k]["rank90"] for k in big if heads[k]["rec32"] < 0.7]; narrow = [heads[k]["rank90"] for k in big if heads[k]["rec32"] >= 0.9]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    top = sorted(heads, key=lambda k: heads[k]["value"], reverse=True)[:16]
    print("head: rank90 / rank99 / top-32 share | rank-32 CE recovery: " + " ".join(f"{k}:{heads[k]['rank90']}/{heads[k]['rank99']}/{heads[k]['share_top32']:.2f}|{heads[k]['rec32']:.2f}" for k in top))
    print(f"native {native:.5f} | median rank90 {med90} (range {ranks90[0]}-{ranks90[-1]}) | 5.7 rank90 {heads['5.7']['rank90']} | Spearman(top-32 energy share, rec32) over {len(big)} heads {rho:.3f} | median rank90: wide (rec32 < 0.7, n={len(wide)}) {med(wide)} vs narrow (rec32 >= 0.9, n={len(narrow)}) {med(narrow)}")
    predictions = {"pred_a_capture_replays": abs(native - NATIVE_V615) <= REPLAY_TOL, "pred_b_energy_rank_small": med90 <= MED_RANK, "pred_c_energy_does_not_predict_ce": abs(rho) <= RHO_MAX,
                   "pred_d_sink_energy_rank": heads["5.7"]["rank90"] <= SINK_RANK, "pred_e_wide_heads_have_more_energy": bool(wide) and bool(narrow) and med(wide) > med(narrow)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "attention_write_energy_result_v722", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"native": native, "heads": heads, "median_rank90": med90, "spearman_energy_vs_rec32": rho, "median_rank90_wide": med(wide), "median_rank90_narrow": med(narrow), "n_wide": len(wide), "n_narrow": len(narrow)},
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
