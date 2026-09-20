#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_matches pred_b_sparse_strong_couplings pred_c_induction_reads_early_writers pred_d_83_strongest_writer_73 pred_e_coupling_decays_with_distance
# BQLANE: cpu
"""Attention lane, v727: the writer -> reader coupling graph implied by the program (CPU fold; no model).

Writer head w (layer l) writes into the rank-64 subspace S_w = colspace(W_o,w A_w B_w) (v716 write maps, 1152 x 64 after orthonormalising);
reader head r (layer l' > l) reads through its content directions V_r (the input-side factors of its four rank-r maps, v718; rows
normalised). Coupling c(w, r) = mean over V_r's rows of the energy inside S_w (a row fully inside S_w gives 1; a random direction gives
64 / 1152 = 0.056). The 162 x 162 upper-triangular table, its distribution, the top couplings, and the strongest writers of a few named
readers (5.5 induction, 8.3, 2.5). Caveat: the residual stream between writer and reader is re-normalised and mixed by MLPs, so this is the
DIRECT (weights-only) coupling, a candidate graph, not a causal one. FOLD only, 0 forwards.
PREDICTIONS (scored as written; failures preserved)
    pred_a_baseline_matches           the median coupling over all pairs is within [0.5, 2] x the 64/1152 baseline. Prior: likely
    pred_b_sparse_strong_couplings    >= 1% of pairs exceed 4 x baseline. Prior: likely
    pred_c_induction_reads_early_writers among 5.5's five strongest writers, >= 3 are in layers 0-2 (the key feature is written by the early positional heads). Prior: unsure
    pred_d_83_strongest_writer_73     7.3 is among 8.3's three strongest writers (v711: 7.3's program is what inflates 8.3). Prior: unsure
    pred_e_coupling_decays_with_distance median coupling at layer distance 1 exceeds the median at distance >= 6. Prior: unsure
PRICE (registered maximum): 0 forwards; 0 backwards; 0 fits; CPU only (162 QR factorisations of 1152 x 64; 13k pair projections).
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import torch
import torch.nn.functional as F
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/attention_coupling_graph_v727_result.json"
PROGS718 = ROOT / "circuits/followups/attention_exact_rank_v718_programs.pt"
SNAP716 = ROOT / "circuits/followups/attention_whole_program_v716_snapshot.pt"
MODEL_BIN = ROOT / ".hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
CANDIDATE_ID = "attention.coupling_graph_v727"
LAYERS = tuple(range(18)); H = 9; D = 1152; HD = 128; R_WRITE = 64
BASELINE = R_WRITE / D
BASE_RANGE, STRONG_MULT, STRONG_FRAC, N_EARLY, TOP_W = (0.5, 2.0), 4.0, 0.01, 3, 5
PREDICTIONS = {"pred_a_baseline_matches": "median in [0.5, 2] x 0.056", "pred_b_sparse_strong_couplings": ">= 1% of pairs >= 4x baseline", "pred_c_induction_reads_early_writers": ">= 3 of 5.5's top-5 writers in layers 0-2",
               "pred_d_83_strongest_writer_73": "7.3 in 8.3's top 3", "pred_e_coupling_decays_with_distance": "median(distance 1) > median(distance >= 6)"}


def load_c_proj():
    """c_proj weights per layer from the model checkpoint on CPU (no model construction)."""
    import glob
    paths = glob.glob(str(ROOT / ".hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/*/pytorch_model.bin"))
    sd = torch.load(paths[0], map_location="cpu")
    out = {}
    for l in LAYERS:
        key = next(k for k in sd if k.endswith(f"h.{l}.attn.c_proj.weight"))
        out[l] = sd[key].float()
    return out


def main() -> None:
    plan = {"candidate_id": CANDIDATE_ID, "forwards_max": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "lane": "cpu", "baseline": BASELINE,
            "bars": {"base_range": BASE_RANGE, "strong_mult": STRONG_MULT, "strong_frac": STRONG_FRAC, "n_early": N_EARLY, "top_w": TOP_W}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    torch.set_num_threads(4)
    fac = torch.load(PROGS718, map_location="cpu")["factored"]; snap = torch.load(SNAP716, map_location="cpu"); Wo = load_c_proj()
    ALL = [(l, h) for l in LAYERS for h in range(H)]
    Q = {}; V = {}
    for (l, h) in ALL:
        key = f"{l}.{h}"; A = snap[key]["A"].float(); B = snap[key]["B"].float()
        W = Wo[l][:, h * HD:(h + 1) * HD] @ (A @ B).T                      # [1152, 128] write map, rank 64
        Q[(l, h)] = torch.linalg.qr(W)[0][:, :R_WRITE]                     # orthonormal basis of the write subspace
        rows = torch.cat([F.normalize(fac[key][n_][1].float(), dim=1) for n_ in ("c_q", "c_k", "c_q2", "c_k2")])
        V[(l, h)] = rows
    coup = {}
    for w in ALL:
        for r in ALL:
            if r[0] > w[0]:
                coup[(w, r)] = float(((V[r] @ Q[w]) ** 2).sum(1).mean())
    vals = sorted(coup.values()); med = vals[len(vals) // 2]
    strong = [k for k, v in coup.items() if v >= STRONG_MULT * BASELINE]; frac_strong = len(strong) / len(coup)
    top = sorted(coup, key=coup.get, reverse=True)[:20]

    def writers_of(r, n=TOP_W):
        cands = [(w, coup[(w, r)]) for w in ALL if w[0] < r[0]]
        return sorted(cands, key=lambda t: t[1], reverse=True)[:n]

    named = {}
    for r in ((5, 5), (8, 3), (2, 5), (0, 3), (6, 3)):
        if r[0] > 0:
            named[f"{r[0]}.{r[1]}"] = [(f"{w[0]}.{w[1]}", round(v / BASELINE, 2)) for w, v in writers_of(r)]
    by_dist = {}
    for (w, r), v in coup.items():
        by_dist.setdefault(r[0] - w[0], []).append(v)
    med_dist = {d: sorted(vs)[len(vs) // 2] for d, vs in by_dist.items()}
    far = sorted([v for d, vs in by_dist.items() if d >= 6 for v in vs]); med_far = far[len(far) // 2]
    early55 = sum(1 for w, _ in writers_of((5, 5)) if w[0] <= 2)
    top3_83 = [w for w, _ in writers_of((8, 3), 3)]
    print(f"pairs {len(coup)} | baseline {BASELINE:.4f} | median {med:.4f} ({med / BASELINE:.2f}x) | >= 4x baseline: {len(strong)} ({100 * frac_strong:.2f}%) | max {vals[-1] / BASELINE:.1f}x")
    print("top couplings (x baseline): " + " ".join(f"{w[0]}.{w[1]}->{r[0]}.{r[1]}:{coup[(w, r)] / BASELINE:.1f}" for (w, r) in top))
    print("strongest writers of named readers: " + json.dumps(named))
    print("median by layer distance (x baseline): " + " ".join(f"d{d}:{med_dist[d] / BASELINE:.2f}" for d in sorted(med_dist)))
    predictions = {"pred_a_baseline_matches": BASE_RANGE[0] * BASELINE <= med <= BASE_RANGE[1] * BASELINE, "pred_b_sparse_strong_couplings": frac_strong >= STRONG_FRAC,
                   "pred_c_induction_reads_early_writers": early55 >= N_EARLY, "pred_d_83_strongest_writer_73": (7, 3) in top3_83, "pred_e_coupling_decays_with_distance": med_dist[1] > med_far}
    OUT.write_text(json.dumps({"schema": "attention_coupling_graph_result_v727", "candidate_id": CANDIDATE_ID, "plan": plan,
                               "report": {"baseline": BASELINE, "median": med, "frac_strong": frac_strong, "n_pairs": len(coup), "top": [(f"{w[0]}.{w[1]}", f"{r[0]}.{r[1]}", coup[(w, r)]) for (w, r) in top],
                                          "strong_pairs": [(f"{w[0]}.{w[1]}", f"{r[0]}.{r[1]}", coup[(w, r)]) for (w, r) in sorted(strong, key=coup.get, reverse=True)],
                                          "named_readers": named, "median_by_distance": {str(d): v for d, v in med_dist.items()},
                                          "coupling": {f"{w[0]}.{w[1]}->{r[0]}.{r[1]}": round(v, 5) for (w, r), v in coup.items()}},
                               "predictions": predictions, "forwards": 0, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": 0}, indent=2))


if __name__ == "__main__":
    main()
