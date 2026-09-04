"""circuit_battery_bilinear_eigen_causal -- does the ICLR'25 weight-space eigendecomposition rank CAUSAL directions at 546M?

Pearce, Dooms, Rigg, Oramas & Sharkey (arXiv:2410.08417, ICLR 2025 Spotlight) fold a gate-free bilinear MLP into a third-order
tensor, contract it with an output direction to get the symmetric form M_u = sym(Left^T diag(Down^T u) Right), and propose its top
eigenvectors as interpretable directions obtainable FROM WEIGHTS ALONE. bilin18 is exactly that architecture. Their evidence is
toy/vision/small-LM; nothing tests the ranking against causal ground truth at this scale, with RMSNorm and a residual stream.

We have reason to doubt it specifically: SS2822-SS2826 measured that energy/magnitude rankings do not track causal effect in this
model (an in-sample rank-4 subspace holds .700 of a removal effect's energy and delivers .139 of its damage; one unembedding-defined
direction holding .0021 of the energy delivers .199). |Eigenvalue| is that family. And the CPU half of this move
(ops/bilinear_eigen_cpu_probe.py) found the spectrum nearly FLAT -- effective rank ~737 of 1152, top-8 holding 2.4% of the absolute
eigenvalue mass, signs split ~50/50.

This rung ablates the block's input along eigendirections and asks whether |eigenvalue| predicts the damage.

# BQGATE: EXPERIMENT  pred_a_eigenvalue_predicts_damage pred_b_top_eigendirection_beats_random
#                     pred_c_the_output_axis_matters pred_d_the_spectrum_is_flat
#                     pred_e_full_basis_removal_is_the_block_input_ablation

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm in margin units, POSITIVE = the arm HURTS the successor answer. No CE and no SS312 L2;
nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_BILINEAR_EIGEN_CAUSAL_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_bilinear_eigen_causal.py")
sys.path.insert(0, str(SELF.parent)); sys.path.insert(0, "/workspace/tensor_language")
import mlp_in_situ_usage_rank_map_probe as R
import circuit_battery as CB
import circuit_battery_tasks as BANK
import bilinear_eigen_cpu_probe as EIG
import fastload
from receipt import dump

if not torch.cuda.is_available():
    raise RuntimeError("lane-1 CUDA script; refusing to fall back to CPU")
DEV = torch.device("cuda")
ROOT = R.ROOT
PREREG = R.POLY / "CIRCUIT_BATTERY_BILINEAR_EIGEN_CAUSAL_PREREGISTRATION.md"
CPUPROBE = ROOT / "bilinear_eigen_cpu_probe_results.json"
RUNG = "circuit_battery_bilinear_eigen_causal"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "6e2703f27a0c1b314dabf6b8772d82a49eeb4dc2c440555f97b78aec46140fbb",
          CPUPROBE: "a88ef6ddd7481ff2ebd26ded2731df2d16364a8a7ebcaa1b0954385b1d2139b5",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D = R.D
LAYERS = (8, 10)
TASK = "numbered_list.index_successor"
SPLIT = "OOD"
PER_CELL = 4 if SMOKE else 24
NPROBE = 24                     # eigendirections scored per block: top 12 by |eigenvalue| + 12 random
SEED = 2854
BARS = {"rho": 0.50, "top_over_random": 4.0, "axis_gain": 0.20, "flat_share": 0.10,
        "exact": 0.05, "floor": 0.5}
NULLS = {"rho_le": 0.10, "top_over_random_le": 1.5, "axis_gain_le": 0.0, "flat_share_ge": 0.40}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def run_dirs(m, tokens, finals, layer, V=None, kill_all=False):
    """Project the block's NORMALIZED input onto the complement of span(V) before the MLP.
    V: (D, k) orthonormal. kill_all=True zeroes the block's mlp output (the reference ablation)."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        x = x + write
        nrm = F.rms_norm(x, (D,))
        if site == layer and (V is not None or kill_all):
            if kill_all:
                out = torch.zeros_like(nrm)
            else:
                z = nrm.float()
                z = z - torch.einsum("btd,dk,ek->bte", z, V, V)
                out = blk.mlp(z.to(nrm.dtype))
        else:
            out = blk.mlp(nrm)
        x = x + out
    return (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()


def main():
    t0 = time.time()
    check_hashes()
    cpu = json.load(open(CPUPROBE))
    m = fastload.load_model_fast().to(DEV).eval()
    g = torch.Generator().manual_seed(SEED)
    rows = [r for r in BANK.build_rows(TASK, per_cell=PER_CELL)
            if r["family"] == "A1" and r["split"] == SPLIT]
    cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(TASK)}), device=DEV)
    ids_num = sorted({BANK.ENC.encode(s)[0] for s in [f" {i}" for i in range(0, 100)]
                      if len(BANK.ENC.encode(s)) == 1})
    WU = m.lm_head.weight.detach().float().cpu()
    u_num = WU[ids_num].mean(0); u_num = u_num / u_num.norm()
    u_rand = torch.randn(D, generator=g); u_rand = u_rand / u_rand.norm()
    fwd = [0]

    def damage(layer, V, kill_all=False):
        acc = []
        for b in CB.batches(rows):
            tok, fin, ans = CB.pack(b, "base")
            lg = run_dirs(m, tok, fin, layer); fwd[0] += 1
            mn = CB.margins(lg, ans, cand)
            lg2 = run_dirs(m, tok, fin, layer, V=V, kill_all=kill_all); fwd[0] += 1
            acc.append((mn - CB.margins(lg2, ans, cand)).cpu().numpy())
        return float(np.concatenate(acc).mean())

    results = {}
    for layer in LAYERS:
        blk = m.transformer.h[layer]
        rec = {}
        for label, u in (("numeric_axis", u_num), ("random_axis", u_rand)):
            M = EIG.symmetric_form(blk.cpu() if False else blk, u.to(DEV))
            ev, vec = torch.linalg.eigh(M)
            order = torch.argsort(ev.abs(), descending=True)
            ev, vec = ev[order], vec[:, order]
            share8 = float(ev.abs()[:8].sum() / ev.abs().sum())
            picks = list(range(min(12, NPROBE // 2)))
            rnd_idx = torch.randperm(vec.shape[1], generator=g)[:NPROBE // 2].tolist()
            per = {}
            for i in picks:
                V = vec[:, i:i + 1].contiguous()
                per[f"eig{i}"] = {"abs_eigenvalue": float(ev[i].abs()), "damage": damage(layer, V)}
            for j, i in enumerate(rnd_idx):
                V = vec[:, i:i + 1].contiguous()
                per[f"rand{j}"] = {"abs_eigenvalue": float(ev[i].abs()), "damage": damage(layer, V)}
            top = [per[f"eig{i}"] for i in picks]
            rnd = [per[f"rand{j}"] for j in range(len(rnd_idx))]
            xs = [v["abs_eigenvalue"] for v in per.values()]
            ys = [v["damage"] for v in per.values()]
            rk = lambda a: np.argsort(np.argsort(np.asarray(a, float))).astype(float)
            rho = float(np.corrcoef(rk(xs), rk(ys))[0, 1]) if np.std(rk(xs)) and np.std(rk(ys)) else float("nan")
            rec[label] = {"per_direction": per, "top8_share": share8, "spearman": rho,
                          "top1_damage": top[0]["damage"],
                          "median_random_damage": float(np.median([v["damage"] for v in rnd])),
                          "mean_top12_damage": float(np.mean([v["damage"] for v in top]))}
            print(f"[eig] mlp{layer} {label:12s} rho={rho:+.3f} top1={rec[label]['top1_damage']:+.4f} "
                  f"rand_med={rec[label]['median_random_damage']:+.4f} top8share={share8:.4f}", flush=True)
        rec["full_basis_damage"] = damage(layer, torch.linalg.eigh(
            EIG.symmetric_form(blk, u_num.to(DEV)))[1].contiguous())
        rec["block_ablation_damage"] = damage(layer, None, kill_all=True)
        results[f"mlp{layer}"] = rec

    med = lambda k, lab="numeric_axis": float(np.median([results[t][lab][k] for t in results]))
    ratio = float(np.median([abs(results[t]["numeric_axis"]["top1_damage"]) /
                             max(abs(results[t]["numeric_axis"]["median_random_damage"]), 1e-6)
                             for t in results]))
    axis_gain = float(np.median([results[t]["numeric_axis"]["mean_top12_damage"] -
                                 results[t]["random_axis"]["mean_top12_damage"] for t in results]))
    exact = max(abs(results[t]["full_basis_damage"] - results[t]["block_ablation_damage"]) /
                max(abs(results[t]["block_ablation_damage"]), BARS["floor"]) for t in results)
    preds = {
        'pred_a_eigenvalue_predicts_damage': bool(med("spearman") >= BARS["rho"]),
        'pred_b_top_eigendirection_beats_random': bool(ratio >= BARS["top_over_random"]),
        'pred_c_the_output_axis_matters': bool(axis_gain >= BARS["axis_gain"]),
        'pred_d_the_spectrum_is_flat': bool(med("top8_share") <= BARS["flat_share"]),
        'pred_e_full_basis_removal_is_the_block_input_ablation': bool(exact <= BARS["exact"]),
    }
    nulls = {
        "a_null_no_prediction": bool(med("spearman") <= NULLS["rho_le"]),
        "b_null_top_no_better": bool(ratio <= NULLS["top_over_random_le"]),
        "c_null_axis_irrelevant": bool(axis_gain <= NULLS["axis_gain_le"]),
        "d_null_spectrum_concentrated": bool(med("top8_share") >= NULLS["flat_share_ge"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "method_source": "arXiv:2410.08417 (Pearce et al., ICLR 2025)",
              "layers": list(LAYERS), "task": TASK, "split": SPLIT, "seed": SEED,
              "summary": {"median_spearman": med("spearman"), "top1_over_random": ratio,
                          "axis_gain": axis_gain, "median_top8_share": med("top8_share"),
                          "full_basis_vs_block_ablation": exact,
                          "cpu_probe_effective_ranks": {k: v["effective_rank_abs"]
                                                        for k, v in cpu["blocks"].items()}},
              "layers_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1300])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: eigendirections of the arXiv:2410.08417 symmetric form for mlp{LAYERS}, "
              f"scored causally on {TASK} {SPLIT}; no model loaded")
        sys.exit(0)
    main()
