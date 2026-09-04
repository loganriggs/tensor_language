"""circuit_battery_bilinear_eigen_moment -- give arXiv:2410.08417 its best shot: eigenvalue TIMES input second moment.

SS2854 tested the ICLR'25 weight-only ranking as specified and found |eigenvalue| ANTI-correlated with causal damage (median
Spearman -.446) on a flat spectrum. Before letting that stand as a verdict on the method, there is an algebraic reason it was
guaranteed to fail in that form, and a cheap repair.

For the symmetric form M_u = sym(Left^T diag(Down^T u) Right), the block's output along u is f_u(z) = z^T M_u z on the NORMALIZED
input z. Removing an eigendirection v (eigenvalue lam) from z changes f_u by EXACTLY -lam * <z,v>^2. So the causally relevant
quantity is not lam but **lam * <z,v>^2** -- eigenvalue weighted by how much the actual input projects onto that eigenvector. A
direction with a huge eigenvalue that the data never occupies does nothing, and ranking by lam alone cannot know that.

That repair costs one second-moment statistic of the block's normalized input -- far less than a causal sweep, and still mostly
weight-space. This rung tests whether it rescues the method, and checks the algebra exactly.

# BQGATE: EXPERIMENT  pred_a_moment_weighted_predicts_damage pred_b_moment_weighting_beats_raw_eigenvalue
#                     pred_c_the_algebraic_identity_is_exact pred_d_moment_top_beats_eigen_top
#                     pred_e_raw_correlation_replicates_2854

SIGN CONVENTION: damage d_m = m_NATIVE - m_arm in margin units, POSITIVE = the arm HURTS the successor answer. No CE and no SS312
L2; nothing installs.
Preregistration: polynomial_causal/CIRCUIT_BATTERY_BILINEAR_EIGEN_MOMENT_PREREGISTRATION.md
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SELF = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient/ops/circuit_battery_bilinear_eigen_moment.py")
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
PREREG = R.POLY / "CIRCUIT_BATTERY_BILINEAR_EIGEN_MOMENT_PREREGISTRATION.md"
EIGRES = ROOT / "circuit_battery_bilinear_eigen_causal_results.json"
RUNG = "circuit_battery_bilinear_eigen_moment"
SMOKE = os.environ.get("SURROGATE_SMOKE") == "1"
OUT = ROOT / (f"{RUNG}_smoke_results.json" if SMOKE else f"{RUNG}_results.json")
HASHES = {PREREG: "d629a57e873ba8def6b859cea818a115ba4ff86a955b7926afcee43c48f5820e",
          EIGRES: "82c09594e34b5eeaf3f677563e1aca24635ec2d30adc71b18bd2913520efccb5",
          R.BLOB: "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"}
D = R.D
LAYERS = (8, 10)
TASK = "numbered_list.index_successor"
SPLIT = "OOD"
PER_CELL = 4 if SMOKE else 24
NDIR = 24
SEED = 2855
BARS = {"rho_moment": 0.60, "gain_over_raw": 0.50, "exact_rel": 1e-4, "top_ratio": 2.0, "repro": 0.15,
        "floor": 0.5}
NULLS = {"rho_moment_le": 0.10, "gain_le": 0.0, "top_ratio_le": 1.0}


def check_hashes():
    for path, expect in HASHES.items():
        if not Path(path).is_file() or R.sha256(path) != expect:
            raise RuntimeError(f"frozen hash mismatch: {path}")


@torch.no_grad()
def collect_and_run(m, tokens, finals, layer, v=None, want_z=False, want_fu=None):
    """One forward. v: remove this direction from the block's normalized input.
    want_z: also return the block's normalized input at the final position.
    want_fu: an output direction u -- also return f_u of the block's output at the final position."""
    x = F.rms_norm(m.transformer.wte(tokens), (D,))
    x0 = x; v1 = None
    ar = torch.arange(tokens.size(0), device=tokens.device)
    z_out, fu_out = None, None
    for site, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        write, v1 = blk.attn(F.rms_norm(x, (D,)), v1)
        x = x + write
        nrm = F.rms_norm(x, (D,))
        if site == layer:
            z = nrm.float()
            if want_z:
                z_out = z[ar, finals].clone()
            if v is not None:
                z = z - (z @ v).unsqueeze(-1) * v
            out = blk.mlp(z.to(nrm.dtype))
            if want_fu is not None:
                fu_out = (out[ar, finals].float() @ want_fu)
        else:
            out = blk.mlp(nrm)
        x = x + out
    logits = (30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0))[ar, finals].float()
    return logits, z_out, fu_out


def spearman(a, b):
    rk = lambda v: np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    ra, rb = rk(a), rk(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) and np.std(rb) else float("nan")


def main():
    t0 = time.time()
    check_hashes()
    prev = json.load(open(EIGRES))
    m = fastload.load_model_fast().to(DEV).eval()
    g = torch.Generator().manual_seed(SEED)
    rows = [r for r in BANK.build_rows(TASK, per_cell=PER_CELL)
            if r["family"] == "A1" and r["split"] == SPLIT]
    cand = torch.tensor(sorted({BANK.ENC.encode(s)[0] for s in BANK.candidate_strings(TASK)}), device=DEV)
    ids_num = sorted({BANK.ENC.encode(s)[0] for s in [f" {i}" for i in range(0, 100)]
                      if len(BANK.ENC.encode(s)) == 1})
    WU = m.lm_head.weight.detach().float()
    u = WU[ids_num].mean(0); u = (u / u.norm()).contiguous()
    fwd = [0]
    results = {}
    for layer in LAYERS:
        blk = m.transformer.h[layer]
        M = EIG.symmetric_form(blk, u)
        ev, vec = torch.linalg.eigh(M)
        order = torch.argsort(ev.abs(), descending=True)
        ev, vec = ev[order], vec[:, order]
        # ---- collect the block's normalized input second moment along every eigendirection ----
        zs = []
        for b in CB.batches(rows):
            tok, fin, _ = CB.pack(b, "base")
            _lg, z, _f = collect_and_run(m, tok, fin, layer, want_z=True); fwd[0] += 1
            zs.append(z)
        Z = torch.cat(zs, 0)                                   # (N, D)
        proj2 = ((Z @ vec) ** 2).mean(0)                        # E[<z,v>^2] per eigendirection
        score_raw = ev.abs()
        score_mom = ev.abs() * proj2
        # ---- pick directions: top-12 by moment score + top-12 by raw eigenvalue ----
        top_mom = torch.argsort(score_mom, descending=True)[:NDIR // 2].tolist()
        top_raw = list(range(NDIR // 2))
        picks = sorted(set(top_mom) | set(top_raw))
        per = {}
        for i in picks:
            v = vec[:, i].contiguous()
            acc, algebra = [], []
            for b in CB.batches(rows):
                tok, fin, ans = CB.pack(b, "base")
                lg0, z0, f0 = collect_and_run(m, tok, fin, layer, want_z=True, want_fu=u); fwd[0] += 1
                lg1, _z, f1 = collect_and_run(m, tok, fin, layer, v=v, want_fu=u); fwd[0] += 1
                acc.append((CB.margins(lg0, ans, cand) - CB.margins(lg1, ans, cand)).cpu().numpy())
                pred = float(ev[i]) * ((z0 @ v) ** 2)           # exact predicted change in f_u
                algebra.append(((f0 - f1) - pred).abs().div(pred.abs().clamp_min(1e-6)).cpu().numpy())
            # DIAGNOSTIC (not a registered predicate): the same identity evaluated in float64 directly
            # from the captured z and the weights, with no differencing of model outputs. pred_c is scored
            # on the fp32 model-output difference as registered; this line exists to say whether a failure
            # there is cancellation or a real algebra error.
            with torch.no_grad():
                M64 = M.double(); v64 = v.double()
                zc = Z[:8].double()
                lhs = torch.einsum("nd,de,ne->n", zc, M64, zc)
                zp = zc - (zc @ v64).unsqueeze(-1) * v64
                rhs = torch.einsum("nd,de,ne->n", zp, M64, zp)
                pred64 = float(ev[i].double()) * ((zc @ v64) ** 2)
                err64 = float(((lhs - rhs) - pred64).abs().div(pred64.abs().clamp_min(1e-30)).max())
            per[int(i)] = {"eigenvalue": float(ev[i]), "abs_eigenvalue": float(ev[i].abs()),
                           "algebra_rel_err_float64_diagnostic": err64,
                           "mean_sq_projection": float(proj2[i]),
                           "moment_score": float(score_mom[i]),
                           "damage": float(np.concatenate(acc).mean()),
                           "algebra_rel_err": float(np.concatenate(algebra).max())}
        xs_raw = [per[i]["abs_eigenvalue"] for i in picks]
        xs_mom = [per[i]["moment_score"] for i in picks]
        ys = [per[i]["damage"] for i in picks]
        rho_raw, rho_mom = spearman(xs_raw, ys), spearman(xs_mom, ys)
        dm_top_mom = float(np.mean([per[i]["damage"] for i in top_mom]))
        dm_top_raw = float(np.mean([per[i]["damage"] for i in top_raw]))
        results[f"mlp{layer}"] = {
            "per_direction": per, "rho_raw_eigenvalue": rho_raw, "rho_moment_weighted": rho_mom,
            "mean_damage_top_moment": dm_top_mom, "mean_damage_top_raw": dm_top_raw,
            "max_algebra_rel_err": max(per[i]["algebra_rel_err"] for i in picks),
            "max_algebra_rel_err_float64_diagnostic": max(
                per[i]["algebra_rel_err_float64_diagnostic"] for i in picks),
            "prev_rho_raw_2854": prev["layers_detail"][f"mlp{layer}"]["numeric_axis"]["spearman"]}
        p = results[f"mlp{layer}"]
        print(f"[mom] mlp{layer} rho_raw={rho_raw:+.3f} rho_moment={rho_mom:+.3f} "
              f"dmg_top_mom={dm_top_mom:+.4f} dmg_top_raw={dm_top_raw:+.4f} "
              f"algebra_err={p['max_algebra_rel_err']:.2e} (SS2854 rho_raw={p['prev_rho_raw_2854']:+.3f})",
              flush=True)

    med = lambda k: float(np.median([results[t][k] for t in results]))
    gain = med("rho_moment_weighted") - med("rho_raw_eigenvalue")
    ratio = float(np.median([abs(results[t]["mean_damage_top_moment"]) /
                             max(abs(results[t]["mean_damage_top_raw"]), 1e-6) for t in results]))
    repro = max(abs(results[t]["rho_raw_eigenvalue"] - results[t]["prev_rho_raw_2854"]) for t in results)
    preds = {
        'pred_a_moment_weighted_predicts_damage': bool(med("rho_moment_weighted") >= BARS["rho_moment"]),
        'pred_b_moment_weighting_beats_raw_eigenvalue': bool(gain >= BARS["gain_over_raw"]),
        'pred_c_the_algebraic_identity_is_exact': bool(med("max_algebra_rel_err") <= BARS["exact_rel"]),
        'pred_d_moment_top_beats_eigen_top': bool(ratio >= BARS["top_ratio"]),
        'pred_e_raw_correlation_replicates_2854': bool(repro <= BARS["repro"]),
    }
    nulls = {
        "a_null_moment_no_better_than_chance": bool(med("rho_moment_weighted") <= NULLS["rho_moment_le"]),
        "b_null_no_gain": bool(gain <= NULLS["gain_le"]),
        "d_null_no_top_gain": bool(ratio <= NULLS["top_ratio_le"]),
    }
    result = {"rung": RUNG, "preds": preds, "nulls": nulls, "bars": BARS, "null_bars": NULLS,
              "method_source": "arXiv:2410.08417 with an input-second-moment correction",
              "layers": list(LAYERS), "task": TASK, "split": SPLIT, "seed": SEED,
              "summary": {"median_rho_raw": med("rho_raw_eigenvalue"),
                          "median_rho_moment": med("rho_moment_weighted"),
                          "gain": gain, "top_damage_ratio": ratio,
                          "median_algebra_rel_err": med("max_algebra_rel_err"),
                          "median_algebra_rel_err_float64_diagnostic":
                              med("max_algebra_rel_err_float64_diagnostic"),
                          "max_repro_gap_vs_2854": repro},
              "layers_detail": results, "smoke": SMOKE,
              "price": {"gpu_forwards": fwd[0], "backwards": 0, "fitted_parameters": 0,
                        "gpu_seconds": time.time() - t0},
              "hashes": {str(k): v for k, v in HASHES.items()}}
    dump(result, OUT)
    print(json.dumps({"preds": preds, "nulls": nulls, "summary": result["summary"],
                      "price": result["price"]}, indent=1)[:1200])


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(f"[dryrun] {RUNG}: eigenvalue x input-second-moment ranking for mlp{LAYERS}, scored causally on "
              f"{TASK} {SPLIT}, with an exact algebraic identity check; no model loaded")
        sys.exit(0)
    main()
