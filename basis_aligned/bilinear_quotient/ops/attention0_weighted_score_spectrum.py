"""RUNG 446 (Claude red-team lane) -- TASK-WEIGHTED SCORE SPECTRUM: DOES THE METRIC EXPLAIN RANK 6?

442 (ledger 2560) proved the RAW per-branch score spectra are near-
uniformly diffuse (e6 = .812/.790), so 424's near-lossless rank-6 block
cannot be explained by raw score low-rankness.  The stated explanation
-- "the compression lives in the response-metric/product geometry" --
is itself a falsifiable claim, and this rung registers it.

Certified construction: for branch-1 scores, the metric edge error from
discarding components x = (I-P)s1[n] obeys
    error_n = x^T V[n] x,   V[n] = B[n] B[n]^T,
    B[n] = diag(s2[n]) @ (metric_factor @ payload[source_n]),   [9,16]
so error <= tr((I-P) A (I-P)) with the WEIGHTED covariance
    A = sum_n w_n (s1[n]-mu_w)(s1[n]-mu_w)^T,  w_n = sigma_max(B[n])^2
-- an upper-bound certificate whose spectrum says how much score energy
MATTERS under the task metric (symmetric for branch 2 with roles
swapped).  Weights and payloads use 419's exact task interface and
424's response-metric factor, rebuilt deterministically from FIT.

Frozen predictions
------------------
pred_a (instrument): FIT digest matches; edge count == 185,760; the
    unweighted spectra reproduce 442's stored energy fractions within
    1e-6 at every rank, both branches.
pred_b (metric concentrates): weighted e6 >= raw e6 + .08 AND weighted
    e6 >= .90, both branches.
pred_c (knee sharpens): weighted (e7-e6) gap-to-residual ratio -- the
    6th-vs-7th structure -- satisfies weighted_gap67 >= 1.5x raw gap67
    on both branches.

Null: weighted e6 < raw e6 + .03 on either branch -- task weighting
does NOT explain rank-6's success; the explanation must be the joint
product/optimization structure, and 2560's statement needs revision.

Price: certificate screen only; no shipped object; no 424/425/442 bar
is altered by any outcome.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_weighted_score_spectrum_results.json"
EDGE = OPS / "attention0_realized_edge_block_term.py"
REF_442 = BQ / "attention0_score_rank_certificate_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
EDGE_COUNT_424 = 185_760
N_HEAD = 9
D = 1152
U_RANK = 16
DOC_BATCH = 4
POSITIONS = tuple(range(16, 241, 16))


def _energy(matrix: torch.Tensor, weights: torch.Tensor | None = None) -> list[float]:
    m = matrix.double()
    if weights is None:
        centered = m - m.mean(0, keepdim=True)
        cov = centered.T @ centered
    else:
        w = weights.double().clamp_min(0)
        mu = (w[:, None] * m).sum(0) / w.sum().clamp_min(1e-30)
        centered = m - mu
        cov = (centered * w[:, None]).T @ centered
    eig = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).flip(0).clamp_min(0)
    return (torch.cumsum(eig, 0) / eig.sum().clamp_min(1e-30)).cpu().tolist()


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EDGE_COUNT_424 == 185_760 and U_RANK == 16
        assert EDGE.exists() and REF_442.exists() and ROWS_RECEIPT.exists()
        print("ATTENTION0 WEIGHTED SCORE SPECTRUM | dry run: metric-weighted covariance certificate")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(QK))
    spec = importlib.util.spec_from_file_location("edge_mod", EDGE)
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    spec2 = importlib.util.spec_from_file_location("ov_base", em.BASE)
    base = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(base)
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    from tier2_model import rope_tables, apply_rot

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    fit_hash = rows_parent.rows_life.base.tensor_sha256(fit_rows)
    model, _ = facade.load_bilin18(device=device, dtype=torch.float32)
    model.eval()
    block0 = model.transformer.h[0]

    with torch.no_grad():
        captured = base._capture_cproj_input(model, fit_rows, device).to(device)
        weight = block0.attn.c_proj.weight.detach().float()
        a_factor, _b = base._asvd(weight, captured)
        interface = torch.linalg.qr(
            a_factor[:, :U_RANK].float(), mode="reduced").Q.to(device)
        embedding = F.rms_norm(
            model.transformer.wte.weight.detach().float(), (D,))[:base.VOCAB]
        all_payload = base._payload_codes(model, interface, embedding)
        write_samples = []
        for start in range(0, len(fit_rows), DOC_BATCH):
            tokens = fit_rows[start:start + DOC_BATCH, :-1].to(device)
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
            attention0, _v = block0.attn(F.rms_norm(token_base, (D,)), None)
            write_samples.append(attention0.float()[:, POSITIONS].reshape(-1, D))
        write_samples = torch.cat(write_samples)
        sigma = torch.sqrt(
            (write_samples @ interface).double().square().mean(0)).float()
        fit_gram, _n, _l = base._response_metric(
            model, fit_rows, interface, sigma, None, device)
        metric_factor, _e = base._metric_factor(fit_gram)
        edges = em._collect_edges(model, fit_rows, rope_tables, apply_rot)

        payload_m = torch.einsum(
            "vhc,dc->vhd", all_payload, metric_factor)  # [V,9,16] metric-factored
        raw = {}
        weighted = {}
        for this, other in ((1, 2), (2, 1)):
            s_this = edges[f"score{this}"]
            s_other = edges[f"score{other}"]
            raw[str(this)] = _energy(s_this)
            weights_all = []
            for start in range(0, len(s_this), 32_768):
                sl = slice(start, start + 32_768)
                B = s_other[sl].unsqueeze(-1) * payload_m[edges["source"][sl]]
                weights_all.append(
                    torch.linalg.matrix_norm(B, ord=2).square())
            weights = torch.cat(weights_all)
            weighted[str(this)] = _energy(s_this, weights)

    ref = json.loads(REF_442.read_text())
    repro_max = max(
        abs(raw[b][r] - ref["spectra_energy_fractions"][b]["full"][r])
        for b in ("1", "2") for r in range(N_HEAD))
    raw_e6 = {b: raw[b][5] for b in ("1", "2")}
    w_e6 = {b: weighted[b][5] for b in ("1", "2")}
    raw_gap = {b: raw[b][6] - raw[b][5] for b in ("1", "2")}
    w_gap = {b: weighted[b][6] - weighted[b][5] for b in ("1", "2")}

    pred_a = (
        fit_hash == receipt["entries"]["FIT"]["tensor_sha256"]
        and len(edges["source"]) == EDGE_COUNT_424
        and repro_max <= 1e-6)
    pred_b = all(
        w_e6[b] >= raw_e6[b] + .08 and w_e6[b] >= .90 for b in ("1", "2"))
    pred_c = all(w_gap[b] >= 1.5 * raw_gap[b] for b in ("1", "2"))
    null = any(w_e6[b] < raw_e6[b] + .03 for b in ("1", "2"))

    result = {
        "status": "attention0_weighted_score_spectrum_complete",
        "rung": 446,
        "claim_level": "metric_weighted_certificate_screen_not_compression",
        "edge_count": int(len(edges["source"])),
        "raw_spectra": raw,
        "weighted_spectra": weighted,
        "raw_reproduction_max_abs_vs_442": repro_max,
        "raw_e6": raw_e6, "weighted_e6": w_e6,
        "raw_gap67": raw_gap, "weighted_gap67": w_gap,
        'pred_a_exact_edges_and_442_reproduction': bool(pred_a),
        'pred_b_metric_weighting_concentrates_rank6': bool(pred_b),
        'pred_c_knee_sharpens_under_metric': bool(pred_c),
        'null_metric_does_not_explain_rank6': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": "certificate_statement_only",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
