"""RUNG 442 (Claude red-team lane) -- MINIMAL-REALIZATION SPECTRUM CERTIFICATE FOR 424's RANK-6 SCORE QUOTIENT.

424/425 established the near-lossless 6/6/32 edge block; the ranks were
chosen, not certified.  Realization view (math review 22:10, move #2):
the exact per-branch score behavior matrices over FIT natural edges are
[n_edges, 9] tables whose singular spectrum bounds any rank-r linear
score quotient from below -- residual energy at rank r is a CERTIFIED
lower bound on marginal reconstruction error, independent of any fit.
This rung measures those spectra exactly and scores whether rank 6 is
adequate, non-wasteful, and minimal at its tolerance.

Construction (424's exact machinery): fit_edges via the unmodified
_collect_edges on the frozen FIT rows (edge count must equal 424's
185,760); per branch, centered score matrix SVD on the full edge set
and on two disjoint document halves (docs 0-47 / 48-95) for stability.
e_r = cumulative energy fraction at rank r.

Frozen predictions
------------------
pred_a (instrument): FIT row digest matches the receipt tensor_sha256;
    edge count == 185,760; per-half energy-fraction vectors agree with
    the full-set vector within .01 at every rank, both branches.
pred_b (rank-6 adequate and non-wasteful): full-set e_6 >= .95 AND the
    6th component alone carries >= .02 of total energy, both branches.
pred_c (minimality certificate): rank-5 residual energy (1-e_5) is
    >= 3x rank-6 residual energy (1-e_6), both branches -- dropping to
    rank 5 at least triples the certified error floor.

Null: e_6 < .90 on either branch (the chosen rank leaves >10% marginal
score energy -- the minimality framing is wrong and 424's ranks were
merely sufficient-for-fit), or the 6->7 gap < .005 of total on either
branch (rank 6 is arbitrary within the spectrum).

Price: certificate screen only; no shipped object; no 424/425 bar is
altered by any outcome.
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

ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
QK = ROOT / "basis_aligned/qk_mdl"
OPS = BQ / "ops"
OUT = BQ / "attention0_score_rank_certificate_results.json"
EDGE = OPS / "attention0_realized_edge_block_term.py"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
EDGE_COUNT_424 = 185_760
N_HEAD = 9


def _energy(matrix: torch.Tensor) -> list[float]:
    centered = matrix.double() - matrix.double().mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    squares = singular.square()
    return (torch.cumsum(squares, 0) / squares.sum().clamp_min(1e-30)).cpu().tolist()


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert EDGE_COUNT_424 == 185_760 and N_HEAD == 9
        assert EDGE.exists() and ROWS_RECEIPT.exists()
        print("ATTENTION0 SCORE RANK CERTIFICATE | dry run: exact spectra, halves, minimality bars")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(QK))
    spec = importlib.util.spec_from_file_location("edge_mod", EDGE)
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    from tier2_model import rope_tables, apply_rot

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    fit_hash = rows_parent.rows_life.base.tensor_sha256(fit_rows)
    model, _ = facade.load_bilin18(device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        edges = em._collect_edges(model, fit_rows, rope_tables, apply_rot)

    halves = {
        "half_a": edges["document"] < 48,
        "half_b": edges["document"] >= 48,
    }
    spectra = {}
    for branch in (1, 2):
        matrix = edges[f"score{branch}"]
        spectra[str(branch)] = {
            "full": _energy(matrix),
            **{name: _energy(matrix[mask]) for name, mask in halves.items()},
        }

    stability_max_dev = max(
        abs(spectra[b][half][r] - spectra[b]["full"][r])
        for b in ("1", "2") for half in ("half_a", "half_b")
        for r in range(N_HEAD))
    e5 = {b: spectra[b]["full"][4] for b in ("1", "2")}
    e6 = {b: spectra[b]["full"][5] for b in ("1", "2")}
    e7 = {b: spectra[b]["full"][6] for b in ("1", "2")}
    sixth = {b: spectra[b]["full"][5] - spectra[b]["full"][4] for b in ("1", "2")}
    gap67 = {b: e7[b] - e6[b] for b in ("1", "2")}
    residual_ratio = {
        b: (1 - e5[b]) / max(1 - e6[b], 1e-30) for b in ("1", "2")}

    pred_a = (
        fit_hash == receipt["entries"]["FIT"]["tensor_sha256"]
        and len(edges["source"]) == EDGE_COUNT_424
        and stability_max_dev <= .01)
    pred_b = all(e6[b] >= .95 and sixth[b] >= .02 for b in ("1", "2"))
    pred_c = all(residual_ratio[b] >= 3 for b in ("1", "2"))
    null = (
        any(e6[b] < .90 for b in ("1", "2"))
        or any(gap67[b] < .005 for b in ("1", "2")))

    result = {
        "status": "attention0_score_rank_certificate_complete",
        "rung": 442,
        "claim_level": "realization_rank_certificate_screen_not_compression",
        "edge_count": int(len(edges["source"])),
        "fit_hash": fit_hash,
        "spectra_energy_fractions": spectra,
        "stability_max_dev": stability_max_dev,
        "e5": e5, "e6": e6, "e7": e7,
        "sixth_component_share": sixth,
        "gap_6_to_7": gap67,
        "rank5_over_rank6_residual_ratio": residual_ratio,
        'pred_a_exact_edges_and_stable_spectrum': bool(pred_a),
        'pred_b_rank6_adequate_and_nonwasteful': bool(pred_b),
        'pred_c_rank6_minimal_at_tolerance': bool(pred_c),
        'null_rank6_diffuse_or_arbitrary': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": "certificate_statement_only",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
