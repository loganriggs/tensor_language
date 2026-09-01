"""RUNG 432 (Claude red-team lane) -- INTERACTION-RANK TAXONOMY: MODULATED-POSITIONAL vs CONJUNCTIVE.

The head census (ledger 2548/2549) left one mechanism distinction
unmeasured: head 3 and head 6 BOTH carry large interaction shares
(.564 / .464) in the pair x offset decomposition, yet the registered
picture says they are different animals -- head 3 a positional head
whose offset profile is MODULATED by token content (predicting a
LOW-RANK interaction: one dominant token-weighting of one offset
pattern), head 6 a CONJUNCTIVE head whose selectivity is genuinely
pair-by-offset (predicting a HIGH-RANK interaction spread across the
7-dim interaction space).

Construction (428/429's exact machinery, seeds 432_777 SELECT /
432_888 FIT, both new): scores s_b[pair, offset, head] on 32,768 pairs
x offsets (1..128); per head/branch remove both main effects and take
the SVD of the interaction residual [pairs, 8]; top1_share =
s1^2 / sum s_i^2 (branch-mean per head).  Controls: per-offset
independent pair permutation of the raw matrix (structureless
interaction; wide sanity window after two window lessons), and the
second population as split/transfer stability.

Frozen predictions
------------------
pred_a (instrument): float64 fold gate <= 1e-10; cross-population
    Spearman of head top1-share vectors >= .8; every shuffle-control
    top1 share inside the wide sanity window (.05, .30).
pred_b (head 3 low-rank modulation): head 3 interaction top1_share
    >= .60 on BOTH populations AND >= 1.5x head 6's on both.
pred_c (head 6 high-rank conjunction): head 6 top1_share strictly
    below the 9-head median on both populations.

Null: head3/head6 top1 ratio < 1.2 on both populations, or head 3
top1 < .40 anywhere -- the modulated-vs-conjunctive taxonomy is wrong
and the census keeps only the main-effect facts of 2548/2549.

Price: identification screen only; no shipped object; no prior bar is
altered by any outcome.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

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
OUT = BQ / "attention0_interaction_rank_taxonomy_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
VOCAB = 50_257
N_HEAD = 9
HD = 128
OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128)
PAIR_COUNT = 32_768
SEEDS = (432_777, 432_888)
HEAD_POS = 3
HEAD_CONJ = 6


def _interaction_top1(matrix: torch.Tensor) -> tuple[float, list[float]]:
    centered = matrix.double() - matrix.double().mean()
    interaction = centered - centered.mean(0, keepdim=True) - centered.mean(1, keepdim=True)
    singular = torch.linalg.svdvals(interaction)
    squares = singular.square()
    total = float(squares.sum())
    return float(squares[0]) / max(total, 1e-30), (squares / max(total, 1e-30)).cpu().tolist()


def _spearman(left, right):
    def ranks(values):
        order = torch.argsort(torch.tensor(values))
        rank = torch.empty(len(values), dtype=torch.float64)
        rank[order] = torch.arange(len(values), dtype=torch.float64)
        return rank
    a, b = ranks(left), ranks(right)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert HEAD_POS == 3 and HEAD_CONJ == 6 and PAIR_COUNT == 32_768
        assert not {426_777, 428_777, 428_888, 429_777, 429_888} & set(SEEDS)
        assert ROWS_RECEIPT.exists()
        print("ATTENTION0 INTERACTION RANK TAXONOMY | dry run: SVD of pairxoffset interaction, two populations")
        return

    started = time.time()
    sys.path.insert(0, str(QK))
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    from tier2_model import load_elriggs, reference_forward, rope_tables, apply_rot
    from tier2_folding import branch_factors, scores_from_factors
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])

    exact_model, _cfg = load_elriggs("bilin18", device=device, dtype=torch.float64)
    exact_factors = {branch: branch_factors(exact_model, branch, dtype=torch.float64)
                     for branch in (1, 2)}
    captured = {}

    def capture(layer, score1, score2):
        if layer == 0:
            captured[1] = score1.detach()
            captured[2] = score2.detach()
        return score1, score2

    gate_tokens = select_rows[:1, :-1].to(device)
    reference_forward(exact_model, gate_tokens, "bf16", capture)
    fold_errors = {}
    for branch in (1, 2):
        folded = scores_from_factors(
            *exact_factors[branch], gate_tokens, HD, table_dtype="bf16")
        fold_errors[str(branch)] = float((folded - captured[branch]).abs().max())
    del exact_model, exact_factors, captured, folded
    torch.cuda.empty_cache()

    model, _cfg2 = load_elriggs("bilin18", device=device, dtype=torch.float32)
    factors = {branch: branch_factors(model, branch, dtype=torch.float32)
               for branch in (1, 2)}
    tables = {
        "q1": factors[1][0][:VOCAB], "k1": factors[1][1][:VOCAB],
        "q2": factors[2][0][:VOCAB], "k2": factors[2][1][:VOCAB],
    }
    token_ids = torch.arange(VOCAB, device=device)
    populations = {
        SEEDS[0]: token_ids[token_ids.remainder(5) == 4],
        SEEDS[1]: token_ids[token_ids.remainder(5) != 4],
    }
    cos, sin = rope_tables(max(OFFSETS) + 1, HD, device, torch.float32, "bf16")

    results = {}
    shuffle_generator = torch.Generator(device="cpu").manual_seed(432_999)
    for seed in SEEDS:
        pool = populations[seed]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        query = pool[torch.randint(
            len(pool), (PAIR_COUNT,), generator=generator).to(device)]
        key = pool[torch.randint(
            len(pool), (PAIR_COUNT,), generator=generator).to(device)]
        per_branch = []
        for branch in (1, 2):
            q = tables[f"q{branch}"][query]
            k = tables[f"k{branch}"][key]
            columns = []
            for offset in OFFSETS:
                columns.append(
                    (apply_rot(q, cos[offset], sin[offset]) * k).sum(-1) / HD)
            per_branch.append(torch.stack(columns, 1))
        heads = {}
        for head in range(N_HEAD):
            shares = []
            spectra = []
            controls = []
            for branch_index in range(2):
                matrix = per_branch[branch_index][:, :, head]
                share, spectrum = _interaction_top1(matrix)
                shares.append(share)
                spectra.append(spectrum)
                shuffled = torch.stack([
                    matrix[torch.randperm(
                        PAIR_COUNT, generator=shuffle_generator).to(device), c]
                    for c in range(len(OFFSETS))], 1)
                control_share, _cs = _interaction_top1(shuffled)
                controls.append(control_share)
            heads[str(head)] = {
                "top1_share": sum(shares) / 2,
                "per_branch_top1": shares,
                "spectra": spectra,
                "control_top1_max": max(controls),
                "control_top1_min": min(controls),
            }
        results[str(seed)] = heads

    vectors = {
        seed: [results[str(seed)][str(h)]["top1_share"] for h in range(N_HEAD)]
        for seed in SEEDS}
    spearman = _spearman(vectors[SEEDS[0]], vectors[SEEDS[1]])
    control_values = [
        results[str(seed)][str(h)][key]
        for seed in SEEDS for h in range(N_HEAD)
        for key in ("control_top1_max", "control_top1_min")]

    def median(vector):
        return sorted(vector)[len(vector) // 2]

    ratios = {
        seed: vectors[seed][HEAD_POS] / max(vectors[seed][HEAD_CONJ], 1e-30)
        for seed in SEEDS}
    pred_a = (
        max(fold_errors.values()) <= 1e-10
        and spearman >= .8
        and all(.05 < value < .30 for value in control_values))
    pred_b = all(
        vectors[seed][HEAD_POS] >= .60 and ratios[seed] >= 1.5
        for seed in SEEDS)
    pred_c = all(
        vectors[seed][HEAD_CONJ] < median(vectors[seed]) for seed in SEEDS)
    null = (
        all(ratios[seed] < 1.2 for seed in SEEDS)
        or any(vectors[seed][HEAD_POS] < .40 for seed in SEEDS))

    result = {
        "status": "attention0_interaction_rank_taxonomy_complete",
        "rung": 432,
        "claim_level": "head_mechanism_taxonomy_identification_screen_not_compression",
        "fold_max_abs_by_branch": fold_errors,
        "samples": results,
        "top1_share_vectors": {str(k): v for k, v in vectors.items()},
        "head3_over_head6_ratio": {str(k): v for k, v in ratios.items()},
        "cross_population_spearman": spearman,
        'pred_a_instrument_and_sane_controls': bool(pred_a),
        'pred_b_head3_low_rank_modulated_positional': bool(pred_b),
        'pred_c_head6_high_rank_conjunctive': bool(pred_c),
        'null_taxonomy_wrong_keep_main_effect_facts_only': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": "census_taxonomy_statement_only",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
