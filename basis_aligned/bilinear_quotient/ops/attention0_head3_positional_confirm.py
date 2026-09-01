"""RUNG 429 (Claude red-team lane) -- CONFIRMATION: HEAD 3 IS THE POSITIONAL HEAD.

428 falsified my head-6 positional hypothesis and surfaced the inverse
discovery: head 3 carries the runaway offset main effect (.1227 vs
second place .0461) and the minimum token main effect (.313), on both
disjoint samples.  Per my own guard discipline the surprise gets a
confirmation rung before anyone builds on it: a THIRD disjoint
SELECT-token sample (seed 429_777) plus a FIT-POPULATION transfer
sample (seed 429_888, tokens mod5 != 4 -- a population never used for
these shares), with the control bar CORRECTED to the theory floor
(shuffled pair main-effect share under O=8 offsets concentrates at
1/O = .125; 428 measured .12644 -- the .11-.14 window replaces my
mis-derived <=.01 bar, documented in ledger 2548).

423/423b established with two independent solvers that head 6 is the
token-input-geometry OUTSIDER of attention0 (payload<->carrier overlap
.074-.091 vs heads 3/7 at ~.40) and has the weakest MLP0-L alignment
(.17).  Hypothesis registered here: head 6's score formation is
dominated by ROTARY OFFSET rather than token identity -- it is the
positional head, which would explain why it barely participates in the
shared token geometry.

Construction (426's exact machinery): per-branch folded token tables
q_b/k_b (float32, fold-gated in float64 at <=1e-10), 32,768 random
SELECT-token pairs per sample (two disjoint samples, seeds 428_777 and
428_888, both distinct from 426's 426_777), scores
s_b[pair, offset, head] over offsets (1,2,4,8,16,32,64,128) via exact
rotary application.  Two-way variance decomposition per head/branch:
offset-share = squared norm of the offset main effect / total centered
sum of squares; token-share = same for the pair main effect
(interaction reported).  Branch-mean shares per head are the scored
object.  Controls: per-offset independent pair permutation (destroys
the pair main effect; validates the estimator), and the second disjoint
pair sample (split stability).

Arms: sample 429_777 = third disjoint SELECT-token pair sample; sample
429_888 = FIT-population pair sample (transfer).

Frozen predictions
------------------
pred_a (instrument, corrected control): float64 fold gate <= 1e-10;
    shuffled-pair control token-share in (.11, .14) for EVERY head (the
    1/O theory window); SELECT-vs-FIT offset-share Spearman >= .8.
pred_b (head-3 positional confirms): head 3's branch-mean offset-share
    is the MAXIMUM of the 9 heads AND >= 2x the second-largest head, on
    BOTH populations (428 measured 2.66x on SELECT).
pred_c (profile stability): head 3's token-share is the MINIMUM on both
    populations AND its centered offset-effect profile has cosine >= .9
    between the two populations (branch-mean).

Null: head 3 is not the offset-share maximum on either population, or
the >=2x margin fails on both -- the head-3 positional finding of 2548
was sample- or population-specific and must not enter the dossier.

Price: identification screen only; no shipped object; no prior bar is
altered (428 stays scored triple-fail as written).
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
OUT = BQ / "attention0_head3_positional_confirm_results.json"
SV = OPS / "attention0_cross_head_sparse_qk_vocabulary.py"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
VOCAB = 50_257
N_HEAD = 9
HD = 128
OFFSETS = (1, 2, 4, 8, 16, 32, 64, 128)
PAIR_COUNT = 32_768
SEEDS = (429_777, 429_888)
TARGET_HEAD = 3


def _shares(matrix: torch.Tensor) -> dict:
    """Two-way variance shares for one [pairs, offsets] score matrix."""
    centered = matrix.double() - matrix.double().mean()
    total = float(centered.square().sum())
    offset_effect = centered.mean(0, keepdim=True)
    pair_effect = centered.mean(1, keepdim=True)
    offset_ss = float(offset_effect.square().sum()) * matrix.shape[0]
    pair_ss = float(pair_effect.square().sum()) * matrix.shape[1]
    interaction = (centered - offset_effect - pair_effect)
    return {
        "offset_share": offset_ss / max(total, 1e-30),
        "token_share": pair_ss / max(total, 1e-30),
        "interaction_share": float(interaction.square().sum()) / max(total, 1e-30),
    }


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
        assert TARGET_HEAD == 3 and PAIR_COUNT == 32_768
        assert not {426_777, 428_777, 428_888} & set(SEEDS) and len(set(SEEDS)) == 2
        assert SV.exists() and ROWS_RECEIPT.exists()
        print("ATTENTION0 HEAD3 POSITIONAL CONFIRM | dry run: third sample + FIT transfer, corrected control window")
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

    samples = {}
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
            per_branch.append(torch.stack(columns, 1))  # [pairs, offsets, heads]
        samples[seed] = per_branch

    results = {}
    shuffle_generator = torch.Generator(device="cpu").manual_seed(428_999)
    for seed, per_branch in samples.items():
        heads = {}
        for head in range(N_HEAD):
            branch_shares = []
            control_token = []
            for branch_index in range(2):
                matrix = per_branch[branch_index][:, :, head]
                branch_shares.append(_shares(matrix))
                shuffled = torch.stack([
                    matrix[torch.randperm(PAIR_COUNT, generator=shuffle_generator).to(device), c]
                    for c in range(len(OFFSETS))], 1)
                control_token.append(_shares(shuffled)["token_share"])
            profile = []
            for branch_index in range(2):
                matrix = per_branch[branch_index][:, :, head].double()
                effect = matrix.mean(0) - matrix.mean()
                profile.append(effect.cpu().tolist())
            heads[str(head)] = {
                "offset_effect_profile": profile,
                "offset_share": sum(s["offset_share"] for s in branch_shares) / 2,
                "token_share": sum(s["token_share"] for s in branch_shares) / 2,
                "interaction_share": sum(s["interaction_share"] for s in branch_shares) / 2,
                "per_branch": branch_shares,
                "shuffled_token_share_max": max(control_token),
            }
        results[str(seed)] = heads

    offset_vectors = {
        seed: [results[str(seed)][str(h)]["offset_share"] for h in range(N_HEAD)]
        for seed in SEEDS}
    token_vectors = {
        seed: [results[str(seed)][str(h)]["token_share"] for h in range(N_HEAD)]
        for seed in SEEDS}
    spearman = _spearman(offset_vectors[SEEDS[0]], offset_vectors[SEEDS[1]])
    control_max = max(
        results[str(seed)][str(h)]["shuffled_token_share_max"]
        for seed in SEEDS for h in range(N_HEAD))

    def is_max(vector, index):
        return vector[index] == max(vector)

    def is_min(vector, index):
        return vector[index] == min(vector)

    def median(vector):
        return sorted(vector)[len(vector) // 2]

    control_min = min(
        results[str(seed)][str(h)]["shuffled_token_share_max"]
        for seed in SEEDS for h in range(N_HEAD))
    cosines = []
    for branch_index in range(2):
        left = torch.tensor(
            results[str(SEEDS[0])][str(TARGET_HEAD)]["offset_effect_profile"][branch_index])
        right = torch.tensor(
            results[str(SEEDS[1])][str(TARGET_HEAD)]["offset_effect_profile"][branch_index])
        cosines.append(float(
            (left @ right) / (left.norm() * right.norm()).clamp_min(1e-30)))
    profile_cosine = sum(cosines) / 2

    def second_largest(vector):
        return sorted(vector)[-2]

    pred_a = (
        max(fold_errors.values()) <= 1e-10
        and .11 < control_min and control_max < .14
        and spearman >= .8)
    pred_b = all(
        is_max(offset_vectors[seed], TARGET_HEAD)
        and offset_vectors[seed][TARGET_HEAD]
            >= 2 * second_largest(offset_vectors[seed])
        for seed in SEEDS)
    pred_c = (all(
        is_min(token_vectors[seed], TARGET_HEAD) for seed in SEEDS)
        and profile_cosine >= .9)
    null = (
        not any(is_max(offset_vectors[seed], TARGET_HEAD) for seed in SEEDS)
        or all(offset_vectors[seed][TARGET_HEAD]
               < 2 * second_largest(offset_vectors[seed]) for seed in SEEDS))

    result = {
        "status": "attention0_head3_positional_confirm_complete",
        "profile_cosine_head3": profile_cosine,
        "rung": 429,
        "claim_level": "head_mechanism_identification_screen_not_compression",
        "fold_max_abs_by_branch": fold_errors,
        "samples": results,
        "offset_share_vectors": {str(k): v for k, v in offset_vectors.items()},
        "token_share_vectors": {str(k): v for k, v in token_vectors.items()},
        "cross_sample_offset_spearman": spearman,
        "shuffled_control_token_share_max": control_max,
        'pred_a_instrument_with_corrected_control_window': bool(pred_a),
        'pred_b_head3_positional_maximum_2x_both_populations': bool(pred_b),
        'pred_c_head3_token_minimum_and_profile_stable': bool(pred_c),
        'null_head3_positional_finding_sample_specific': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": "mechanism_statement_only",
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
