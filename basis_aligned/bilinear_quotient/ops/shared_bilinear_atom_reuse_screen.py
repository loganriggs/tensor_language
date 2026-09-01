"""RUNG 299 -- DIRECT CROSS-LAYER SHARING OF BILINEAR MLP TENSORS.

Question
--------
Do the eighteen native MLP coefficient tensors share enough *literal bilinear
atoms* or whole-tensor layer modes to support a substantially cheaper joint CP
program?  This is deliberately not another within-layer HOSVD/Tucker sweep;
those coefficient-space mode ranks are already known to be diffuse.

For MLP layer j write the gauge-invariant symmetric coefficient tensor as

    T_j = sum_u d_ju tensor sym(l_ju tensor r_ju).

The normalized inner product of two atoms is computed exactly without
materializing a 1152^3 tensor:

    <S(l,r),S(l',r')> = 1/2[(l.l')(r.r') + (l.r')(r.l')].

Absolute atom cosine is invariant to CP permutation, reciprocal rescaling,
Left/Right swaps, and sign absorbed into a layer coefficient.  We measure
sampled-source/all-target nearest matches for 35 frozen layer pairs.  A null
applies independent signed coordinate permutations to the target output and
input modes; it preserves every within-layer atom norm and inner product while
destroying cross-layer alignment.

Independently, a randomized polarization sketch estimates the layer-mode
spectrum of the full invariant tensors.  Each measurement is

    a^T T_j[x,y] = sum_u (a.d_u) *
        ((x.l_u)(y.r_u) + (x.r_u)(y.l_u))/2,

with independent Gaussian a,x,y.  This is an unbiased coefficient-Frobenius
inner-product sketch without the trace contamination of quadratic x=x probes.
Two seeds are frozen.  Independent signed-coordinate nulls receive the same
sketches.

Positive and negative controls
------------------------------
A planted six-layer bank shares gauge-scrambled atoms and has a rank-3 layer
coefficient matrix.  It must show >=95% exact atom matches and >=98% top-3
layer-mode energy.  An independently drawn bank must have <5% matches at 0.95.

Frozen real predictions
-----------------------
pred_a_native_atom_reuse:
    Some frozen real layer pair has >=10% one-to-one source matches at absolute
    tensor cosine >=0.95, and its median nearest-match cosine exceeds its
    coordinate null by >=0.15.
pred_b_shared_layer_basis:
    Across both sketch seeds, the normalized layer-mode top-13 energy is >=0.95
    and exceeds the matched null by >=0.10.  Rank 13 is the largest whole-layer
    basis count compatible with at least 25% raw MLP storage saving if every
    basis tensor can subsequently be represented with one native-width CP bank.
pred_c_optimistic_25pct_reuse_capacity:
    The mean one-to-one >=0.95 match fraction across the 17 adjacent-layer
    pairs is at least the exact pooled-atom reuse fraction required for a 25%
    joint-bank saving.  This is an optimistic capacity screen, not a constructed
    global matching or an adoption claim.

Null / decision
---------------
If the positive control fails, the instrument is invalid.  If all real
predictions fail and real match/spectrum statistics are close to the orthogonal
nulls, prune *native-atom reuse* and low-rank whole-layer mixing.  Arbitrary
joint CP refactorization under activation/Fisher metrics remains logically open.

Literal prices
--------------
Native eighteen-MLP bank, including biases:
    18 * (4608 * 3 * 1152 + 1152) = 286,675,200 scalars.
A common R-atom bank with one coefficient per atom per layer costs:
    R * (3*1152 + 18) + 18*1152 scalars.
The exact R ceiling and pooled-atom reuse fraction for a 25% saving are emitted.
No checkpoint storage is subtracted until an executable replacement exists.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch


OUT = "/workspace/tensor_language/basis_aligned/bilinear_quotient/shared_bilinear_atom_reuse_screen_results.json"
DEV = "cuda"
D = 1152
H = 4608
N_LAYER = 18
ANCHORS = 128
SKETCH_M = 768
SKETCH_SEEDS = (29930, 29931)
THRESHOLDS = (0.80, 0.90, 0.95, 0.99)


def _sym_atom_norm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(
        0.5 * (
            left.square().sum(1) * right.square().sum(1)
            + (left * right).sum(1).square()
        )
    ).clamp_min(1e-12)


@torch.no_grad()
def _similarity(
    source: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    target: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    source_index: torch.Tensor,
) -> torch.Tensor:
    down_s, left_s, right_s = source
    down_t, left_t, right_t = target
    ds = down_s[source_index]
    ls = left_s[source_index]
    rs = right_s[source_index]
    output_dot = ds @ down_t.T
    input_dot = 0.5 * ((ls @ left_t.T) * (rs @ right_t.T) + (ls @ right_t.T) * (rs @ left_t.T))
    ns = ds.norm(dim=1) * _sym_atom_norm(ls, rs)
    nt = down_t.norm(dim=1) * _sym_atom_norm(left_t, right_t)
    return (output_dot * input_dot / (ns[:, None] * nt[None, :]).clamp_min(1e-20)).abs()


def _greedy_unique_fraction(similarity: torch.Tensor, threshold: float) -> float:
    best_value, best_index = similarity.max(1)
    order = torch.argsort(best_value, descending=True)
    used: set[int] = set()
    accepted = 0
    for row in order.tolist():
        value = float(best_value[row])
        target = int(best_index[row])
        if value < threshold:
            break
        if target not in used:
            used.add(target)
            accepted += 1
    return accepted / len(best_value)


def _summary(similarity: torch.Tensor) -> dict[str, object]:
    nearest = similarity.max(1).values
    q = torch.quantile(nearest, torch.tensor((0.10, 0.25, 0.50, 0.75, 0.90, 0.99), device=nearest.device))
    return {
        "nearest_quantiles": {
            name: float(value)
            for name, value in zip(("q10", "q25", "q50", "q75", "q90", "q99"), q)
        },
        "fraction_over_threshold": {
            str(threshold): float((nearest >= threshold).float().mean())
            for threshold in THRESHOLDS
        },
        "greedy_one_to_one_fraction": {
            str(threshold): _greedy_unique_fraction(similarity, threshold)
            for threshold in THRESHOLDS
        },
    }


def _coordinate_null(
    bank: tuple[torch.Tensor, torch.Tensor, torch.Tensor], seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    down, left, right = bank
    generator = torch.Generator(device=DEV).manual_seed(seed)
    output_perm = torch.randperm(down.shape[1], generator=generator, device=DEV)
    input_perm = torch.randperm(left.shape[1], generator=generator, device=DEV)
    output_sign = torch.where(
        torch.rand(down.shape[1], generator=generator, device=DEV) > 0.5, 1.0, -1.0,
    )
    input_sign = torch.where(
        torch.rand(left.shape[1], generator=generator, device=DEV) > 0.5, 1.0, -1.0,
    )
    return (
        down[:, output_perm] * output_sign,
        left[:, input_perm] * input_sign,
        right[:, input_perm] * input_sign,
    )


def _toy_bank(shared: bool, seed: int) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
    generator = torch.Generator(device=DEV).manual_seed(seed)
    toy_d, toy_h, layers, layer_rank = 48, 96, 6, 3
    base_down = torch.randn(toy_h, toy_d, generator=generator, device=DEV)
    base_left = torch.randn(toy_h, toy_d, generator=generator, device=DEV)
    base_right = torch.randn(toy_h, toy_d, generator=generator, device=DEV)
    codes = torch.randn(layers, layer_rank, generator=generator, device=DEV)
    atom_codes = torch.randn(layer_rank, toy_h, generator=generator, device=DEV)
    coefficients = codes @ atom_codes
    banks = []
    for layer in range(layers):
        if shared:
            down, left, right = base_down.clone(), base_left.clone(), base_right.clone()
        else:
            down = torch.randn(toy_h, toy_d, generator=generator, device=DEV)
            left = torch.randn(toy_h, toy_d, generator=generator, device=DEV)
            right = torch.randn(toy_h, toy_d, generator=generator, device=DEV)
        scale_l = 0.4 + torch.rand(toy_h, generator=generator, device=DEV)
        scale_r = 0.4 + torch.rand(toy_h, generator=generator, device=DEV)
        if shared:
            down = down * (coefficients[layer] / (scale_l * scale_r))[:, None]
        left = left * scale_l[:, None]
        right = right * scale_r[:, None]
        swap = torch.rand(toy_h, generator=generator, device=DEV) > 0.5
        old_left = left.clone()
        left[swap] = right[swap]
        right[swap] = old_left[swap]
        permutation = torch.randperm(toy_h, generator=generator, device=DEV)
        banks.append((down[permutation], left[permutation], right[permutation]))
    return banks, coefficients


@torch.no_grad()
def _layer_sketch(
    banks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], seed: int,
) -> dict[str, object]:
    width = banks[0][0].shape[1]
    generator = torch.Generator(device=DEV).manual_seed(seed)
    a = torch.randn(SKETCH_M, width, generator=generator, device=DEV) / math.sqrt(width)
    x = torch.randn(SKETCH_M, width, generator=generator, device=DEV) / math.sqrt(width)
    y = torch.randn(SKETCH_M, width, generator=generator, device=DEV) / math.sqrt(width)
    rows = []
    for down, left, right in banks:
        output = a @ down.T
        polarized = 0.5 * ((x @ left.T) * (y @ right.T) + (x @ right.T) * (y @ left.T))
        rows.append((output * polarized).sum(1))
    matrix = torch.stack(rows)
    raw_singular = torch.linalg.svdvals(matrix)
    normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normalized_singular = torch.linalg.svdvals(normalized)

    def curve(values: torch.Tensor) -> dict[str, float]:
        energy = values.square()
        cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-20)
        requested = (1, 3, 6, 9, 12, min(13, len(values)), len(values))
        ranks = sorted({rank for rank in requested if 1 <= rank <= len(values)})
        return {str(rank): float(cumulative[rank - 1]) for rank in ranks}

    return {
        "raw_energy_curve": curve(raw_singular),
        "row_normalized_energy_curve": curve(normalized_singular),
        "row_norm_min": float(matrix.norm(dim=1).min()),
        "row_norm_max": float(matrix.norm(dim=1).max()),
    }


def _toy_controls() -> dict[str, object]:
    shared_banks, _ = _toy_bank(True, 29900)
    independent_banks, _ = _toy_bank(False, 29901)
    index = torch.arange(shared_banks[0][0].shape[0], device=DEV)
    shared_summary = _summary(_similarity(shared_banks[0], shared_banks[1], index))
    negative_summary = _summary(_similarity(independent_banks[0], independent_banks[1], index))

    # The common factors with a rank-3 coefficient matrix make the exact stack
    # rank <=3. Use more measurements for the toy than tensor coordinates need.
    global SKETCH_M
    old_m = SKETCH_M
    SKETCH_M = 512
    shared_spectrum = _layer_sketch(shared_banks, 29902)
    SKETCH_M = old_m
    positive_valid = bool(
        shared_summary["greedy_one_to_one_fraction"]["0.95"] >= 0.95
        and shared_spectrum["row_normalized_energy_curve"]["3"] >= 0.98
        and negative_summary["greedy_one_to_one_fraction"]["0.95"] < 0.05
    )
    return {
        "shared_atom_match": shared_summary,
        "independent_negative_match": negative_summary,
        "shared_layer_spectrum": shared_spectrum,
        "positive_control_valid": positive_valid,
    }


@torch.no_grad()
def _load_real_banks() -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    sys.path.insert(0, "/workspace/tensor_language/basis_aligned/qk_mdl")
    from tier2_model import load_elriggs

    model, cfg = load_elriggs("bilin18")
    assert cfg["n_embd"] == D and len(model.transformer.h) == N_LAYER
    banks = []
    for block in model.transformer.h:
        mlp = block.mlp
        down = mlp.Down.weight.detach().T.float().contiguous()
        left = mlp.Left.weight.detach().float().contiguous()
        right = mlp.Right.weight.detach().float().contiguous()
        assert down.shape == left.shape == right.shape == (H, D)
        banks.append((down, left, right))
    return banks


def _real_pairs() -> list[tuple[int, int]]:
    pairs = {(layer, layer + 1) for layer in range(17)}
    pairs.update((layer, layer + 6) for layer in range(12))
    pairs.update((layer, layer + 12) for layer in range(6))
    return sorted(pairs)


@torch.no_grad()
def _real_screen(banks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> dict[str, object]:
    generator = torch.Generator(device=DEV).manual_seed(29920)
    source_indices = {
        layer: torch.randperm(H, generator=generator, device=DEV)[:ANCHORS]
        for layer in range(N_LAYER)
    }
    pair_results: dict[str, object] = {}
    for source_layer, target_layer in _real_pairs():
        index = source_indices[source_layer]
        real = _summary(_similarity(banks[source_layer], banks[target_layer], index))
        null_bank = _coordinate_null(banks[target_layer], 299000 + 100 * source_layer + target_layer)
        null = _summary(_similarity(banks[source_layer], null_bank, index))
        key = f"{source_layer}-{target_layer}"
        pair_results[key] = {
            "real": real,
            "coordinate_null": null,
            "median_gap": real["nearest_quantiles"]["q50"] - null["nearest_quantiles"]["q50"],
        }

    real_spectra = []
    null_spectra = []
    null_banks = [_coordinate_null(bank, 299500 + layer) for layer, bank in enumerate(banks)]
    for seed in SKETCH_SEEDS:
        real_spectra.append(_layer_sketch(banks, seed))
        null_spectra.append(_layer_sketch(null_banks, seed))

    adjacent = [pair_results[f"{layer}-{layer + 1}"] for layer in range(17)]
    adjacent_match = float(np.mean([
        pair["real"]["greedy_one_to_one_fraction"]["0.95"] for pair in adjacent
    ]))
    best_key = max(
        pair_results,
        key=lambda key: pair_results[key]["real"]["greedy_one_to_one_fraction"]["0.95"],
    )
    return {
        "anchors_per_source_layer": ANCHORS,
        "pairs": pair_results,
        "best_pair_by_greedy_095": best_key,
        "mean_adjacent_greedy_095": adjacent_match,
        "real_layer_spectra": real_spectra,
        "coordinate_null_layer_spectra": null_spectra,
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print("SHARED BILINEAR ATOM REUSE SCREEN | dry run: frozen controls, pairs, prices, and bars valid")
        return
    started = time.time()
    native_price = N_LAYER * (H * 3 * D + D)
    target_price = math.floor(0.75 * native_price)
    rank_ceiling = math.floor((target_price - N_LAYER * D) / (3 * D + N_LAYER))
    pooled_atoms = N_LAYER * H
    required_reuse_fraction = 1.0 - rank_ceiling / pooled_atoms
    price = {
        "native_18_mlp_scalars": native_price,
        "target_25pct_saving_scalars": target_price,
        "common_bank_rank_ceiling": rank_ceiling,
        "pooled_native_atoms": pooled_atoms,
        "required_pooled_atom_reuse_fraction": required_reuse_fraction,
        "common_bank_formula": "R*(3*1152+18)+18*1152",
    }
    print("SHARED BILINEAR ATOM REUSE | rung 299 planted controls", flush=True)
    controls = _toy_controls()
    print(json.dumps(controls, indent=2), flush=True)
    print("SHARED BILINEAR ATOM REUSE | rung 299 real invariant coefficient tensors", flush=True)
    banks = _load_real_banks()
    real = _real_screen(banks)
    best = real["pairs"][real["best_pair_by_greedy_095"]]
    pred_a = bool(
        best["real"]["greedy_one_to_one_fraction"]["0.95"] >= 0.10
        and best["median_gap"] >= 0.15
    )
    top13_real = np.array([
        item["row_normalized_energy_curve"]["13"] for item in real["real_layer_spectra"]
    ])
    top13_null = np.array([
        item["row_normalized_energy_curve"]["13"] for item in real["coordinate_null_layer_spectra"]
    ])
    pred_b = bool(np.all(top13_real >= 0.95) and np.all(top13_real - top13_null >= 0.10))
    pred_c = bool(real["mean_adjacent_greedy_095"] >= required_reuse_fraction)
    instrument_valid = bool(controls["positive_control_valid"])
    result = {
        "status": "shared_bilinear_atom_reuse_screen_complete",
        "rung": 299,
        "claim_level": "coefficient_space_screen_only",
        "price": price,
        "controls": controls,
        "real": real,
        "instrument_valid": instrument_valid,
        'pred_a_native_atom_reuse': pred_a if instrument_valid else None,
        'pred_b_shared_layer_basis': pred_b if instrument_valid else None,
        'pred_c_optimistic_25pct_reuse_capacity': pred_c if instrument_valid else None,
        "null_all_real_predictions_fail": bool(instrument_valid and not (pred_a or pred_b or pred_c)),
        "runtime_s": time.time() - started,
    }
    with open(OUT, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        "price": price,
        "best_pair": real["best_pair_by_greedy_095"],
        "best_pair_receipt": best,
        "mean_adjacent_greedy_095": real["mean_adjacent_greedy_095"],
        "top13_real": top13_real.tolist(),
        "top13_null": top13_null.tolist(),
        "predicates": [pred_a, pred_b, pred_c],
        "instrument_valid": instrument_valid,
        "runtime_s": result["runtime_s"],
    }, indent=2), flush=True)
    print("SHARED BILINEAR ATOM REUSE SCREEN DONE", flush=True)


if __name__ == "__main__":
    main()
