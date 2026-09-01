"""RUNG 380 -- GAUGE-INVARIANT TUCKER RECOVERY TOY.

Recover planted input/output Tucker subspaces from the invariant symmetric
bilinear tensor, not its gauge-dependent CP factors.  Verify fresh values and
directional derivatives; a dense matched-norm tensor is the negative.

Frozen predictions
------------------
pred_a: planted overlaps>=.99 and tensor/value/Jacobian R2>=.999.
pred_b: factor gauge scramble changes tensor<=1e-6 and projectors<=.001.
pred_c: dense negative projected tensor/value R2<=.50; dimensions/price exact.

Null: planted value/Jacobian R2<.95, gauge error>1e-4, or dense value R2>=.80.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "bilinear_invariant_tucker_toy_results.json"
D = 48
R = 12
P = 10
K = 18
SEED = 2026090101
TUCKER_PRICE = D * R + D * P + P * R * (R + 1) // 2
DENSE_PRICE = D * D * (D + 1) // 2


def _orth(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(cols, rows, generator=generator))
    return q.T


def _tensor(q, u, a, b, c):
    left = a @ q
    right = b @ q
    sym = .5 * (torch.einsum("ka,kb->kab", left, right)
                + torch.einsum("ka,kb->kab", right, left))
    return torch.einsum("op,pk,kab->oab", u, c, sym)


def _recover(tensor):
    u, _s, _vh = torch.linalg.svd(tensor.reshape(D, D * D), full_matrices=False)
    out = u[:, :P]
    mode_a = tensor.permute(1, 0, 2).reshape(D, D * D)
    mode_b = tensor.permute(2, 0, 1).reshape(D, D * D)
    gram = mode_a @ mode_a.T + mode_b @ mode_b.T
    values, vectors = torch.linalg.eigh(gram)
    inp = vectors[:, torch.argsort(values, descending=True)[:R]]
    pout, pin = out @ out.T, inp @ inp.T
    projected = torch.einsum("oi,iaj,ab->ojb", pout, tensor, pin)
    projected = torch.einsum("ojb,bc->ojc", projected, pin)
    return out, inp, projected


def _r2(reference, estimate):
    return float(1.0 - (reference - estimate).square().sum()
                 / (reference - reference.mean()).square().sum().clamp_min(1e-30))


def _subspace(true_rows, estimated_cols):
    return float((true_rows @ estimated_cols).square().sum() / true_rows.shape[0])


def _function(tensor, x):
    return torch.einsum("oab,na,nb->no", tensor, x, x)


def _jvp(tensor, x, delta):
    return (torch.einsum("oab,na,nb->no", tensor, delta, x)
            + torch.einsum("oab,na,nb->no", tensor, x, delta))


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert TUCKER_PRICE == 1_836 and DENSE_PRICE == 56_448
        assert R < D and P < D and K > max(R, P)
        print("INVARIANT TUCKER TOY | dry run: dims, price, bars valid")
        return

    g = torch.Generator().manual_seed(SEED)
    q = _orth(R, D, g)
    u_rows = _orth(P, D, g)
    u = u_rows.T
    a = torch.randn(K, R, generator=g) / R ** .5
    b = torch.randn(K, R, generator=g) / R ** .5
    c = torch.randn(P, K, generator=g) / K ** .5
    teacher = _tensor(q, u, a, b, c)

    permutation = torch.randperm(K, generator=g)
    scale = torch.exp(.5 * torch.randn(K, generator=g))
    swap = torch.rand(K, generator=g) < .5
    a_scrambled = a[permutation].clone()
    b_scrambled = b[permutation].clone()
    a_scrambled[swap], b_scrambled[swap] = (b_scrambled[swap].clone(),
                                            a_scrambled[swap].clone())
    a_scrambled *= scale[:, None]
    c_scrambled = c[:, permutation] / scale[None, :]
    scrambled = _tensor(q, u, a_scrambled, b_scrambled, c_scrambled)
    gauge_relative_error = float((teacher - scrambled).norm() / teacher.norm())

    out, inp, projected = _recover(teacher)
    out_s, inp_s, projected_s = _recover(scrambled)
    input_overlap = _subspace(q, inp)
    output_overlap = _subspace(u_rows, out)
    recovered_projector_overlap = min(
        float((inp.T @ inp_s).square().sum() / R),
        float((out.T @ out_s).square().sum() / P))
    tensor_r2 = _r2(teacher, projected)

    x = torch.randn(1024, D, generator=g)
    delta = torch.randn(1024, D, generator=g)
    value_r2 = _r2(_function(teacher, x), _function(projected, x))
    jacobian_r2 = _r2(_jvp(teacher, x, delta), _jvp(projected, x, delta))

    dense = torch.randn(D, D, D, generator=g)
    dense = .5 * (dense + dense.transpose(1, 2))
    dense *= teacher.norm() / dense.norm()
    _out_d, _inp_d, dense_projected = _recover(dense)
    dense_tensor_r2 = _r2(dense, dense_projected)
    dense_value_r2 = _r2(_function(dense, x), _function(dense_projected, x))

    pred_a = (input_overlap >= .99 and output_overlap >= .99 and tensor_r2 >= .999
              and value_r2 >= .999 and jacobian_r2 >= .999)
    pred_b = gauge_relative_error <= 1e-6 and recovered_projector_overlap >= .999
    pred_c = (dense_tensor_r2 <= .50 and dense_value_r2 <= .50
              and TUCKER_PRICE == 1_836 and DENSE_PRICE == 56_448)
    null = (value_r2 < .95 or jacobian_r2 < .95 or gauge_relative_error > 1e-4
            or dense_value_r2 >= .80)
    result = {
        "status": "bilinear_invariant_tucker_toy_complete",
        "rung": 380,
        "claim_level": "planted_gauge_invariant_bilinear_tucker_instrument_only",
        "dimensions": {"d": D, "input_rank": R, "output_rank": P,
                       "product_width": K},
        "literal_symmetric_tucker_price": TUCKER_PRICE,
        "literal_dense_symmetric_tensor_price": DENSE_PRICE,
        "saving_fraction": (DENSE_PRICE - TUCKER_PRICE) / DENSE_PRICE,
        "input_subspace_overlap": input_overlap,
        "output_subspace_overlap": output_overlap,
        "gauge_relative_tensor_error": gauge_relative_error,
        "gauge_recovered_projector_overlap_min": recovered_projector_overlap,
        "planted_tensor_r2": tensor_r2,
        "planted_fresh_value_r2": value_r2,
        "planted_fresh_directional_derivative_r2": jacobian_r2,
        "dense_negative_tensor_r2": dense_tensor_r2,
        "dense_negative_fresh_value_r2": dense_value_r2,
        'pred_a_planted_tucker_function_and_subspaces_recover': bool(pred_a),
        'pred_b_recovery_is_bilinear_factor_gauge_invariant': bool(pred_b),
        'pred_c_dense_negative_does_not_look_tucker_low_rank': bool(pred_c),
        "null_tucker_instrument_is_not_discriminating": bool(null),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
