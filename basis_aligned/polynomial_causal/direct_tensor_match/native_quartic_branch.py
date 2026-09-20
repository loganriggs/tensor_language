"""Isolate the pure MLP16 -> MLP17 quartic branch; retain normalization externally."""
import torch


def bilinear(x, left, right, down):
    return ((x @ left.T) * (x @ right.T)) @ down.T


def pure_branch(previous_output, previous_bias, residual_scale, left, right, down):
    # Previous output includes a bias which is NOT part of the homogeneous quartic.
    m = residual_scale * (previous_output - previous_bias)
    return bilinear(m, left, right, down)


def replace_branch(native_output, prenorm_input, original_pure, replacement_pure, eps=None):
    """All outputs are residual-space vectors, before the next/final RMSNorm.

    prenorm_input is the ACTUAL second MLP input before RMSNorm. Its denominator
    stays fixed: this edits one additive term in its bilinear expansion.
    """
    if eps is None:
        eps = torch.finfo(prenorm_input.dtype).eps
    denominator = prenorm_input.square().mean(-1, keepdim=True) + eps
    return native_output + (replacement_pure - original_pure) / denominator


def oracle():
    """Independent expanded-polynomial check including bias, cross terms and lambda."""
    torch.manual_seed(260920)
    torch.set_num_threads(2)
    dtype = torch.float64
    x = torch.randn(31, 7, dtype=dtype)
    A, B = [torch.randn(9, 7, dtype=dtype) / 3 for _ in range(2)]
    D = torch.randn(7, 9, dtype=dtype) / 3
    L, R = [torch.randn(11, 7, dtype=dtype) / 3 for _ in range(2)]
    O = torch.randn(7, 11, dtype=dtype) / 3
    b16, b17 = [torch.randn(7, dtype=dtype) for _ in range(2)]
    lam = 0.63
    prev = bilinear(x, A, B, D) + b16
    background = torch.randn_like(x)
    m = lam * (prev - b16)
    h = background + lam * prev
    r = background + lam * b16
    eps = torch.finfo(dtype).eps
    den = h.square().mean(-1, keepdim=True) + eps
    native = bilinear(torch.nn.functional.rms_norm(h, (7,)), L, R, O) + b17
    pure = pure_branch(prev, b16, lam, L, R, O)
    cross = ((r @ L.T)*(m @ R.T) + (m @ L.T)*(r @ R.T)) @ O.T
    replacement = torch.randn_like(pure)
    expected = (bilinear(r, L, R, O) + cross + replacement) / den + b17
    actual = replace_branch(native, h, pure, replacement)
    relative = lambda a,b: float((a-b).norm()/b.norm())
    result = dict(replacement_replay=relative(actual,expected),
                  exact_noop=relative(replace_branch(native,h,pure,pure),native),
                  ablation_replay=relative(replace_branch(native,h,pure,torch.zeros_like(pure)),
                                          (bilinear(r,L,R,O)+cross)/den+b17),
                  wrong_bias_error=relative(replace_branch(native,h,bilinear(lam*prev,L,R,O),replacement),expected),
                  wrong_scale_error=relative(replace_branch(native,h,bilinear(prev-b16,L,R,O),replacement),expected),
                  wrong_denominator_error=relative(native+replacement-pure,expected))
    assert max(result[k] for k in ('replacement_replay','exact_noop','ablation_replay')) < 1e-12
    assert min(result[k] for k in ('wrong_bias_error','wrong_scale_error','wrong_denominator_error')) > .01
    return result


if __name__ == '__main__':
    import json
    from pathlib import Path
    result = oracle()
    Path(__file__).with_name('NATIVE_QUARTIC_BRANCH_ORACLE_V1.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
