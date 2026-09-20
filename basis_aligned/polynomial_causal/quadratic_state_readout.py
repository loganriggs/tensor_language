"""Exact readout of a quadratic hidden-state curve, with a shared quartic norm."""
from final_readout_field_program import evaluate_fields


def compile_core(states, rows):
    """states[B,3,D] contains coefficients of 1,t,t²; rows[B,O,2,D]."""
    import torch
    assert states.shape[1] == 3 and states.shape[2] >= 3
    r = torch.linalg.qr(states.transpose(1, 2) / states.shape[-1]**.5,
                        mode='reduced').R
    index = torch.triu_indices(3, 3, device=states.device)
    return dict(numerator=torch.einsum('bkd,bopd->bkop',states,rows).flatten(2),
                norm_triangular=r[:, index[0], index[1]])


def fields_at(core, radius):
    """Three shared quadratic features keep the denominator a sum of squares."""
    import torch
    n = core['numerator']; b = len(n)
    t = torch.as_tensor(radius, dtype=n.dtype, device=n.device).reshape(-1).expand(b)
    square = t*t
    numerators = n[:,0] + t[:,None]*n[:,1] + square[:,None]*n[:,2]
    r = core['norm_triangular']
    u0 = r[:,0] + r[:,1]*t + r[:,2]*square
    u1 = r[:,3]*t + r[:,4]*square
    u2 = r[:,5]*square
    norm = u0.square()+u1.square()+u2.square()+torch.finfo(torch.float32).eps
    return torch.cat([numerators,norm[:,None]],dim=-1)


def execute(core, radius):
    return evaluate_fields(fields_at(core, radius))
