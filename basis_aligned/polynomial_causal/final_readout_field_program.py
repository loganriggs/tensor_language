"""Shared RMS denominator and explicit softcap over token numerator fields."""


def pack_fields(state, rows):
    """state[B,D], unembedding rows[B,O,2,D] -> fields[B,2*O+1]."""
    import torch
    numerators = (state[:, None, None, :] * rows).sum(-1).flatten(1)
    square_mean = state.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps
    return torch.cat([numerators, square_mean], dim=-1)


def evaluate_fields(fields):
    """The last field is the shared squared RMS denominator, including epsilon."""
    import torch
    assert fields.shape[-1] >= 3 and fields.shape[-1] % 2 == 1
    denominator = fields[..., -1:]
    if not bool((denominator > 0).all()):
        raise ValueError('Quadratic denominator is nonpositive; no clamped fallback')
    scores = 30 * torch.tanh(fields[..., :-1] / (30 * denominator.sqrt()))
    scores = scores.reshape(*scores.shape[:-1], -1, 2)
    return scores[..., 0] - scores[..., 1]


def evaluate_jet(jet, radius):
    """jet[B,3,F] stores coefficients of 1,t,t² (not raw derivatives)."""
    return evaluate_fields(jet[:, 0] + radius * jet[:, 1] + radius**2 * jet[:, 2])
