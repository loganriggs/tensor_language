"""Exact local real-arithmetic signature; no semantic or minimality claim."""
import torch

EPS = torch.finfo(torch.float32).eps


def rotary(position, width):
    """Native half-split convention, including BF16 table rounding."""
    inv = 1.0 / (10000 ** (torch.arange(0, width, 2).float() / width))
    angle = (torch.tensor(float(position), dtype=torch.float32) * inv)
    c, s = angle.cos().bfloat16().double(), angle.sin().bfloat16().double()
    return torch.cat((torch.cat((c.diag(), s.diag()), 1),
                      torch.cat((-s.diag(), c.diag()), 1)), 0)


def fold(weights, t, s):
    """Weights ordered Q1,K1,Q2,K2, each [head_width,residual_width]."""
    d = weights[0].shape[0]
    rt, rs = rotary(t, d), rotary(s, d)
    result = []
    for q, k in (weights[:2], weights[2:]):
        result.append((q.T @ rt.T @ rs @ k / d, q.T @ q / d, k.T @ k / d))
    return result


def evaluate(signature, x, y):
    numerator = torch.ones(x.shape[:-1], dtype=x.dtype)
    denominator_squared = torch.ones_like(numerator)
    for a, gq, gk in signature:
        numerator = numerator * ((x @ a) * y).sum(-1)
        denominator_squared = denominator_squared * (((x @ gq) * x).sum(-1) + EPS)
        denominator_squared = denominator_squared * (((y @ gk) * y).sum(-1) + EPS)
    if not bool((denominator_squared > 0).all()):
        raise ValueError('Nonpositive norm product: numerical signature evaluation invalid')
    return numerator / denominator_squared.sqrt(), numerator


def direct(weights, x, y, t, s):
    """Independent projection -> head RMSNorm -> rotary -> score evaluation."""
    import torch.nn.functional as F
    d = weights[0].shape[0]
    rt, rs = rotary(t, d).to(x.dtype), rotary(s, d).to(x.dtype)
    scores = []
    for q, k in (weights[:2], weights[2:]):
        qx = F.rms_norm(x @ q.T, (d,), eps=EPS) @ rt.T
        ky = F.rms_norm(y @ k.T, (d,), eps=EPS) @ rs.T
        scores.append((qx * ky).sum(-1) / d)
    return scores[0] * scores[1]


def paired_change(weights, scale):
    q, k, q2, k2 = weights
    return [scale[:, None] * q, k / scale[:, None], q2, k2]
