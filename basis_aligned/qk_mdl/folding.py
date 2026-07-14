"""qk_mdl folding: exact vocab-space QK expansion for the tiny bilinear models.

Verified architecture (LOG tick 0): pre-RMSNorm (affine-free), rotate-half RoPE
on all d_head dims (planes pair dim f with dim f + d_head/2, frequency
omega_f = base^(-2f/d_head)), NO softmax: per head,

    pattern = s1 * s2 / d_head^2 * causal_mask
    s_b(t_q @ i, t_k @ j) = sum_f cos(omega_f * (i-j)) * C^b_f[t_q, t_k]
                                + sin(omega_f * (i-j)) * S^b_f[t_q, t_k]

with, per plane f (a = dims [:F], b = dims [F:], F = d_head/2):

    C^b_f[t_q, t_k] = qa_f(t_q) ka_f(t_k) + qb_f(t_q) kb_f(t_k)      (rank <= 2)
    S^b_f[t_q, t_k] = qa_f(t_q) kb_f(t_k) - qb_f(t_q) ka_f(t_k)      (rank <= 2)

where q(t) = head-slice of (e_hat_t @ W_qb^T), e_hat = RMSNorm(embedding row).
This is EXACT for layer 0 (its input is the raw embedding). delta convention:
delta = i_query - j_key (>= 0 under the causal mask).
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/workspace/tensor_language')
from deep_model import DeepModel  # noqa: E402

RUNS = Path('/workspace/tensor_language/runs_owt')


def load_tiny(name, dtype=torch.float64, device='cpu'):
    cfg = json.load(open(RUNS / name / 'config.json'))
    model = DeepModel(n_vocab=cfg['vocab'], d_model=cfg['d_model'],
                      n_head=cfg['n_head'], spec=cfg['spec'], n_ctx=cfg['n_ctx'],
                      norm=cfg.get('norm', False),
                      residual=cfg.get('residual', 'lerp'),
                      attention=cfg.get('attention', 'bilinear'))
    sd = torch.load(RUNS / name / 'model.pt', map_location='cpu')
    if not isinstance(sd, dict) or 'embed.weight' not in sd:
        sd = sd.state_dict() if hasattr(sd, 'state_dict') else sd['model']
    model.load_state_dict(sd)
    return model.to(dtype=dtype, device=device).eval(), cfg


@torch.no_grad()
def effective_embedding(model, layer_idx=0):
    """e_hat rows: the layer's own norm module applied to the embedding (exact)."""
    return model.layers[layer_idx].norm(model.embed.weight.detach())


@torch.no_grad()
def branch_factors(model, layer_idx, branch):
    """Per-token RoPE-plane factors. Returns qa, qb, ka, kb: (V, n_head, F)."""
    layer = model.layers[layer_idx]
    Eh = effective_embedding(model, layer_idx)
    Wq = getattr(layer, f'q{branch}').weight.detach()
    Wk = getattr(layer, f'k{branch}').weight.detach()
    V = Eh.shape[0]
    nh, dh = layer.n_head, layer.d_head
    F = dh // 2
    Q = (Eh @ Wq.T).reshape(V, nh, dh)
    K = (Eh @ Wk.T).reshape(V, nh, dh)
    return Q[..., :F], Q[..., F:], K[..., :F], K[..., F:]


def omegas(d_head, dtype=torch.float64, base=10_000):
    return 1.0 / base ** (torch.arange(0, d_head, 2, dtype=dtype) / d_head)


@torch.no_grad()
def expanded_branch_scores(model, layer_idx, branch, tokens, use_model_trig=True):
    """Branch scores (B, n_head, T, T) from the {C_f, S_f} expansion (unmasked).

    use_model_trig=True builds cos(w*delta), sin(w*delta) from the model's own
    cached rotary tables via the exact difference identities — this matches the
    DEPLOYED model bit-for-bit-in-algebra (the source Rotary computes its tables
    in fp32, so analytic fp64 trig differs from the checkpointed model by ~1e-7
    per entry; see LOG tick 1). use_model_trig=False uses analytic fp64 omegas
    (the right choice for downstream folded objects, stated deviation ~1e-4 on
    scores).
    """
    layer = model.layers[layer_idx]
    dh = layer.d_head
    F = dh // 2
    dtype = model.embed.weight.dtype
    T = tokens.shape[-1]
    if use_model_trig:
        cs = layer.rotary.cos_cached[0, :T, 0, :F].to(dtype)   # (T, F)
        sn = layer.rotary.sin_cached[0, :T, 0, :F].to(dtype)
        cosD = torch.einsum('if,jf->ijf', cs, cs) + torch.einsum('if,jf->ijf', sn, sn)
        sinD = torch.einsum('if,jf->ijf', sn, cs) - torch.einsum('if,jf->ijf', cs, sn)
    else:
        om = omegas(dh, dtype).to(tokens.device)
        pos = torch.arange(T, dtype=dtype, device=tokens.device)
        delta = pos[:, None] - pos[None, :]                    # i - j
        cosD = torch.cos(delta[..., None] * om)                # (T, T, F)
        sinD = torch.sin(delta[..., None] * om)
    qa, qb, ka, kb = branch_factors(model, layer_idx, branch)
    QA, QB, KA, KB = (M[tokens] for M in (qa, qb, ka, kb))    # (B, T, nh, F)
    s = (torch.einsum('bihf,bjhf,ijf->bhij', QA, KA, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', QB, KB, cosD)
         + torch.einsum('bihf,bjhf,ijf->bhij', QA, KB, sinD)
         - torch.einsum('bihf,bjhf,ijf->bhij', QB, KA, sinD))
    return s


@torch.no_grad()
def expanded_pattern(model, layer_idx, tokens):
    """Full head pattern (B, n_head, T, T) = s1*s2/d_head^2 * causal mask."""
    layer = model.layers[layer_idx]
    T = tokens.shape[-1]
    s1 = expanded_branch_scores(model, layer_idx, 1, tokens)
    s2 = expanded_branch_scores(model, layer_idx, 2, tokens)
    mask = torch.tril(torch.ones(T, T, dtype=s1.dtype, device=s1.device))
    return s1 * s2 / layer.d_head ** 2 * mask


@torch.no_grad()
def band_mass(model, layer_idx, branch):
    """Descriptive frequency profile: per head, Frobenius mass of C_f, S_f per
    band, computed WITHOUT materializing V x V (rank-2 identities)."""
    qa, qb, ka, kb = branch_factors(model, layer_idx, branch)

    def rank2_fro2(u1, v1, u2, v2, sign=1.0):
        # || u1 v1^T + sign * u2 v2^T ||_F^2, batched over (n_head, F)
        n = lambda M: (M ** 2).sum(0)                     # (nh, F)
        dot = lambda A, B: (A * B).sum(0)
        return n(u1) * n(v1) + n(u2) * n(v2) + 2 * sign * dot(u1, u2) * dot(v1, v2)

    C2 = rank2_fro2(qa, ka, qb, kb, +1.0)
    S2 = rank2_fro2(qa, kb, qb, ka, -1.0)
    return C2, S2   # each (n_head, F)
