"""Unfold the register leader through attn1: which heads, which attended tokens,
which offsets?

§17 established that MLP1's causal leader (39% of the layer) is 76% attn1's output
interacting with itself: attn1 aggregates context, MLP1 squares it, and the result is a
document-register signal. That names the WRITER. This names what the writer reads,
staying exact at every step:

  LEVEL 1 -- heads. attn1's output is a sum over 9 heads, A1 = sum_h A1_h (exact:
  c_proj is linear, so each head's context vector maps through its own slice of the
  projection). The attn1 x attn1 term therefore splits into an exact 9x9 head-pair
  grid, T_hh' = A1_h^T M A1_h' / r^2, and variance shares of the leader coefficient
  attribute the register signal to specific head pairs.

  LEVEL 2 -- attended tokens. Within the quadratic, each KEY position k contributes
  exactly g_qk = x_qk^T M a_q, where x_qk = sum_h pat_h[q,k] y_hk / r_q is what the
  attention actually transported from key k, and a_q is the full attn1 component. These
  attributions sum to the attn1 x attn1 term identically (gate reported). Accumulated
  by the key's TOKEN ID they answer "attending to which tokens drives the register
  feature"; accumulated by OFFSET q-k they answer "how local is it".

The point of the exercise: §15's token-list naming failed on this direction because the
meaning is not in the current token. If the register story is right, the key-token
attribution should be dominated by whitespace/structural tokens at many offsets --
i.e., 'the context is full of layout characters', measured rather than inferred.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_joint_removal import fwd, orth, m, FW, LAYER, DEV
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot

NH, HD, D = 9, 128, 1152
enc = tiktoken.get_encoding('gpt2')
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_leader_unfold_results.json')


@torch.no_grad()
def collect_out(seqs):
    accs = []
    for i in range(0, seqs.shape[0], 6):
        acc = []
        fwd(seqs[i:i + 6].to(DEV), collect=LAYER, acc=acc)
        accs.append(acc[0])
    return torch.cat(accs, 0)


@torch.no_grad()
def run_batch(idx, M):
    """One batch: per-head A1 components, per-key attributions, exact gates."""
    B, T = idx.shape
    x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    x = x0
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    v1 = None
    pat1 = v_1 = None
    for li in (0, 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (D,))

        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        ctx = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        ao = a.c_proj(ctx.reshape(B, T, -1))
        if li == 1:
            pat1, v_1 = pat, v
            ctx1 = ctx
        x = x + ao
        if li == 0:
            xhat = F.rms_norm(x, (D,))
            mlp = blk.mlp
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
    r = (x.norm(dim=-1, keepdim=True) / D ** 0.5)          # (B,T,1)
    Wp = m.transformer.h[1].attn.c_proj.weight.detach()     # (D, D)

    # per-head A1 components (post-projection), divided by the rms scalar
    A1h = torch.einsum('bqhd,ehd->bqhe',
                       ctx1, Wp.view(D, NH, HD).to(ctx1.dtype)) / r[..., None].permute(0, 1, 3, 2)
    A1h = torch.einsum('bqhd,ehd->bqeh', ctx1,
                       Wp.view(D, NH, HD).to(ctx1.dtype))
    A1h = A1h / r[..., None]                                # (B,T,D,NH)
    a1 = A1h.sum(-1)                                        # (B,T,D) full attn1 comp

    Mf = M.to(a1.dtype)
    u = torch.einsum('bqd,de->bqe', a1, Mf)                 # M a_q
    # level 1: head-pair terms
    hm = torch.einsum('bqdh,bqd->bqh', A1h, u)              # A1_h^T M a  (sums over h' inside u? no)
    # careful: T_hh' needs pairwise; compute z_h = A1_h^T M A1_h' via projections
    Zh = torch.einsum('bqdh,de->bqeh', A1h, Mf)             # (B,T,D,NH) = M^T A1_h
    Thh = torch.einsum('bqdh,bqdg->bqhg', A1h, Zh)          # (B,T,NH,NH) A1_h^T M A1_g
    c_attn = torch.einsum('bqd,bqd->bq', a1, u)             # a^T M a
    gate1 = float((Thh.sum((-1, -2)) - c_attn).abs().mean() /
                  c_attn.abs().mean().clamp_min(1e-30))

    # level 2: per-key attribution g[q,k] = sum_h pat_h[q,k] * (y_hk . u_q) / r_q
    yh = torch.einsum('bkhd,ehd->bkeh', v_1, Wp.view(D, NH, HD).to(v_1.dtype))
    G = torch.einsum('bkeh,bqe->bhqk', yh, u)               # y_hk . u_q
    g = (pat1 * G).sum(1) / r.squeeze(-1)[:, :, None]       # (B,T_q,T_k)
    gate2 = float((g.sum(-1) - c_attn).abs().mean() /
                  c_attn.abs().mean().clamp_min(1e-30))
    return Thh.float(), c_attn.float(), g.float(), gate1, gate2


def main():
    t0 = time.time()
    Y = collect_out(FW[0:300, :513])
    _, _, Vh = torch.linalg.svd((Y - Y.mean(0)).float(), full_matrices=False)
    Q = orth(Vh[:32].T)
    d = Q[:, 0].float()
    M = form_for_direction(m.transformer.h[LAYER].mlp, d / d.norm()).float()

    V = 50257
    tok_g = torch.zeros(V, dtype=torch.float64, device=DEV)
    tok_n = torch.zeros(V, dtype=torch.float64, device=DEV)
    off_g = torch.zeros(513, dtype=torch.float64, device=DEV)
    Th_cov = None
    c_all, Th_all = [], []
    gates = []
    n_seq = 96
    for i in range(0, n_seq, 6):
        idx = FW[i:i + 6, :513].to(DEV)
        Thh, c_attn, g, g1, g2 = run_batch(idx, M)
        gates.append((g1, g2))
        B, T = idx.shape
        # accumulate per-key-token and per-offset attributions
        gk = g.sum(1)                                       # (B,T_k) total per key
        tok_g.index_add_(0, idx.reshape(-1), gk.reshape(-1).double())
        tok_n.index_add_(0, idx.reshape(-1), torch.ones_like(gk.reshape(-1),
                                                             dtype=torch.float64))
        for off in range(0, T):
            dg = torch.diagonal(g, offset=-off, dim1=1, dim2=2)
            off_g[off] += float(dg.double().sum())
        c_all.append(c_attn.reshape(-1))
        Th_all.append(Thh.reshape(-1, NH, NH))
    g1s, g2s = max(g[0] for g in gates), max(g[1] for g in gates)
    print(f'{n_seq} sequences | exactness gates: head-pairs {g1s:.1e}, '
          f'per-key {g2s:.1e}\n')

    c = torch.cat(c_all); Th = torch.cat(Th_all, 0)
    var_c = float(c.var())
    covs = torch.zeros(NH, NH)
    cm = c - c.mean()
    for h in range(NH):
        for g_ in range(NH):
            t = Th[:, h, g_]
            covs[h, g_] = float(((t - t.mean()) * cm).mean()) / max(var_c, 1e-30)
    sym = covs + covs.T - torch.diag(covs.diag())
    flat = [(h, g_, float(sym[h, g_])) for h in range(NH) for g_ in range(h, NH)]
    flat.sort(key=lambda t: -abs(t[2]))
    out = {'gates': {'head_pairs': g1s, 'per_key': g2s},
           'head_pairs_top': [{'heads': [h, g_], 'var_share': s}
                              for h, g_, s in flat[:6]]}
    print('== level 1: which head pairs carry the register signal ==')
    for h, g_, s in flat[:6]:
        tag = f'head {h} x head {g_}' if h != g_ else f'head {h} squared'
        print(f'   {tag:22s} {100*s:+6.1f}% of leader variance')

    print('\n== level 2: attending to which tokens drives the leader ==')
    keep = tok_n >= 50
    mean_g = torch.where(keep, tok_g / tok_n.clamp_min(1), torch.zeros_like(tok_g))
    top = mean_g.argsort(descending=True)[:14]
    bot = mean_g.argsort()[:8]
    out['key_tokens_top'] = [{'tok': enc.decode([int(t)]),
                              'mean_g': float(mean_g[t]),
                              'n': int(tok_n[t])} for t in top]
    out['key_tokens_bottom'] = [{'tok': enc.decode([int(t)]),
                                 'mean_g': float(mean_g[t])} for t in bot]
    print('   strongest positive (attending here pushes the register signal up):')
    print('   ' + str([enc.decode([int(t)]) for t in top]))
    print('   strongest negative:')
    print('   ' + str([enc.decode([int(t)]) for t in bot]))

    tot = float(off_g.abs().sum())
    cum = off_g.abs().cumsum(0) / max(tot, 1e-30)
    half = int((cum < 0.5).sum()) + 1
    out['offset'] = {'half_mass_within': half,
                     'share_offset_0_4': float(off_g.abs()[:5].sum() / tot),
                     'share_beyond_64': float(off_g.abs()[64:].sum() / tot)}
    print(f'\n== level 2b: how far back does it look ==')
    print(f'   half the attribution mass lies within {half} tokens; '
          f'offsets 0-4 carry {100*out["offset"]["share_offset_0_4"]:.0f}%, '
          f'beyond 64 carries {100*out["offset"]["share_beyond_64"]:.0f}%')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
