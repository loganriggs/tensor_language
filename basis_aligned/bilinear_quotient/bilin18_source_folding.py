"""Fold the upstream writers into MLP1: what is actually driving each causal direction?

The residual stream at MLP1's input is an EXACT sum of four writers -- there is no
approximation in this decomposition, which is the payoff of the architecture being
bilinear:

    x = E + A0 + M0 + A1
        E   the embedding path (rms-normed wte, carried and re-injected through the
            per-block lambdas)
        A0  attention block 0's output
        M0  MLP block 0's output
        A1  attention block 1's output (note: its value stream mixes in block 0's
            values via the model's lamb parameter, so A1 carries some block-0 content)

The MLP input is xhat = x / r (per-position rms scalar), so each writer's share of xhat
is exact too. And because MLP1's coefficient along an output direction d is a quadratic
form, it splits exactly into writer PAIRS:

    c_d = xhat^T M_d xhat = sum_{a<=b} (2 - delta_ab) * xhat_a^T M_d xhat_b

Ten terms, summing to c_d identically (gate: reconstruction error reported). The share
of Var(c_d) credited to each pair is Cov(term, c_d)/Var(c_d), which also sums to 1.

This answers "what writes to the causal directions" mechanically: an E x E dominated
direction is a current-token (bigram) feature; E x A1 is a token-in-context
conjunction; A0/A1 x themselves is pure context. And for the E x E part there is a free
bonus: it is computable for EVERY vocabulary token directly from the weights,
    s_t = rmsnorm(wte_t)^T M_d rmsnorm(wte_t),
giving a full-vocab naming of the embedding-driven component -- the "semantic
hypothesis" the folding is for.

Directions analysed: the two sample-size-stable Shapley leaders of the refit basis
(§16: new directions 0 and 1, cos 0.985/0.946 to the old basis), plus new direction 5.
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
SRC = ('emb', 'attn0', 'mlp0', 'attn1')
enc = tiktoken.get_encoding('gpt2')
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_source_folding_results.json')


@torch.no_grad()
def forward_tracked(idx):
    """Reproduce the model's forward to MLP1's input, carrying the four writers'
    components exactly. Returns components of xhat (post-rms), plus xhat for the gate."""
    B, T = idx.shape
    x0 = F.rms_norm(m.transformer.wte(idx), (D,))
    comp = {'emb': x0.clone(), 'attn0': None, 'mlp0': None, 'attn1': None}
    x = x0
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    v1 = None
    for li in (0, 1):
        blk = m.transformer.h[li]
        # lambda re-mix: scales every existing component, re-injects x0 into emb
        for k in comp:
            if comp[k] is not None:
                comp[k] = blk.lambdas[0] * comp[k]
        comp['emb'] = comp['emb'] + blk.lambdas[1] * x0
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
        ao = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        comp[f'attn{li}'] = ao
        x = x + ao
        if li == 0:
            xhat = F.rms_norm(x, (D,))
            mlp = blk.mlp
            mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
            comp['mlp0'] = mo
            x = x + mo
    # MLP1's input
    xhat = F.rms_norm(x, (D,))
    r = x.norm(dim=-1, keepdim=True) / D ** 0.5      # rms_norm divisor
    parts = {k: (comp[k] / r).reshape(-1, D).float() for k in SRC}
    gate = (sum(parts.values()) - xhat.reshape(-1, D).float()).norm() / \
        xhat.reshape(-1, D).norm()
    return parts, xhat.reshape(-1, D).float(), float(gate)


@torch.no_grad()
def collect_out(seqs):
    accs = []
    for i in range(0, seqs.shape[0], 6):
        acc = []
        fwd(seqs[i:i + 6].to(DEV), collect=LAYER, acc=acc)
        accs.append(acc[0])
    return torch.cat(accs, 0)


def main():
    t0 = time.time()
    # the refit (154k) basis of §16
    Y = collect_out(FW[0:300, :513])
    _, _, Vh = torch.linalg.svd((Y - Y.mean(0)).float(), full_matrices=False)
    Q = orth(Vh[:32].T)
    mlp1 = m.transformer.h[LAYER].mlp

    # sources on a subset (memory: 4 tensors of n x 1152)
    parts_l, gates = [], []
    Xh = []
    for i in range(0, 120, 6):
        p, xh, gt = forward_tracked(FW[i:i + 6, :513].to(DEV))
        parts_l.append(p); gates.append(gt); Xh.append(xh)
    parts = {k: torch.cat([p[k] for p in parts_l], 0) for k in SRC}
    Xh = torch.cat(Xh, 0)
    print(f'{Xh.shape[0]:,} positions | writer-sum reconstruction gate: '
          f'{max(gates):.2e} (exact decomposition)\n')

    wte = m.transformer.wte.weight.detach().float()
    wte_n = F.rms_norm(wte, (D,))
    out = {'gate_max': max(gates), 'directions': []}
    for di in (0, 1, 5):
        d = Q[:, di].float()
        M = form_for_direction(mlp1, d / d.norm()).float()
        c = torch.einsum('ni,ij,nj->n', Xh, M, Xh)
        var_c = float(c.var())
        rows = []
        recon = torch.zeros_like(c)
        for a in range(4):
            for b in range(a, 4):
                Ta = parts[SRC[a]]; Tb = parts[SRC[b]]
                t = torch.einsum('ni,ij,nj->n', Ta, M, Tb)
                if b != a:
                    t = t + torch.einsum('ni,ij,nj->n', Tb, M, Ta)
                recon = recon + t
                cov = float(((t - t.mean()) * (c - c.mean())).mean())
                rows.append({'pair': f'{SRC[a]}x{SRC[b]}',
                             'var_share': cov / max(var_c, 1e-30),
                             'mean': float(t.mean())})
        gate2 = float((recon - c).abs().mean() / c.abs().mean().clamp_min(1e-30))
        rows.sort(key=lambda r: -abs(r['var_share']))
        # full-vocab naming of the E x E component
        s_t = torch.einsum('vi,ij,vj->v', wte_n, M, wte_n)
        top = [enc.decode([t]) for t in s_t.argsort(descending=True)[:10].tolist()]
        bot = [enc.decode([t]) for t in s_t.argsort()[:10].tolist()]
        rec = {'direction': di, 'pair_var_shares': rows, 'pair_gate': gate2,
               'emb_curvature_top': top, 'emb_curvature_bottom': bot}
        out['directions'].append(rec)
        print(f'== new direction {di} ==   (pair-sum gate {gate2:.1e})')
        for r in rows[:5]:
            print(f'   {r["pair"]:>12}: {100*r["var_share"]:+6.1f}% of variance')
        print(f'   emb-only curvature, most positive: {top}')
        print(f'   emb-only curvature, most negative: {bot}\n', flush=True)

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
