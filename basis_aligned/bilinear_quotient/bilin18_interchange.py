"""Phase A': interchange interventions on layer 1's variable graph.

The verified surrogate (§19) implies a causal abstraction in the Geiger sense — a
high-level model with named variables and computations between them:

    z  := u . xhat            [scalar; computed, per §18, mostly by attn1 head 4]
    c0 := a z^2 + b           [the leader coefficient]
    out := c0 * d0 + rest     [what downstream layers read]

§19 tested only the OUTPUT of this graph (swap c0's realizer, measure CE). An
abstraction claim is stronger: each EDGE must be individually intervenable, and
interventions on the high-level variable must produce the same downstream behaviour as
interventions on its low-level realizer. Two edges, two tests:

E1 — the z -> c0 edge (interchange faithfulness).
    Pair each base sequence with a source sequence. At every position, either
      c-patch:  replace c0(base) by c0(source)          [low-level ground truth]
      z-patch:  replace c0(base) by a*z(source)^2 + b   [what the abstraction says]
    If z abstracts c0 faithfully, the two patched models must agree DOWNSTREAM, not
    just on the coefficient: KL(P_cpatch || P_zpatch) << KL(P_base || P_cpatch), and
    the top-1 predictions after the two patches should agree wherever the patch
    changed anything. Controls: a position-shuffled c-patch (breaks the pairing) and
    the same machinery on a random direction (calibrates the KL scale).

E2 — the head4 -> z edge (which upstream part moves the variable).
    Replace head h's attn1 context (pre-projection) with the source's, one head at a
    time, and measure how much z moves. §18's exact attribution says head 4 carries
    90% on-distribution; the interchange version asks whether MOVING head 4 MOVES z --
    an intervention, not a decomposition, and §19's lesson is that these can disagree.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV, LAYER
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot

NH, HD, D = 9, 128, 1152
N_PAIR = 24
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_interchange_results.json')

COEFF_FN = None


@torch.no_grad()
def collect_out(seqs):
    accs = []
    for i in range(0, seqs.shape[0], 6):
        acc = []
        fwd(seqs[i:i + 6].to(DEV), collect=LAYER, acc=acc)
        accs.append(acc[0])
    return torch.cat(accs, 0)


@torch.no_grad()
def fwd_logits(idx, head_ctx_patch=None):
    """Forward returning log-probs; COEFF_FN hooks MLP-LAYER's output; optionally
    replaces attn1 head contexts (dict head -> (B,T,HD) tensor) before c_proj."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    xhat1 = None
    for li in range(len(m.transformer.h)):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
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
        if li == 1 and head_ctx_patch is not None:
            for h, rep in head_ctx_patch.items():
                ctx[:, :, h, :] = rep
        x = x + a.c_proj(ctx.reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        if li == 1:
            xhat1 = xhat
        mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        if li == LAYER and COEFF_FN is not None:
            mo = COEFF_FN(xhat, mo)
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    return F.log_softmax(logits.float(), dim=-1), xhat1


@torch.no_grad()
def ctx1_of(idx):
    """attn1 per-head contexts (pre-projection) and MLP1 input for a batch."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in (0, 1):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
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
        if li == 1:
            ctx1 = ctx
        x = x + a.c_proj(ctx.reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,)); mlp = blk.mlp
        if li == 0:
            x = x + mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
    xh1 = F.rms_norm(x + 0 * x, (D,))
    return ctx1, xh1


def main():
    t0 = time.time()
    # rebuild the surrogate objects exactly as §19 did
    Y = collect_out(FW[0:300, :513])
    _, _, Vh = torch.linalg.svd((Y - Y.mean(0)).float(), full_matrices=False)
    Q = orth(Vh[:32].T)
    d0 = Q[:, 0].float()
    mlp1 = m.transformer.h[LAYER].mlp
    M = form_for_direction(mlp1, d0).float()

    from bilin18_source_folding import forward_tracked
    parts_l, Xh_l = [], []
    for i in range(0, 96, 6):
        p, xh, _ = forward_tracked(FW[i:i + 6, :513].to(DEV))
        parts_l.append(p['attn1']); Xh_l.append(xh)
    A1c = torch.cat(parts_l); Xh = torch.cat(Xh_l)
    S = (A1c.T @ A1c / A1c.shape[0]).double()
    ev, U_ = torch.linalg.eigh(S)
    kd = ev > 1e-8 * ev.max()
    Sh = (U_[:, kd] * ev[kd].sqrt()) @ U_[:, kd].T
    Sih = (U_[:, kd] * ev[kd].rsqrt()) @ U_[:, kd].T
    Mw = Sh @ M.double() @ Sh
    ew, Uw = torch.linalg.eigh(Mw)
    u = (Sih @ Uw[:, ew.abs().argmax()]).float(); u = u / u.norm()
    c_fit = torch.einsum('ni,ij,nj->n', Xh, M, Xh)
    p2 = (Xh @ u) ** 2
    co = torch.linalg.lstsq(torch.stack([p2, torch.ones_like(p2)], 1),
                            c_fit[:, None]).solution.squeeze()
    a_s, b_s = float(co[0]), float(co[1])
    out = {'surrogate': {'a': a_s, 'b': b_s}}

    base_rows = FW[300:300 + N_PAIR, :257].to(DEV)
    src_rows = FW[400:400 + N_PAIR, :257].to(DEV)

    # source values of the variables at every position
    ctx_s, xh_s = ctx1_of(src_rows)
    c_src = torch.einsum('bti,ij,btj->bt', xh_s.float(), M, xh_s.float())
    z_src = (xh_s.float() @ u)

    def mk_hook(cvals):
        def hook(xhat, mo):
            c = mo.float() @ d0
            return mo + ((cvals - c)[..., None] * d0).to(mo.dtype)
        return hook

    global COEFF_FN
    print('== E1: is z a faithful abstraction of c0? (interchange on the z->c0 edge) ==')
    lp_base, _ = fwd_logits(base_rows)
    COEFF_FN = mk_hook(c_src)
    lp_c, _ = fwd_logits(base_rows)
    COEFF_FN = mk_hook(a_s * z_src ** 2 + b_s)
    lp_z, _ = fwd_logits(base_rows)
    perm = torch.randperm(c_src.shape[1])
    COEFF_FN = mk_hook(c_src[:, perm])
    lp_shuf, _ = fwd_logits(base_rows)
    COEFF_FN = None

    def kl(lp1, lp2):
        return (lp1.exp() * (lp1 - lp2)).sum(-1)

    effect = kl(lp_c, lp_base)
    mism_z = kl(lp_c, lp_z)
    mism_s = kl(lp_c, lp_shuf)
    sel = effect > effect.flatten().kthvalue(int(0.5 * effect.numel())).values
    faith = 1 - float(mism_z[sel].mean() / effect[sel].mean())
    faith_s = 1 - float(mism_s[sel].mean() / effect[sel].mean())
    t1_c = lp_c.argmax(-1); t1_z = lp_z.argmax(-1); t1_b = lp_base.argmax(-1)
    changed = t1_c != t1_b
    agree = float((t1_z[changed] == t1_c[changed]).float().mean()) \
        if changed.any() else float('nan')
    out['e1'] = {'effect_kl_mean': float(effect[sel].mean()),
                 'mismatch_z_kl_mean': float(mism_z[sel].mean()),
                 'mismatch_shuffle_kl_mean': float(mism_s[sel].mean()),
                 'faithfulness': faith, 'faithfulness_shuffle_control': faith_s,
                 'n_top1_changed': int(changed.sum()),
                 'top1_agreement_on_changed': agree}
    print(f'  effect size of the c-patch:      KL {float(effect[sel].mean()):.4f} '
          f'(top-half positions)')
    print(f'  z-patch mismatch vs c-patch:     KL {float(mism_z[sel].mean()):.4f} '
          f'-> faithfulness {100*faith:.1f}%')
    print(f'  shuffled-c control:              KL {float(mism_s[sel].mean()):.4f} '
          f'-> faithfulness {100*faith_s:.1f}%')
    print(f'  top-1 prediction changed by c-patch at {int(changed.sum())} positions; '
          f'z-patch reproduces {100*agree:.1f}% of those changes', flush=True)

    # ============ E2: which head moves z? ============
    print('\n== E2: which attn1 head MOVES z? (interchange on the head->z edge) ==')
    ctx_b, xh_b = ctx1_of(base_rows)
    z_base = (xh_b.float() @ u)
    moves = []
    for h in range(NH):
        _, xh_p = (None, None)
        # rerun with head h's context replaced
        lp_dummy, xh_p = fwd_logits(base_rows,
                                    head_ctx_patch={h: ctx_s[:, :, h, :]})
        z_p = (xh_p.float() @ u)
        moves.append(float(((z_p - z_base) ** 2).mean()))
    tot = sum(moves)
    out['e2'] = {'z_move_per_head': moves,
                 'head_shares': [round(v / tot, 3) for v in moves]}
    for h in range(NH):
        bar = '#' * int(40 * moves[h] / max(moves))
        print(f'  head {h}: {100*moves[h]/tot:5.1f}% of z-movement  {bar}')
    print(f'  (§18 attribution said head 4 carries 90% on-distribution)')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
