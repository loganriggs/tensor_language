"""Phase A: causally verify the register-leader story, replace it, red-team it.

The story under test (§17-§18, every clause so far correlational or exact-algebraic,
none yet causal): MLP1's dominant causal direction is a document-register signal --
head 4 of attn1 aggregates layout tokens in the recent context, MLP1 squares that, and
the result carries 39% of the layer's causal effect. Three causal claims follow, each
tested here with its own controls:

A1 SEMANTICS. If the leader is a register signal, deleting ONLY the leader (one
   direction of one layer, mean-ablated) must damage layout-heavy contexts and spare
   prose. Positions are binned by the layout-token fraction of their trailing 32
   tokens; the prediction is a monotone damage profile, and the reverse prediction
   (prose bins ~unharmed) is scored too. Control: the same test for a Shapley-matched
   non-register direction (new dir 1), which should NOT show the layout gradient.

A2 REPLACEMENT + MDL. The story says the leader's 664k-parameter quadratic form is
   really "square one projection of head 4's output". If true, swapping c_0(x) for a
   surrogate must hold CE near the intact model at a tiny parameter cost. The ladder,
   scored on (CE fidelity, parameter count):

       delete            c_0 -> mean                                0 params
       story surrogate   c_0 -> a*(u . xhat_attn1)^2 + b            1,154 params
       rank-2 whitened   c_0 -> 2-eigvec truncation of M            2,308 params
       full form         c_0 exact                                  ~664k params

   (u is fit as the top whitened eigenvector of M restricted to the attn1 component;
   a, b by least squares on 20k positions; all fitting on FIT rows only.)
   MDL verdict = fidelity per parameter, reported as the fraction of the deletion
   damage each rung repairs.

A3 RED TEAM.
   r1 matched-size null: same surrogate form, u random unit vector, a,b refit. If the
      story surrogate's win is generic to "any squared projection", this matches it.
   r2 document transfer: surrogate fit on rows 0-150, scored on rows 300-450 (the §16
      heterogeneity is the obvious failure mode; a register feature should transfer,
      a document-memorising fit should not).
   r3 TEXT-LEVEL INTERVENTION, the decisive one for the semantics: take prose
      sequences, inject layout tokens (newlines/#/<) at random interior positions,
      and measure the leader coefficient before and after ON THE SAME POSITIONS
      (the untouched suffix tokens). The story predicts the leader rises at positions
      whose context now contains layout. Reverse arm: strip layout tokens from
      markup-heavy sequences, leader must fall. Control: inject random non-layout
      tokens instead -- the leader should move far less.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import tiktoken
from bilin18_joint_removal import fwd, held, orth, m, FW, LAYER, DEV, PATCH, B0, HELD
from bilin18_identifiable import form_for_direction
from bilin18_source_folding import forward_tracked
from tier2_model import rope_tables, apply_rot

# --- a forward with a coefficient-replacement hook at LAYER ---
# (the first version of this script patched mlp1.forward and silently measured the
# intact model everywhere: joint_removal.fwd INLINES the MLP computation and never
# calls Module.forward. A1/A2/r1 from that run were void. This forward applies the
# hook to mo directly.)
COEFF_FN = None

NH_, HD_, D_ = 9, 128, 1152


@torch.no_grad()
def fwd2(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D_,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD_, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(len(m.transformer.h)):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (D_,))

        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH_, HD_), (HD_,))
            return apply_rot(z, cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH_, HD_)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k1_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k1_) / HD_
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD_
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        xhat = F.rms_norm(x, (D_,)); mlp = blk.mlp
        mo = mlp.Down(mlp.Left(xhat) * mlp.Right(xhat)) + mlp.Down_bias
        if li == LAYER and COEFF_FN is not None:
            mo = COEFF_FN(xhat, mo)
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D_,))) / 30)
    V_ = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1].reshape(-1, V_).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(B, T - 1)

enc = tiktoken.get_encoding('gpt2')
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_leader_verify_results.json')

# layout tokens: whitespace, markup, list/table markers (from §18's attribution list)
LAYOUT_STR = [' ', '\n', '\n\n', '\r', '\t', '#', '##', '###', '####', '<', ' <',
              '</', '>', '|', '||', '-', '--', '*', '=', '$', '%', '&', '_', '[',
              ']', '{', '}', '(', ')', '/']
LAYOUT = set()
for s_ in LAYOUT_STR:
    try:
        ids = enc.encode(s_)
        if len(ids) == 1:
            LAYOUT.add(ids[0])
    except Exception:
        pass


@torch.no_grad()
def collect_out(seqs):
    accs = []
    for i in range(0, seqs.shape[0], 6):
        acc = []
        fwd(seqs[i:i + 6].to(DEV), collect=LAYER, acc=acc)
        accs.append(acc[0])
    return torch.cat(accs, 0)


@torch.no_grad()
def held_per_pos():
    """per-position CE on the held set, (n_seq, T-1), honouring COEFF_FN."""
    outp = []
    for i in range(0, HELD.shape[0], B0):
        outp.append(fwd2(HELD[i:i + B0].to(DEV)))
    return torch.cat(outp, 0)


def layout_frac(seqs, window=32):
    """fraction of the trailing `window` tokens that are layout tokens, per position."""
    isl = torch.zeros_like(seqs, dtype=torch.float32)
    for t in LAYOUT:
        isl = isl + (seqs == t).float()
    csum = isl.cumsum(1)
    w = torch.clamp(torch.arange(seqs.shape[1], device=seqs.device)[None, :].float() + 1,
                    max=window)
    shifted = torch.zeros_like(csum)
    shifted[:, window:] = csum[:, :-window]
    return (csum - shifted) / w


def make_patch(d0, mode, u=None, a=1.0, b=0.0, M2=None, cbar=0.0):
    """Return a COEFF_FN replacing the write along d0 with a surrogate coefficient."""
    def hook(xhat, mo):
        c = mo.float() @ d0
        if mode == 'delete':
            chat = torch.full_like(c, cbar)
        elif mode == 'surrogate':
            chat = a * (xhat.float() @ u) ** 2 + b
        elif mode == 'rank2':
            xf = xhat.float()
            chat = torch.einsum('...i,ij,...j->...', xf, M2, xf) + b
        return mo + ((chat - c)[..., None] * d0).to(mo.dtype)
    return hook


def main():
    t0 = time.time()
    base = held(); BASE = float(base.mean())
    Y = collect_out(FW[0:300, :513])
    _, _, Vh = torch.linalg.svd((Y - Y.mean(0)).float(), full_matrices=False)
    Q = orth(Vh[:32].T)
    mlp1 = m.transformer.h[LAYER].mlp
    d0 = Q[:, 0].float()
    d1 = Q[:, 1].float()
    M = form_for_direction(mlp1, d0).float()
    out = {'base_ce': BASE}
    print(f'base CE {BASE:.4f}\n')

    def ce_with(hook):
        global COEFF_FN
        COEFF_FN = hook
        try:
            return held_per_pos()
        finally:
            COEFF_FN = None

    # ============ A1: damage by register bin ============
    print('== A1: does deleting ONLY the leader hurt layout-heavy contexts? ==')
    # collect inputs & c means on fit rows
    parts_l, Xh_l = [], []
    for i in range(0, 96, 6):
        p, xh, _ = forward_tracked(FW[i:i + 6, :513].to(DEV))
        parts_l.append(p['attn1']); Xh_l.append(xh)
    A1c = torch.cat(parts_l, 0); Xh = torch.cat(Xh_l, 0)
    c_fit = torch.einsum('ni,ij,nj->n', Xh, M, Xh)
    cbar = float(c_fit.mean())

    per_base = held_per_pos()
    frac = layout_frac(HELD[:, :-1])[:, : per_base.shape[1]]
    for tag, dvec in (('leader (register story)', d0),
                      ('control: direction 1', d1)):
        Mv = form_for_direction(mlp1, dvec).float()
        cb = float(torch.einsum('ni,ij,nj->n', Xh, Mv, Xh).mean())
        per = ce_with(make_patch(dvec, 'delete', cbar=cb))
        dmg = (per - per_base)
        bins = [0.0, 0.05, 0.15, 0.3, 1.01]
        row = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            k = (frac >= lo) & (frac < hi)
            row.append((float(dmg[k].mean()), int(k.sum())))
        out.setdefault('a1', {})[tag] = row
        print(f'  {tag:28s} damage by layout fraction of trailing context:')
        print('     ' + ' | '.join(f'[{lo:.2f}-{hi:.2f}) {v:+.4f} (n={n})'
                                   for (lo, hi), (v, n) in
                                   zip(zip(bins[:-1], bins[1:]), row)), flush=True)

    # ============ A2: the replacement ladder ============
    print('\n== A2: the MDL ladder ==')
    # story surrogate: u = top whitened eigvec of M restricted to attn1 comp space
    S = (A1c.T @ A1c / A1c.shape[0]).float()
    ev, U_ = torch.linalg.eigh(S.double())
    keepd = ev > 1e-8 * ev.max()
    Sh = (U_[:, keepd] * ev[keepd].sqrt()) @ U_[:, keepd].T
    Sih = (U_[:, keepd] * ev[keepd].rsqrt()) @ U_[:, keepd].T
    Mw = Sh @ M.double() @ Sh
    ew, Uw = torch.linalg.eigh(Mw)
    iw = ew.abs().argmax()
    u = (Sih @ Uw[:, iw]).float(); u = u / u.norm()
    proj2 = (Xh @ u) ** 2
    A_ = torch.stack([proj2, torch.ones_like(proj2)], 1)
    coef = torch.linalg.lstsq(A_, c_fit[:, None]).solution.squeeze()
    a_s, b_s = float(coef[0]), float(coef[1])
    r2_story = 1 - float(((a_s * proj2 + b_s - c_fit) ** 2).mean() / c_fit.var())
    # rank-2 whitened truncation of M
    idx2 = ew.abs().argsort(descending=True)[:2]
    M2 = (Sih @ (Uw[:, idx2] * ew[idx2]) @ Uw[:, idx2].T @ Sih).float()
    c2 = torch.einsum('ni,ij,nj->n', Xh, M2, Xh)
    b2 = float((c_fit - c2).mean())
    r2_rank2 = 1 - float(((c2 + b2 - c_fit) ** 2).mean() / c_fit.var())

    rungs = [('delete', make_patch(d0, 'delete', cbar=cbar), 0),
             ('story: a*(u.x)^2+b', make_patch(d0, 'surrogate', u=u,
                                               a=a_s, b=b_s), 1154),
             ('rank-2 whitened', make_patch(d0, 'rank2', M2=M2, b=b2), 2308)]
    ce_del = None
    for tag, patch, npar in rungs:
        ce = float(ce_with(patch).mean())
        if tag == 'delete':
            ce_del = ce
        rep = 1 - (ce - BASE) / max(ce_del - BASE, 1e-9)
        out.setdefault('a2', {})[tag] = {'ce': ce, 'params': npar,
                                         'repair_frac': rep}
        print(f'  {tag:22s} CE {ce:.4f} (+{ce-BASE:.4f})  params {npar:>6,}  '
              f'repairs {100*rep:5.1f}% of the deletion damage', flush=True)
    out['a2']['fit_r2'] = {'story': r2_story, 'rank2': r2_rank2}
    print(f'  (coefficient fit R^2: story {r2_story:.3f}, rank-2 {r2_rank2:.3f}; '
          f'full form = 664,128 params)')

    # ============ A3: red team ============
    print('\n== A3: red team ==')
    # r1 matched-size random-direction surrogate
    g = torch.Generator(device=DEV).manual_seed(0)
    ur = torch.randn(1152, device=DEV, generator=g); ur = ur / ur.norm()
    pr = (Xh @ ur) ** 2
    Ar = torch.stack([pr, torch.ones_like(pr)], 1)
    cr = torch.linalg.lstsq(Ar, c_fit[:, None]).solution.squeeze()
    ce_r1 = float(ce_with(make_patch(d0, 'surrogate', u=ur,
                                     a=float(cr[0]), b=float(cr[1]))).mean())
    rep_r1 = 1 - (ce_r1 - BASE) / max(ce_del - BASE, 1e-9)
    out['a3_r1'] = {'ce': ce_r1, 'repair_frac': rep_r1}
    print(f'  r1 random-u surrogate:   CE {ce_r1:.4f}, repairs {100*rep_r1:5.1f}% '
          f'(story repaired {100*out["a2"]["story: a*(u.x)^2+b"]["repair_frac"]:.1f}%)')

    # r2 document transfer of the story surrogate's coefficient fit
    parts2, Xh2 = [], []
    for i in range(300, 396, 6):
        p, xh, _ = forward_tracked(FW[i:i + 6, :513].to(DEV))
        Xh2.append(xh)
    Xh2 = torch.cat(Xh2, 0)
    c_far = torch.einsum('ni,ij,nj->n', Xh2, M, Xh2)
    pf = (Xh2 @ u) ** 2
    r2_far = 1 - float(((a_s * pf + b_s - c_far) ** 2).mean() / c_far.var())
    out['a3_r2'] = {'r2_fit_rows': r2_story, 'r2_transfer_rows': r2_far}
    print(f'  r2 document transfer:    R^2 {r2_story:.3f} on fit rows -> '
          f'{r2_far:.3f} on unseen rows')

    # r3 text-level intervention
    print('\n  r3 text intervention (the decisive semantic test):')
    nl = enc.encode('\n')[0]; hsh = enc.encode('#')[0]; lt = enc.encode('<')[0]
    inject = [nl, hsh, lt, nl]
    # pick prose-ish sequences (low layout frac) and markup-ish ones
    fr_all = layout_frac(FW[:200, :513]).mean(1)
    prose_idx = fr_all.argsort()[:12]
    markup_idx = fr_all.argsort(descending=True)[:12]

    def leader_at(seqs):
        cs = []
        for i in range(0, seqs.shape[0], 6):
            _, xh, _ = forward_tracked(seqs[i:i + 6].to(DEV))
            cs.append(torch.einsum('ni,ij,nj->n', xh, M, xh)
                      .view(min(6, seqs.shape[0] - i), -1))
        return torch.cat(cs, 0)

    gcpu = torch.Generator().manual_seed(1)
    res = {}
    for tag, rows, toks in (('inject layout into prose', prose_idx, inject),
                            ('inject control tokens into prose', prose_idx,
                             [enc.encode(' apple')[0], enc.encode(' seven')[0],
                              enc.encode(' blue')[0], enc.encode(' walked')[0]]),
                            ):
        seqs = FW[rows, :513].clone()
        pos = torch.randint(64, 400, (seqs.shape[0], 24), generator=gcpu)
        mod = seqs.clone()
        for r_ in range(seqs.shape[0]):
            for j, p_ in enumerate(pos[r_]):
                mod[r_, p_] = toks[j % len(toks)]
        c_b = leader_at(seqs); c_a = leader_at(mod)
        # score only untouched positions AFTER the first injection
        touched = torch.zeros_like(seqs, dtype=torch.bool)
        for r_ in range(seqs.shape[0]):
            touched[r_, pos[r_]] = True
        after = torch.zeros_like(touched)
        for r_ in range(seqs.shape[0]):
            after[r_, int(pos[r_].min()):] = True
        sel = (~touched & after)
        delta = float((c_a[sel] - c_b[sel]).mean())
        sd = float((c_a[sel] - c_b[sel]).std() / sel.sum().float().sqrt())
        res[tag] = {'delta': delta, 'se': sd}
        print(f'    {tag:36s} leader shift {delta:+.1f} +/- {sd:.1f}')
    # strip layout from markup docs
    seqs = FW[markup_idx, :513].clone()
    space = enc.encode(' the')[0]
    mod = seqs.clone()
    is_layout = torch.zeros_like(mod, dtype=torch.bool)
    for t in LAYOUT:
        is_layout |= mod == t
    mod[is_layout] = space
    c_b = leader_at(seqs); c_a = leader_at(mod)
    sel = ~is_layout
    delta = float((c_a[sel] - c_b[sel]).mean())
    sd = float((c_a[sel] - c_b[sel]).std() / sel.sum().float().sqrt())
    res['strip layout from markup'] = {'delta': delta, 'se': sd}
    print(f'    {"strip layout from markup docs":36s} leader shift {delta:+.1f} '
          f'+/- {sd:.1f}')
    out['a3_r3'] = res
    out['c_scale'] = {'std_fit_rows': float(c_fit.std()),
                      'mean_fit_rows': float(c_fit.mean())}
    print(f'    (scale: leader coefficient std on natural data '
          f'{float(c_fit.std()):.1f}, mean {float(c_fit.mean()):.1f})')

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
