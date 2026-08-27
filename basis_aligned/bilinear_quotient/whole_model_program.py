# whole_model_program: BOTH HALVES AT ONCE — the first whole-model program in this arc
#
# §1664-§1692 priced the two halves of bilin18 separately and never together:
#   18 MLPs, linear maps of the residual stream, compiled bottom-up   -> 60.81% of 4.3301 nats
#   18 attention output writes, lag-1 maps [x_t, x_{t-1}]             -> 56.26% of 3.5570 nats
# Both are ceilings measured with the OTHER half running exactly as trained. Nobody has asked
# what happens when neither half is real.
#
# That is the question the pricing programme actually needs answered, and it is not implied by
# the two halves. §1669 showed substitutions compound off-distribution badly enough to go
# NEGATIVE at eighteen sites, and the fix -- bottom-up compilation -- is what made the MLP
# program work at all. Here there are THIRTY-SIX substituted sites, and the compilation has to
# interleave: within block L, fit attn_L against everything already substituted below it, install
# it, then fit mlp_L with attn_L ALSO substituted. Fitting the two halves independently and
# installing them together would be exactly the §1668 configuration that returned -42.99%.
#
# The stake is recomputed for the joint condition: every attention write AND every MLP replaced
# by its optimal constant. That is a different, larger object than either half's stake, so the
# joint ceiling is not comparable to 60.81% or 56.26% as a fraction -- only as an answer to
# "how much of what these thirty-six modules do can a compiled program reproduce".
#
# CONTROLS, both load-bearing: substituting ONLY the MLPs must reproduce §1676's 60.81%, and
# ONLY the attention writes must reproduce §1685's 56.26%. They are computed here against their
# OWN half-stakes, exactly as those sections did, so a drift means the harness moved and the
# joint number is not interpretable.
#
# Registered predictions:
#   pred_a THE HALVES COMPOUND, NOT COMPOSE: the joint ceiling falls below BOTH half-ceilings
#          (60.81% and 56.26%). Each half's program was fitted against a real other half, and
#          neither has ever had to absorb the other's error.
#   pred_b BUT IT DOES NOT COLLAPSE: the joint ceiling stays above 30%. Below that the
#          interleaved compilation is not holding the thirty-six-site substitution together and
#          the answer is about §1669's limits rather than about bilin18.
#   pred_c CONTROLS: the MLP-only arm lands within 1 point of 60.81% and the attention-only arm
#          within 1 point of 56.26%.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_program_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1676_MLP = 0.6081
S1685_ATTN = 0.5626
S1683_CE_LIVE = 3.29205
STATE = {}
SEENREF = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def lagged(x):
    """[x_t, x_{t-1}], zero-padded at the start (§1685)."""
    p = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
    return torch.cat([x, p], dim=-1).reshape(-1, 2 * D)


def mlp_const_hook(c):
    def hook(mod, args, out):
        return c.to(out.dtype).expand_as(out)
    return hook


def attn_const_hook(c):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def mlp_prog_hook(W):
    def hook(mod, args, out):
        sub = (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
        return torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def attn_prog_hook(W):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = (lagged(args[0]) @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(SEENREF['m'][STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def install(prog):
    hs = []
    for (kind, L), W in prog.items():
        if kind == 'mlp':
            hs.append(H[L].mlp.register_forward_hook(mlp_prog_hook(W)))
        else:
            hs.append(H[L].attn.register_forward_hook(attn_prog_hook(W)))
    return hs


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
    hs = list(hooks)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            STATE['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            if score is not None:
                lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
                score(lg, bb[:, 1:].to(DEV), idx)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def fit_site(rows, kind, L, prog):
    din = D if kind == 'mlp' else 2 * D
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = torch.zeros(din, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = (args[0].reshape(-1, D) if kind == 'mlp' else lagged(args[0])).double()
        y = (out if kind == 'mlp' else (out[0] if isinstance(out, tuple) else out))
        A.add_(x.T @ x); B.add_(x.T @ y.reshape(-1, D).double()); n['v'] += x.shape[0]
        return None
    tgt = H[L].mlp if kind == 'mlp' else H[L].attn
    sweep(rows, hooks=install(prog) + [tgt.register_forward_hook(collect)])
    assert n['v'] > 0, f'{kind}{L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(din, device=DEV, dtype=torch.float64)
    return torch.linalg.solve(a + reg, B / n['v']).float()


@torch.no_grad()
def compile_stack(rows, kinds):
    """Interleaved bottom-up: within block L, attn_L then mlp_L (§1669)."""
    prog = {}
    for L in ALL18:
        for kind in ('attn', 'mlp'):
            if kind in kinds:
                prog[(kind, L)] = fit_site(rows, kind, L, prog)
    return prog


@torch.no_grad()
def seen_mask(rows):
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        t = rows[i:i + 8, :-1].to(DEV).reshape(-1)
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    return c > 0


@torch.no_grad()
def ce(rows, seen, hooks=()):
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    ev = load(EVAL_ROWS)
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)

    cl = ce(ev, seen)
    # KNOWN-ANSWER CHECK ON THE BASELINE. The first version of this script registered the
    # constant hooks directly on the modules and then measured cl THROUGH them, returning
    # CE live 8.86042 instead of 3.29205 -- every stake came out negative and every ceiling
    # NaN. Identity arms verify the INTERVENTION; nothing was verifying the BASELINE, and it
    # is just as available a known answer (§1683 and §1693 both report 3.29205 for this eval
    # set and mask).
    assert abs(cl - S1683_CE_LIVE) <= 1e-3, (
        f'baseline CE {cl:.5f} disagrees with the known live CE {S1683_CE_LIVE:.5f} -- '
        'something is substituted that should not be')

    def const_ce(which):
        hs = []
        if 'mlp' in which:
            hs += [H[L].mlp.register_forward_hook(mlp_const_hook(K[f'mlp{L}'].to(DEV).float()))
                   for L in ALL18]
        if 'attn' in which:
            hs += [H[L].attn.register_forward_hook(attn_const_hook(K[f'attn{L}'].to(DEV).float()))
                   for L in ALL18]
        return ce(ev, seen, hooks=hs)

    stakes = {k: round(const_ce(k) - cl, 5) for k in (('mlp',), ('attn',), ('mlp', 'attn'))}
    print(f'WHOLE MODEL PROGRAM | interleaved bottom-up compilation over 36 sites | '
          f'CE live {cl:.5f}', flush=True)
    for k, v in stakes.items():
        print(f'  stake {str(k):20s} {v:7.4f} nats', flush=True)
    print(f'  half-ceiling references: MLP {S1676_MLP:.2%} (§1676) | attn {S1685_ATTN:.2%} '
          f'(§1685)', flush=True)

    arms = {}
    for name, kinds in (('mlp_only', ('mlp',)), ('attn_only', ('attn',)),
                        ('both', ('mlp', 'attn'))):
        prog = compile_stack(fit, kinds)
        ct = ce(ev, seen, hooks=install(prog))
        cc = const_ce(kinds)
        st = cc - cl
        arms[name] = {'sites': len(prog), 'stake': round(st, 5), 'ce': round(ct, 5),
                      'ceiling': round((cc - ct) / st if st > 1e-6 else float('nan'), 5)}
        print(f'  {name:10s} ({len(prog):2d} sites) stake {st:7.4f} | CEILING '
              f'{arms[name]["ceiling"]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    j = arms['both']['ceiling']
    mo = arms['mlp_only']['ceiling']
    ao = arms['attn_only']['ceiling']

    pa = (j < mo) and (j < ao)
    pb = j > 0.30
    pc = (abs(mo - S1676_MLP) <= 0.01) and (abs(ao - S1685_ATTN) <= 0.01)

    print(f'\n  JOINT (36 sites) {j:.2%} vs MLP-only {mo:.2%} vs attn-only {ao:.2%}', flush=True)
    print(f'  halves compound rather than compose {pa} | does not collapse {pb}', flush=True)
    print(f'  CONTROLS mlp {mo:.2%} vs §1676 {S1676_MLP:.2%} | attn {ao:.2%} vs §1685 '
          f'{S1685_ATTN:.2%} -> {pc}', flush=True)
    print(f'  joint stake {stakes[("mlp", "attn")]:.4f} nats vs halves '
          f'{stakes[("mlp",)]:.4f} + {stakes[("attn",)]:.4f} = '
          f'{stakes[("mlp",)] + stakes[("attn",)]:.4f}', flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'programs': 'linear map of the residual stream at each MLP; lag-1 map '
                                  '[x_t, x_(t-1)] at each attention output write (§1685)',
                      'compilation': 'INTERLEAVED bottom-up -- within block L, attn_L is fitted '
                                     'against everything substituted below it, installed, then '
                                     'mlp_L is fitted with attn_L also substituted (§1669)',
                      'why_interleaved': 'fitting the halves independently and installing them '
                                         'together is the §1668 configuration that returned -42.99%',
                      'coverage': 'hybrid, mask pinned to n96_skip80 (§1676)',
                      'scoring': 'covered positions only',
                      'stake': 'recomputed per condition; the joint stake is a larger object than '
                               'either half-stake, so the joint ceiling is not a fraction comparable '
                               'to 60.81% or 56.26%',
                      'v1_scope': 'attention v1 passed through unchanged (§1682, §1684)',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1676_mlp': S1676_MLP, 's1685_attn': S1685_ATTN},
           'ce_live': round(cl, 5),
           'stakes': {'_'.join(k): v for k, v in stakes.items()},
           'arms': arms,
           'predictions': {'pred_a_halves_compound': bool(pa),
                           'pred_b_does_not_collapse_gt_30': bool(pb),
                           'pred_c_controls_hold': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
