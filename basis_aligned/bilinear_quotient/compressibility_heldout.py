# compressibility_heldout: DOES THE COMPRESSIBILITY ORDERING SURVIVE HELD-OUT DOCUMENTS?
#
# §1692's headline rests on one eval set: at rank 64 of 1152, attention routing +62.82%,
# attention values +2.37%, MLP readout -15.16%, MLP feature-forming -52.88%. Four numbers, one
# document sample, and a claim ("a compact program can approximate WHERE the model looks but
# not WHAT it computes") that every later decision about where to spend budget would rest on.
#
# §1683 is the precedent: the MLP program figures were re-scored on fineweb_n192_skip11000 and
# all four arms held within 0.91 points. That set has been used once, for §1683. Nothing in the
# §1689-§1692 rank work has touched it.
#
# This is rung 2, second-class confirmation, house pattern (§1595, §1598, §1603): the SAME
# fitted programs, compiled once, scored on both eval sets. Only the scoring documents change.
#
# The ORDERING is what the claim needs, not the levels -- four paths whose separations are
# 60.45, 17.53 and 37.72 points at rank 64 should not reorder under a document resample, and if
# they do the claim was never about the model.
#
# Arms at rank 64 (the rank the claim is quoted at) plus each path's full-rank identity, which
# must return ~100% on BOTH eval sets -- a known answer that travels.
#
# Registered predictions:
#   pred_a THE ORDERING IS PRESERVED on the held-out set: routing > values > Down > LeftRight,
#          the same order as skip7000.
#   pred_b THE LEVELS ARE STABLE: every arm moves by <= 5 percentage points between the two
#          eval sets. §1683's four program arms moved by at most 0.91, but these arms sit in
#          steeper parts of their curves, so 5 is the bar rather than 1.
#   pred_c IDENTITY CHECKS TRAVEL: all four full-rank arms return >= 0.99 on BOTH eval sets.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
RANKS = [64, 1152]
DH = 4608
ATTN_PROJ = {'routing': ('c_q', 'c_k', 'c_q2', 'c_k2'), 'values': ('c_v',)}
MLP_PROJ = {'Down': ('Down',), 'LeftRight': ('Left', 'Right')}
ARMS = dict(MLP_PROJ)
S1692_RANK64 = {'routing': 0.6282, 'values': 0.0237, 'Down': -0.1516, 'LeftRight': -0.5288}
ARMSTATE = {'a': 'Down'}
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'compressibility_heldout_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
EVAL_ROWS = EVAL_SETS[0][1]
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1687_BEST_POSITIONAL = 0.7008
S1685_LAG1 = 0.5626
S1682_LAG0 = 0.1638
STATE = {}
RANKSTATE = {'r': D}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def const_hook(const):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = const.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def value_hook(W):
    """Replace a projection's output with (its own input) @ W."""
    def hook(mod, args, out):
        din = W.shape[0]
        return (args[0].reshape(-1, din) @ W).reshape(out.shape).to(out.dtype)
    return hook


def owner(L):
    return H[L].mlp if ARMSTATE['a'] in MLP_PROJ else H[L].attn


def install(prog):
    hs = []
    for L, Ws in prog.items():
        for nm, W in Ws.items():
            hs.append(getattr(owner(L), nm).register_forward_hook(value_hook(W)))
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
def fit_site(rows, L, prog):
    """Rank-r maps for the projections of the current arm, from one pass."""
    a = ARMSTATE['a']
    names = (MLP_PROJ | ATTN_PROJ)[a]
    din = DH if names[0] == 'Down' else D
    dout = D if names[0] == 'Down' else (DH if a == 'LeftRight' else D)
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = {nm: torch.zeros(din, dout, device=DEV, dtype=torch.float64) for nm in names}
    n = {'v': 0}
    hs = []

    def mk(nm, first):
        def hook(mod, args, out):
            x = args[0].reshape(-1, din).double()
            B[nm].add_(x.T @ out.reshape(-1, dout).double())
            if first:
                A.add_(x.T @ x); n['v'] += x.shape[0]
            return None
        return hook
    for j, nm in enumerate(names):
        hs.append(getattr(owner(L), nm).register_forward_hook(mk(nm, j == 0)))
    sweep(rows, hooks=install(prog) + hs)
    assert n['v'] > 0, f'site {L} arm {a}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(din, device=DEV, dtype=torch.float64)
    r = RANKSTATE['r']
    out = {}
    for nm in names:
        W = torch.linalg.solve(a + reg, B[nm] / n['v']).float()
        if r < min(din, dout):
            U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
            W = ((U[:, :r] * S[:r]) @ Vh[:r]).float()
        out[nm] = W
    return out


@torch.no_grad()
def compile_stack(rows):
    prog = {}
    for L in ALL18:
        prog[L] = fit_site(rows, L, prog)
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
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)
    print(f'COMPRESSIBILITY HELD-OUT | four matched paths at rank 64, compiled ONCE, scored on '
          f'{[n for n, _ in EVAL_SETS]} | §1692 reference: ' +
          '  '.join(f'{k} {v:.2%}' for k, v in S1692_RANK64.items()), flush=True)

    progs = {}
    for arm in ('routing', 'values', 'Down', 'LeftRight'):
        ARMSTATE['a'] = arm
        for r in RANKS:
            RANKSTATE['r'] = r
            progs[(arm, r)] = compile_stack(fit)
        print(f'  compiled {arm}', flush=True)
    del fit
    torch.cuda.empty_cache()

    out = {}
    for ename, epath in EVAL_SETS:
        ev = load(epath)
        row = {}
        for fam, mods in (('mlp', ALL18), ('attn', ALL18)):
            pass
        cl = ce(ev, seen)
        cc_mlp = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
            (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
                K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
        cc_att = ce(ev, seen, hooks=[H[L].attn.register_forward_hook(
            (lambda cst: (lambda mo, a, o: (
                cst.to(o[0].dtype).expand_as(o[0]),) + tuple(o[1:])
                if isinstance(o, tuple) else cst.to(o.dtype).expand_as(o)))(
                K[f'attn{L}'].to(DEV).float())) for L in ALL18])
        row['stake_mlp'] = round(cc_mlp - cl, 5)
        row['stake_attn'] = round(cc_att - cl, 5)
        print(f'  {ename:10s} CE live {cl:.5f} | MLP stake {cc_mlp - cl:.4f} | '
              f'attn stake {cc_att - cl:.4f}', flush=True)
        for arm in ('routing', 'values', 'Down', 'LeftRight'):
            ARMSTATE['a'] = arm
            cc = cc_mlp if arm in MLP_PROJ else cc_att
            st = cc - cl
            for r in RANKS:
                ct = ce(ev, seen, hooks=install(progs[(arm, r)]))
                row[f'{arm}_r{r}'] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
            print(f'      {arm:10s} rank64 {row[f"{arm}_r64"]:8.2%} | identity '
                  f'{row[f"{arm}_r1152"]:8.2%}', flush=True)
        out[ename] = row
        del ev
        torch.cuda.empty_cache()

    ref, held = out['skip7000'], out['skip11000']
    names = ('routing', 'values', 'Down', 'LeftRight')

    def order(r):
        return sorted(names, key=lambda k: -r[f'{k}_r64'])
    deltas = {k: held[f'{k}_r64'] - ref[f'{k}_r64'] for k in names}

    pa = order(held) == list(names)
    pb = all(abs(v) <= 0.05 for v in deltas.values())
    pc = all(out[e][f'{k}_r1152'] >= 0.99 for e in out for k in names)

    print(f'\n  ORDERING held out: {order(held)} -> preserved {pa}', flush=True)
    print(f'  rank-64 deltas: ' + '  '.join(f'{k} {deltas[k]:+.2%}' for k in names) +
          f'  -> stable {pb}', flush=True)
    print(f'  identity checks travel to both eval sets {pc}', flush=True)
    print(f'  vs §1692 on skip7000: ' +
          '  '.join(f'{k} {ref[f"{k}_r64"] - S1692_RANK64[k]:+.2%}' for k in names), flush=True)

    res = {'config': {'sites': ALL18, 'ranks': RANKS,
                      'arms': {**{k: list(v) for k, v in MLP_PROJ.items()},
                               **{k: list(v) for k, v in ATTN_PROJ.items()}},
                      'eval_sets': [n for n, _ in EVAL_SETS],
                      'held_out': 'fineweb_n192_skip11000 -- untouched by the §1689-§1692 rank work',
                      'compilation': 'bottom-up (§1669); programs compiled ONCE and scored on both evals',
                      'coverage': 'mask pinned to n96_skip80', 'fit_rows': 'fineweb_n480_skip80.pt',
                      'pattern': 'house second-class confirmation (§1595, §1598, §1603; §1683 precedent)',
                      's1692_rank64': S1692_RANK64},
           'evals': out, 'rank64_deltas': {k: round(v, 5) for k, v in deltas.items()},
           'ordering': {'reference': order(ref), 'held_out': order(held)},
           'predictions': {'pred_a_ordering_preserved': bool(pa),
                           'pred_b_levels_stable_le_5pts': bool(pb),
                           'pred_c_identity_checks_travel': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
