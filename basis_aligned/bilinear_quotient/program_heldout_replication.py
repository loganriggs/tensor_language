# program_heldout_replication: THE PROGRAM NUMBERS ON DOCUMENTS THEY HAVE NEVER SEEN
#
# Everything in §1664-§1681 -- the 60.81% whole-stack linear program, the family ordering,
# the per-site table boundary, the price curve -- was measured on ONE eval set,
# fineweb_n192_skip7000. Fits were on disjoint documents throughout, so nothing is fitted on
# its own eval. But a single eval set is a single sample, and every downstream claim in the
# arc inherits whatever is idiosyncratic about those 192 documents.
#
# fineweb_n192_skip11000 has been sitting in .rowcache untouched for the whole arc. This is
# the house second-class-confirmation pattern (§1595, §1598, §1603): same artifacts, same
# protocol, documents never used.
#
# WHAT IS AND IS NOT HELD FIXED. The programs are compiled once, on n480_skip80, with the
# coverage mask pinned to n96_skip80 (§1676) -- identical artifacts scored twice. The STAKE
# is recomputed per eval set, because it is a property of the eval documents; the ceiling is
# a ratio within its own eval set and that is what makes the two comparable. Both stakes are
# reported so any difference in difficulty between the two document samples is visible
# rather than hidden inside a ratio.
#
# Four arms, the ones the arc's conclusions actually rest on:
#   all_linear         §1676's 60.81% -- the headline program
#   table_mlp0_2       §1672's per-site boundary result
#   additive           §1676's 59.08% -- the family closing fastest with data
#   linear_rank128     §1680's price point, 22% of the parameters
#
# Registered predictions:
#   pred_a THE HEADLINE REPLICATES: all_linear on the held-out set is within 3 points of
#          60.81%.
#   pred_b THE ORDERING REPLICATES: linear > additive > table on the held-out set, as on
#          skip7000. The ordering is what the arc's conclusions use; the levels matter less.
#   pred_c CONTROL -- the skip7000 arm reproduces §1676's 60.81% within 0.5 points. Without
#          it a difference on skip11000 cannot be attributed to the documents.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'program_heldout_replication_results.json'
FIT_SETS = [('n96_skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt'),
            ('n96_skip80', PT + '.rowcache/fineweb_n96_skip80.pt'),
            ('n480_skip80', PT + '.rowcache/fineweb_n480_skip80.pt')]
FIT_ROWS = FIT_SETS[0][1]
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
EVAL_ROWS = EVAL_SETS[0][1]
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1672_BEST_PURE = 0.5729
S1676_REF = {'linear': 0.6081, 'additive': 0.5908, 'table_mlp0_2': 0.5716}
S1680_RANK128 = 0.5412
RANKSTATE = {}
S1675_GROWING_MASK_SIZE_EFFECT = {'additive': 0.0123, 'linear': -0.0329, 'table_mlp0_2': -0.0045}
S1669_ALL_LINEAR = 0.5428
S1668_NAIVE_TABLE = 0.3427
S1668_BANDS = {'front_token': 0.7645, 'front_linear': 0.6868, 'middle_linear': 0.6233,
               'late_linear': 0.8360}
STATE = {}
SEENREF = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def linear_hook(W):
    def hook(mod, args, out):
        return (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
    return hook


def table_hook(tbl, seen):
    def hook(mod, args, out):
        sub = tbl[STATE['idx'].reshape(-1)].reshape(out.shape).to(out.dtype)
        return torch.where(seen[STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def linear_hybrid_hook(W, seen):
    def hook(mod, args, out):
        sub = (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
        return torch.where(seen[STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def additive_hybrid_hook(tbl, W, seen):
    def hook(mod, args, out):
        b = tbl[STATE['idx'].reshape(-1)]
        sub = (b + args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
        return torch.where(seen[STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def additive_hook(tbl, W):
    """y_hat = b(token) + xW, defined at EVERY position (b falls back to the
    position-weighted mean at unseen tokens, which is already baked into tbl)."""
    def hook(mod, args, out):
        b = tbl[STATE['idx'].reshape(-1)]
        return (b + args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
    return hook


def install(prog):
    """prog: site -> ('linear', W) | ('table', tbl, seen)"""
    hs = []
    for L, p in prog.items():
        hs.append(H[L].mlp.register_forward_hook(
            linear_hook(p[1]) if p[0] == 'linear'
            else linear_hybrid_hook(p[1], SEENREF['m']) if p[0] == 'linear_hybrid'
            else additive_hook(p[1], p[2]) if p[0] == 'additive'
            else additive_hybrid_hook(p[1], p[2], SEENREF['m']) if p[0] == 'additive_hybrid'
            else table_hook(p[1], p[2])))
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
def fit_site(rows, L, kind, prog, seen):
    """Fit site L's program with everything already in `prog` installed."""
    if kind == 'linear':
        A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
        B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
        n = {'v': 0}

        def collect(mod, args, out):
            x = args[0].reshape(-1, D).double(); y = out.reshape(-1, D).double()
            A.add_(x.T @ x); B.add_(x.T @ y); n['v'] += x.shape[0]
            return None
        sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect)])
        assert n['v'] > 0, f'site {L}: no fit positions'
        a = A / n['v']
        reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
        W = torch.linalg.solve(a + reg, B / n['v']).float()
        r = RANKSTATE.get('r', D)
        if r < D:
            U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
            W = ((U[:, :r] * S[:r]) @ Vh[:r]).float()
        return ('linear', W)
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)

    def collect_t(mod, args, out):
        t = STATE['idx'].reshape(-1)
        s.index_add_(0, t, out.float().reshape(-1, D))
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect_t)])
    assert float(c.sum()) > 0, f'site {L}: no token counts'
    sn = c > 0
    tbl = (s.sum(0) / c.sum()).unsqueeze(0).repeat(50257, 1)
    tbl[sn] = s[sn] / c[sn].unsqueeze(1)
    return ('table', tbl, sn)


@torch.no_grad()
def fit_additive(rows, L, prog, seen):
    """Table first, then least squares on its residual. Two passes."""
    _, tbl, _ = fit_site(rows, L, 'table', prog, seen)
    A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = args[0].reshape(-1, D).double()
        r = (out.float().reshape(-1, D) - tbl[STATE['idx'].reshape(-1)]).double()
        A.add_(x.T @ x); B.add_(x.T @ r); n['v'] += x.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect)])
    assert n['v'] > 0, f'site {L}: no additive fit positions'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    return ('additive', tbl, torch.linalg.solve(a + reg, B / n['v']).float())


@torch.no_grad()
def compile_program(rows, kinds, seen):
    prog = {}
    for L in ALL18:
        k = kinds[L]
        if k in ('additive', 'additive_hybrid'):
            p = fit_additive(rows, L, prog, seen)
            prog[L] = (k, p[1], p[2])
        elif k == 'linear_hybrid':
            p = fit_site(rows, L, 'linear', prog, seen)
            prog[L] = ('linear_hybrid', p[1])
        else:
            prog[L] = fit_site(rows, L, k, prog, seen)
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
    mask_rows = load(FIT_SETS[1][1])
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_SETS[2][1])
    print(f'PROGRAM HELD-OUT REPLICATION | compile once on n480_skip80 (mask n96_skip80), '
          f'score on {[n for n, _ in EVAL_SETS]} | ridge {RIDGE}', flush=True)

    ARMS = {
        'all_linear': ({L: 'linear_hybrid' for L in ALL18}, D),
        'table_mlp0_2': ({L: ('table' if L < 3 else 'linear') for L in ALL18}, D),
        'additive': ({L: 'additive_hybrid' for L in ALL18}, D),
        'linear_rank128': ({L: 'linear_hybrid' for L in ALL18}, 128),
    }
    progs = {}
    for name, (kinds, r) in ARMS.items():
        RANKSTATE['r'] = r
        progs[name] = compile_program(fit, kinds, seen)
        print(f'  compiled {name}', flush=True)
    RANKSTATE['r'] = D
    del fit
    torch.cuda.empty_cache()

    out = {}
    for ename, epath in EVAL_SETS:
        ev = load(epath)
        cl = ce(ev, seen)
        cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
            (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
                K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
        st = cc - cl
        row = {'ce_live': round(cl, 5), 'ce_const': round(cc, 5), 'stake': round(st, 5)}
        print(f'  {ename:10s} CE live {cl:.5f} | stake {st:.4f} nats', flush=True)
        for name in ARMS:
            ct = ce(ev, seen, hooks=install(progs[name]))
            row[name] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
            print(f'      {name:15s} CEILING {row[name]:7.2%}', flush=True)
        out[ename] = row
        del ev
        torch.cuda.empty_cache()

    ref, held = out['skip7000'], out['skip11000']
    vals = [held[n] for n in ARMS]
    assert len(set(vals)) > 1, f'all arms identical on the held-out set: {vals}'

    def order(r):
        return [n for n in sorted(('all_linear', 'additive', 'table_mlp0_2'),
                                  key=lambda k: -r[k])]

    pa = abs(held['all_linear'] - S1676_REF['linear']) <= 0.03
    pb = order(held) == order(ref) == ['all_linear', 'additive', 'table_mlp0_2']
    pc = abs(ref['all_linear'] - S1676_REF['linear']) <= 0.005

    print(f'\n  ARM-BY-ARM, reference -> held out:', flush=True)
    for name in ARMS:
        print(f'    {name:15s} {ref[name]:7.2%} -> {held[name]:7.2%}   '
              f'{held[name] - ref[name]:+.2%}', flush=True)
    print(f'  stakes: skip7000 {ref["stake"]:.4f} | skip11000 {held["stake"]:.4f} '
          f'({held["stake"] - ref["stake"]:+.4f} nats -- document difficulty, not a ceiling)',
          flush=True)
    print(f'  headline replicates within 3 points {pa} | ordering replicates {pb} '
          f'({order(held)}) | control {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n480_skip80.pt',
                      'coverage_mask': 'fineweb_n96_skip80.pt, pinned (§1676)',
                      'eval_sets': [n for n, _ in EVAL_SETS],
                      'held_out': 'fineweb_n192_skip11000 -- untouched for the whole §1664-§1681 arc',
                      'compilation': 'bottom-up (§1669); programs compiled ONCE and scored on both evals',
                      'stake': 'recomputed per eval set (a property of the eval documents); the ceiling '
                               'is a ratio within its own set, which is what makes the two comparable',
                      'pattern': 'house second-class confirmation (§1595, §1598, §1603)',
                      's1676_reference': S1676_REF, 's1680_rank128': S1680_RANK128},
           'evals': out,
           'deltas': {n: round(held[n] - ref[n], 5) for n in ARMS},
           'ordering': {'reference': order(ref), 'held_out': order(held)},
           'predictions': {'pred_a_headline_replicates_within_3pts': bool(pa),
                           'pred_b_ordering_replicates': bool(pb),
                           'pred_c_control_reproduces_s1676': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
