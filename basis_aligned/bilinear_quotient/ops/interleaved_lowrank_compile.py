# INTERLEAVED BOTTOM-UP COMPILATION OF table + x W_r
#
# §1747: every MLP with a gap worth measuring is 88-100% linearly correctable over its per-token
# table, the median attention site is at zero, and installing all 36 corrections TOGETHER costs half
# a nat MORE than plain tables (-0.5462 held out). Each W was fitted against an all-tabled context
# which stops existing the moment the other thirty-five are live.
#
# §1669 is the known answer to exactly that failure, and it has never been applied to this program
# class: compile BOTTOM-UP AND INTERLEAVED. Walk the sites in forward order; at each one fit W with
# everything already compiled BELOW it substituted and everything above still LIVE; install it; move
# on. Each map is then fitted against the inputs it will actually receive from the compiled prefix.
# §1669 measured the alternative -- independently fitted programs installed jointly -- at -42.99%.
#
# Two ranks, because §1746 found the rank curve flat and rank 8 the efficient point (0.018M reals per
# site), so the question is whether the cheap rank composes as well as the expensive one.
#
# ROLES. Fitting uses the fit rows; both eval roles reported. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other so no arm is decided
# by another's outcome:
#   pred_a INTERLEAVING TURNS THE JOINT PROGRAM POSITIVE: at rank 8 it recovers more than zero nats
#          over table-only, held out. If FALSE, this program class does not compose even under the
#          procedure built to make programs compose, and the 88-100% per-site figures of §1747 are an
#          illusion for compiler purposes -- which is the outcome that matters most to state plainly.
#   pred_b IT BEATS THE SIMULTANEOUS FIT BY AT LEAST 1.0 NAT: rank-8 interleaved minus §1747's
#          -0.5462 >= 1.0 held out. Scored independently of pred_a, since it can beat -0.5462 by a
#          lot and still be negative.
#   pred_c IT DOES NOT REACH THE SUM OF THE PARTS: the joint recovery stays below the sum of the
#          per-site solo recoveries §1747 measured (2.3618 nats held out). If FALSE, composition is
#          fully solved by interleaving and the sites are effectively independent once compiled --
#          a much stronger result than I expect and one that would change the compiler plan.
#   pred_d CONTROLS: table-only CE reproduces 7.35114 within 0.005, live CE reproduces 3.29205 within
#          1e-3, fit coverage is 5419 of 50257, and every one of the 36 per-site fits fires on the
#          full 24576 positions.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANKS = (8, 32)
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/interleaved_lowrank_compile_results.json'
MAP = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1747_SIMULTANEOUS = {'skip7000': -0.5235, 'skip11000': -0.5462}   # rank 8, all-36 joint
COV = {}
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def table_hook(tbl, seen):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def lowrank_hook(tbl, seen, W):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = sub + (args[0].reshape(-1, D).to(W.dtype) @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


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
def ce(rows, hooks=()):
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        cov = COV['seen'][idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc['t'] / acc['n']


@torch.no_grad()
def fit_tables(rows, sites):
    s = {st: torch.zeros(50257, D, device=DEV) for st in sites}
    c = torch.zeros(50257, device=DEV)
    fired = {'n': 0}

    def mk(st, first):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).float().reshape(-1, D)
            t = STATE['idx'].reshape(-1)
            s[st].index_add_(0, t, y)
            if first:
                c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
                fired['n'] += 1
            return None
        return hook
    sweep(rows, hooks=[mod_of(*st).register_forward_hook(mk(st, j == 0))
                       for j, st in enumerate(sites)])
    assert fired['n'] > 0, 'table fit never fired'
    seen = c > 0
    out = {}
    for st in sites:
        mean = s[st].sum(0) / c.sum()
        tbl = mean.unsqueeze(0).repeat(50257, 1)
        tbl[seen] = s[st][seen] / c[seen].unsqueeze(1)
        out[st] = tbl
    return out, seen


@torch.no_grad()
def fit_one(rows, st, tables, seen, installed, rank):
    """Fit W at ONE site with the already-compiled prefix substituted and everything else LIVE."""
    xtx = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    xtr = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'k': 0}

    def hook(mod, args, out):
        y = (out[0] if isinstance(out, tuple) else out).reshape(-1, D).double()
        x = args[0].reshape(-1, D).double()
        nonlocal xtx, xtr
        xtx += x.T @ x
        xtr += x.T @ (y - tables[st][STATE['idx'].reshape(-1)].double())
        n['k'] += x.shape[0]
        return None
    hooks = [mod_of(*st).register_forward_hook(hook)]
    for s2, W2 in installed.items():
        hooks.append(mod_of(*s2).register_forward_hook(lowrank_hook(tables[s2], seen, W2)))
    sweep(rows, hooks=hooks)
    assert n['k'] > 0, f'fit at {st} never fired'
    A = xtx + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (n['k'] / D)
    U, S, Vh = torch.linalg.svd(torch.linalg.solve(A, xtr))
    return ((U[:, :rank] * S[:rank]) @ Vh[:rank]).float(), n['k']


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    # forward order within a block: attention writes before the MLP reads it
    order = [(k, L) for L in range(18) for k in ('attn', 'mlp')]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    prior = json.load(open(MAP))['results']
    print(f'INTERLEAVED BOTTOM-UP COMPILE | table + x W_r, ranks {RANKS} | 36 sites in forward '
          f'order, each fitted against the compiled prefix | DISCOVERY ONLY', flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    fits_ok = True
    progs = {}
    for rank in RANKS:
        installed = {}
        for st in order:
            W, nk = fit_one(fit, st, tables, seen, installed, rank)
            fits_ok = fits_ok and (nk == 24576)
            installed[st] = W
        progs[rank] = dict(installed)
        print(f'  rank {rank}: compiled all 36 sites ({time.time() - t0:.0f}s elapsed)', flush=True)

    out = {}
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        cl = ce(ev)
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        tbl_only = ce(ev, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                           for st in sites])
        stake = tbl_only - cl
        arms = {}
        for rank in RANKS:
            c1 = ce(ev, [mod_of(*st).register_forward_hook(
                lowrank_hook(tables[st], seen, progs[rank][st])) for st in sites])
            arms[rank] = {'ce': round(c1, 5), 'recovered': round(tbl_only - c1, 5),
                          'frac_of_stake': round((tbl_only - c1) / stake, 5),
                          'cost_M': round(36 * 2 * rank * D / 1e6, 4)}
        solo_sum = sum(v['recovered'] for v in prior[ename]['per_site'].values())
        print(f'\n  {ename}: live {cl:.5f} | table-only {tbl_only:.5f} | stake {stake:.4f} nats',
              flush=True)
        for rank in RANKS:
            a = arms[rank]
            print(f'    INTERLEAVED rank {rank:3d}: recovers {a["recovered"]:+8.4f} = '
                  f'{a["frac_of_stake"]:+7.2%} of the stake, for {a["cost_M"]:.3f}M reals',
                  flush=True)
        print(f'    (§1747 SIMULTANEOUS rank 8: {S1747_SIMULTANEOUS[ename]:+.4f} | sum of the 36 '
              f'solo recoveries: {solo_sum:+.4f})', flush=True)
        out[ename] = {'live_ce': round(cl, 5), 'table_only_ce': round(tbl_only, 5),
                      'stake_nats': round(stake, 5), 'arms': arms,
                      'simultaneous_rank8': S1747_SIMULTANEOUS[ename],
                      'sum_of_solo_recoveries': round(solo_sum, 5)}
        del ev
        torch.cuda.empty_cache()

    ho = out['skip11000']
    pa = ho['arms'][RANKS[0]]['recovered'] > 0.0
    pb = (ho['arms'][RANKS[0]]['recovered'] - S1747_SIMULTANEOUS['skip11000']) >= 1.0
    pc = ho['arms'][RANKS[0]]['recovered'] < ho['sum_of_solo_recoveries']
    pd = (abs(out['skip7000']['table_only_ce'] - S1738_PROGRAM_CE) <= 0.005
          and abs(out['skip7000']['live_ce'] - 3.29205) <= 1e-3 and ncov == 5419 and fits_ok)

    print(f'\n  interleaved rank {RANKS[0]} is POSITIVE held out '
          f'({ho["arms"][RANKS[0]]["recovered"]:+.4f}) -> {pa}', flush=True)
    print(f'  it beats the simultaneous fit by >=1.0 nat '
          f'({ho["arms"][RANKS[0]]["recovered"] - S1747_SIMULTANEOUS["skip11000"]:+.4f}) -> {pb}',
          flush=True)
    print(f'  it stays below the sum of the parts ({ho["sum_of_solo_recoveries"]:+.4f}) -> {pc}',
          flush=True)
    print(f'  table-only + live CE + coverage {ncov} + all 36 fits full-size -> control {pd}',
          flush=True)

    r = {'config': {'ranks': list(RANKS), 'ridge': RIDGE,
                    'order': 'forward, attention then MLP within each block (§1669)',
                    'procedure': 'at each site fit W with the already-compiled prefix substituted '
                                 'and everything above still LIVE, then install it before moving on',
                    'comparison': 'against §1747, where all 36 W were fitted simultaneously against '
                                  'an all-tabled context and the joint install came out NEGATIVE',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'results': out,
         'predictions': {'pred_a_interleaved_is_positive': bool(pa),
                         'pred_b_beats_simultaneous_by_a_nat': bool(pb),
                         'pred_c_below_sum_of_parts': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
