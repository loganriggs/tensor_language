# program_family_assignment: WHICH SITES ACTUALLY WANT A TABLE?
#
# §1671 located the compiled program's missing 43.71% and found it DIFFUSE -- fourteen of
# eighteen sites sit in a narrow 1.18-1.94 point band, and the worst single site (mlp3)
# holds under 10% of the shortfall. No bottleneck to attack.
#
# But it handed over a clean sub-pattern. Inside the front band, where §1670's program uses
# token tables, the gain from exempting a site RISES monotonically with depth:
#     mlp0 +0.23   mlp1 +1.66   mlp2 +3.17   mlp3 +4.33
# The table is nearly perfect at mlp0 and steadily worse by mlp3. That matches the per-site
# ceilings in §1662 exactly -- mlp0 90.27%, mlp1 96.01%, mlp2 76.98%, mlp3 67.55% -- and it
# says the mlp0-3 boundary I used was inherited from §1668's BAND-LEVEL verdict, which was
# a joint measurement over four sites that are not alike.
#
# So the boundary is probably in the wrong place, and the program should improve by moving
# it. Arms, every one compiled bottom-up and scored identically:
#     all linear (no tables)          -- §1670's 54.28%, a second control
#     tables at mlp0 only
#     tables at mlp0-1
#     tables at mlp0-2
#     tables at mlp0-3                -- §1670's 56.29%, the control
#     greedy: each site takes whichever family reconstructs its OWN output better in L2,
#             decided during compilation with the stack below already substituted
# The greedy arm is the interesting one: it is chosen by a purely LOCAL criterion, with no
# access to the end-to-end CE it is scored on. If a local choice recovers the end-to-end
# optimum, family assignment is a per-site property that can be decided cheaply.
#
# Registered predictions:
#   pred_a THE BOUNDARY IS IN THE WRONG PLACE: the best fixed split beats §1670's mlp0-3
#          program by >= 1 percentage point.
#   pred_b AND IT MOVES FORWARD, NOT BACK: the winning fixed split uses FEWER table sites
#          than mlp0-3, i.e. tables earn their place only at the very front.
#   pred_c CONTROL -- the mlp0-3 arm reproduces §1670's 56.29% within 1 point. Every
#          comparison here is against that number, so if it drifts nothing else is readable.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'program_family_assignment_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1670_MIXED_0_3 = 0.5629
S1669_ALL_LINEAR = 0.5428
S1671_FRONT_GAINS = {'mlp0': 0.0023, 'mlp1': 0.0166, 'mlp2': 0.0317, 'mlp3': 0.0433}
S1668_NAIVE_TABLE = 0.3427
S1668_BANDS = {'front_token': 0.7645, 'front_linear': 0.6868, 'middle_linear': 0.6233,
               'late_linear': 0.8360}
STATE = {}


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


def install(prog):
    """prog: site -> ('linear', W) | ('table', tbl, seen)"""
    hs = []
    for L, p in prog.items():
        hs.append(H[L].mlp.register_forward_hook(
            linear_hook(p[1]) if p[0] == 'linear' else table_hook(p[1], p[2])))
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
        return ('linear', torch.linalg.solve(a + reg, B / n['v']).float())
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
def fit_both_and_pick(rows, L, prog, seen):
    """Fit both families at site L and keep whichever reconstructs L's OWN output better
    in position-weighted L2. Purely local -- no access to end-to-end CE."""
    lin = fit_site(rows, L, 'linear', prog, seen)
    tab = fit_site(rows, L, 'table', prog, seen)
    err = {'linear': 0.0, 'table': 0.0, 'n': 0.0}

    def judge(mod, args, out):
        y = out.float().reshape(-1, D)
        xin = args[0].reshape(-1, D)
        pl = xin @ lin[1]
        sub = tab[1][STATE['idx'].reshape(-1)]
        pt = torch.where(tab[2][STATE['idx'].reshape(-1)].unsqueeze(-1), sub, y)
        err['linear'] += float(((pl - y) ** 2).sum())
        err['table'] += float(((pt - y) ** 2).sum())
        err['n'] += y.numel()
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(judge)])
    assert err['n'] > 0, f'site {L}: greedy judge never fired'
    pick = 'table' if err['table'] < err['linear'] else 'linear'
    return (tab if pick == 'table' else lin), pick, err


def compile_program(rows, kinds, seen):
    prog = {}
    picks = {}
    for L in ALL18:
        if kinds[L] == 'greedy':
            prog[L], picks[L], _ = fit_both_and_pick(rows, L, prog, seen)
        else:
            prog[L] = fit_site(rows, L, kinds[L], prog, seen)
            picks[L] = kinds[L]
    return prog, picks


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
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    seen = seen_mask(fit)
    print(f'PROGRAM FAMILY ASSIGNMENT | moving the table/linear boundary | ridge {RIDGE} | '
          f'fit skip1200, eval skip7000', flush=True)
    print(f'  §1671 front-band exemption gains, which motivate this: ' +
          '  '.join(f'{k} +{v:.2%}' for k, v in S1671_FRONT_GAINS.items()), flush=True)

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'  CE live {cl:.5f} | all-MLP constant {cc:.5f} | stake {st:.4f} nats', flush=True)

    arms = {}
    for n in (0, 1, 2, 3):
        arms[f'table_mlp0_{n - 1}' if n else 'all_linear'] = \
            {L: ('table' if L < n else 'linear') for L in ALL18}
    arms['table_mlp0_3'] = {L: ('table' if L < 4 else 'linear') for L in ALL18}
    arms['greedy_local_l2'] = {L: 'greedy' for L in ALL18}

    out = {}
    for name, kinds in arms.items():
        prog, picks = compile_program(fit, kinds, seen)
        ct = ce(ev, seen, hooks=install(prog))
        ceil = (cc - ct) / st if st > 1e-6 else float('nan')
        ntab = sum(1 for v in picks.values() if v == 'table')
        out[name] = {'ceiling': round(ceil, 5), 'ce': round(ct, 5), 'n_table_sites': ntab,
                     'picks': {f'mlp{L}': picks[L] for L in ALL18}}
        print(f'  {name:16s} {ntab:2d} table sites | CE {ct:.5f} | CEILING {ceil:7.2%}',
              flush=True)
        if name == 'greedy_local_l2':
            print(f'      greedy picked tables at: '
                  f'{[L for L in ALL18 if picks[L] == "table"]}', flush=True)
        del prog
        torch.cuda.empty_cache()

    ctrl = out['table_mlp0_3']['ceiling']
    fixed = {k: v for k, v in out.items() if k != 'greedy_local_l2'}
    best_name = max(fixed, key=lambda k: fixed[k]['ceiling'])
    best = fixed[best_name]['ceiling']
    greedy = out['greedy_local_l2']['ceiling']

    pa = (best - ctrl) >= 0.01
    pb = fixed[best_name]['n_table_sites'] < 4
    pc = abs(ctrl - S1670_MIXED_0_3) <= 0.01

    print(f'\n  BEST FIXED SPLIT: {best_name} at {best:.2%}  vs §1670 mlp0-3 {ctrl:.2%}  '
          f'-> boundary was wrong {pa}', flush=True)
    print(f'  it uses {fixed[best_name]["n_table_sites"]} table sites (was 4) -> moves '
          f'forward {pb}', flush=True)
    print(f'  GREEDY (local L2, no access to CE): {greedy:.2%} with '
          f'{out["greedy_local_l2"]["n_table_sites"]} table sites  '
          f'-> vs best fixed {greedy - best:+.2%}', flush=True)
    print(f'  CONTROL mlp0-3 {ctrl:.2%} vs §1670 {S1670_MIXED_0_3:.2%} -> {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'compilation': 'bottom-up (§1669)',
                      'table_substitution': 'HYBRID -- table at covered positions, module live elsewhere (§1661)',
                      'scoring': 'covered positions only',
                      'greedy_criterion': 'position-weighted L2 on the site OWN output, decided during '
                                          'compilation with the stack below substituted -- purely local',
                      's1670_mixed_0_3': S1670_MIXED_0_3, 's1669_all_linear': S1669_ALL_LINEAR,
                      's1671_front_gains': S1671_FRONT_GAINS},
           'stake': round(st, 5), 'arms': out, 'best_fixed': best_name,
           'predictions': {'pred_a_better_split_exists_ge_1pt': bool(pa),
                           'pred_b_boundary_moves_forward': bool(pb),
                           'pred_c_control_reproduces_s1670': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
