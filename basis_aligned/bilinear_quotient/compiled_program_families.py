# compiled_program_families: THE BEST COMPILED PROGRAM FOR ALL EIGHTEEN MLPs
#
# §1669 showed that bottom-up compilation is not an optimisation but a requirement: the
# same eighteen linear maps score -42.99% fitted naively and 54.28% fitted with everything
# below them already substituted. §546's opposite finding (refitting made a two-block TABLE
# substitution worse) does not survive to this scale, or does not survive the change of
# family -- unresolved, and this run settles part of it.
#
# THE PROBLEM WITH THE NUMBERS I HAVE BEEN COMPARING. §1668's whole-stack token-table
# figure of 34.27% was fitted NAIVELY, and §1669 has just shown that naive fitting can be
# catastrophically wrong at eighteen sites. So the table family has never been given the
# treatment the linear family needed to work at all, and "linear 54.28% beats table 34.27%"
# is not yet a fair statement about families -- it may be a statement about compilation.
#
# THREE ARMS, all compiled bottom-up, all scored identically:
#   1. ALL TABLE   -- every site a per-token table (§1661 hybrid hook: table at covered
#                     positions, module live elsewhere)
#   2. ALL LINEAR  -- every site a least-squares linear map. CONTROL: must reproduce
#                     §1669's 54.28%.
#   3. MIXED       -- the best family per band from §1668: token table at mlp0-3 (where the
#                     table beat the linear map by 7.8 points) and linear maps at mlp4-17.
# Arm 3 is the actual object of interest: the cheapest faithful program for this model's
# MLP stack, given what each band turned out to be.
#
# Registered predictions:
#   pred_a COMPILATION HELPS TABLES TOO, so §1668's 34.27% was not a family verdict: the
#          compiled all-table arm exceeds it by >= 10 percentage points.
#   pred_b MIXING BY BAND BEATS EITHER PURE FAMILY: arm 3 exceeds both arm 1 and arm 2.
#   pred_c CONTROL -- the compilation procedure is stable across runs: the compiled
#          all-linear arm lands within 1 point of §1669's 54.28%. Without this the other
#          two arms are not comparable to anything.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'compiled_program_families_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1669_ALL_LINEAR = 0.5428
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
def compile_program(rows, kinds, seen):
    prog = {}
    for L in ALL18:
        prog[L] = fit_site(rows, L, kinds[L], prog, seen)
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
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    seen = seen_mask(fit)
    print(f'COMPILED PROGRAM FAMILIES | all 18 MLPs, bottom-up compilation | ridge {RIDGE} | '
          f'fit skip1200, eval skip7000', flush=True)

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'  CE live {cl:.5f} | all-MLP constant {cc:.5f} | stake {st:.4f} nats', flush=True)

    arms = {
        'all_table': {L: 'table' for L in ALL18},
        'all_linear': {L: 'linear' for L in ALL18},
        'mixed_front_table': {L: ('table' if L in FRONT else 'linear') for L in ALL18},
    }
    out = {}
    for name, kinds in arms.items():
        prog = compile_program(fit, kinds, seen)
        ct = ce(ev, seen, hooks=install(prog))
        ceil = (cc - ct) / st if st > 1e-6 else float('nan')
        out[name] = {'ceiling': round(ceil, 5), 'ce': round(ct, 5),
                     'kinds': {f'mlp{L}': k for L, k in kinds.items()}}
        print(f'  {name:18s} CE {ct:.5f} | CEILING {ceil:7.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    tab = out['all_table']['ceiling']
    lin = out['all_linear']['ceiling']
    mix = out['mixed_front_table']['ceiling']

    pa = (tab - S1668_NAIVE_TABLE) >= 0.10
    pb = (mix > tab) and (mix > lin)
    pc = abs(lin - S1669_ALL_LINEAR) <= 0.01

    print(f'\n  compiled all-table {tab:.2%} vs §1668 NAIVE all-table '
          f'{S1668_NAIVE_TABLE:.2%} -> compilation helps tables too {pa}', flush=True)
    print(f'  MIXED (table at mlp0-3, linear at mlp4-17) {mix:.2%}  vs all-table {tab:.2%} '
          f'vs all-linear {lin:.2%} -> mixing wins {pb}', flush=True)
    print(f'  CONTROL all-linear {lin:.2%} vs §1669 {S1669_ALL_LINEAR:.2%} -> stable {pc}',
          flush=True)
    print(f'  (band ceilings that motivated the mix, §1668: front token '
          f'{S1668_BANDS["front_token"]:.2%} > front linear {S1668_BANDS["front_linear"]:.2%})',
          flush=True)

    res = {'config': {'sites': ALL18, 'front_sites': FRONT, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'compilation': 'bottom-up -- site L fitted with sites below already substituted (§1669)',
                      'table_substitution': 'HYBRID -- table at covered positions, module live elsewhere (§1661)',
                      'scoring': 'covered positions only',
                      's1669_all_linear': S1669_ALL_LINEAR,
                      's1668_naive_all_table': S1668_NAIVE_TABLE,
                      's1668_band_ceilings': S1668_BANDS},
           'stake': round(st, 5), 'ce_live': round(cl, 5), 'ce_const': round(cc, 5),
           'arms': out,
           'predictions': {'pred_a_compilation_helps_tables_ge_10pts': bool(pa),
                           'pred_b_mixing_beats_both_pure_families': bool(pb),
                           'pred_c_control_all_linear_stable': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
