# IS IT FREQUENCY, OR IS IT COVERAGE?  -- disentangling §1789's confound.
#
# §1789 split top-1 by the true target's count in the FIT rows and found the program keeps ~63% of the
# live model on targets seen 125+ times and ~3% on targets never seen. §1790 and §1791 both rest on
# that split. But the bucketing variable is CONFOUNDED: a token absent from the fit rows is usually
# also rare in the world, so "the program cannot produce unseen targets" and "the program cannot
# produce rare targets" are not distinguished by anything measured so far.
#
# The cell that separates them is FREQUENT IN A HELD-OUT CORPUS but ABSENT FROM THE FIT ROWS. Target
# frequency is estimated LEAVE-ONE-ROLE-OUT -- scoring skip7000 uses skip11000 and skip1200 as the
# frequency source, and so on -- so the ranking never sees the rows it is applied to. The run asserts
# the scored role is absent from its own source.
#
# CELLS. freq_uncovered = held-out rank < 1000 AND fit-row count 0; freq_covered = held-out rank <
# 1000 AND fit-row count >= 125; rare_any = held-out rank >= 1000; all = every scored position.
#
# ROLES. skip7000, skip11000, skip1200; full-rank settled program. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# with the cross-run control LESSON 42 demands:
#   pred_a THE LIMIT IS COVERAGE: on frequent-but-uncovered targets the program's top-1 is below 5%,
#          at every role. If FALSE the program does generalise to common tokens it never saw at fit
#          time, and §1789's story is about frequency rather than coverage -- which would mean the
#          tail is reachable by a bigger fit set rather than only by context.
#   pred_b AND THE CELL IS NOT INTRINSICALLY HARD: the live model exceeds 25% top-1 on those same
#          positions, at every role. Scored separately because pred_a alone is uninterpretable if the
#          cell is simply unpredictable -- both arms failing would say the cell, not the program.
#   pred_c AND COVERAGE IS WHAT MOVES IT: on frequent-AND-covered targets the program exceeds 30% and
#          is at least 6x its accuracy on the frequent-but-uncovered cell, at every role. Both cells
#          are matched on held-out frequency, so this isolates coverage as the variable.
#   pred_d CONTROLS: overall top-1 reproduces §1789's PUBLISHED 0.1355 / 0.1425 / 0.1364 (program) and
#          0.3932 / 0.4235 / 0.3888 (live) within 0.001 -- the cross-run check of LESSON 42; the
#          scored role never appears in its own frequency source; the three scientific cells are
#          pairwise disjoint and do not exceed the scored positions; coverage is 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None,)
MAP_RANK = 64
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/frequency_vs_coverage_results.json'
TOPRANK = 1000
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1789_PROG = {'skip7000': 0.1355, 'skip11000': 0.1425, 'skip1200': 0.1364}
S1789_LIVE = {'skip7000': 0.3932, 'skip11000': 0.4235, 'skip1200': 0.3888}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def row_hook(full_rows):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = full_rows[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


@torch.no_grad()
def forward_logits(idx, hooks=()):
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
    STATE['idx'] = idx
    try:
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def compare_cells(rows, hooks, hrank):
    """Disentangle FREQUENCY from COVERAGE.

    §1789 bucketed by the target's count in the FIT rows, but that variable is confounded: a token
    absent from the fit rows is usually also rare in the world. The cell that separates them is
    FREQUENT IN A HELD-OUT CORPUS but ABSENT FROM THE FIT ROWS. If the program fails there while the
    live model does not, its limitation is coverage; if it succeeds, §1789's story was about
    frequency after all."""
    CELLS = ('freq_uncovered', 'freq_covered', 'rare_any', 'all')
    a = {c: {'live': 0, 'prog': 0, 'n': 0} for c in CELLS}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        hit = {'live': forward_logits(idx)[:, 64:].argmax(-1) == tg,
               'prog': forward_logits(idx, hooks)[:, 64:].argmax(-1) == tg}
        for k in hit:
            assert hit[k].shape == tg.shape, f'{k} arm is {hit[k].shape}, targets are {tg.shape}'
        fitc = COV['freq'][tg]            # count in the FIT rows
        hr = hrank[tg]                    # rank in the HELD-OUT roles (0 = commonest)
        top = hr < TOPRANK
        msk = {'freq_uncovered': top & (fitc == 0),
               'freq_covered': top & (fitc >= 125),
               'rare_any': ~top,
               'all': torch.ones_like(top)}
        for c in CELLS:
            a[c]['n'] += int(msk[c].sum())
            for k in ('live', 'prog'):
                a[c][k] += int(hit[k][msk[c]].sum())
    # the three scientific cells must be pairwise disjoint and must not exceed the whole
    assert (a['freq_uncovered']['n'] + a['freq_covered']['n'] + a['rare_any']['n']
            <= a['all']['n']), 'cells overlap or exceed the scored positions'
    out = {}
    for c in CELLS:
        n = max(a[c]['n'], 1)
        out[c] = {'acc_live': a[c]['live'] / n, 'acc_prog': a[c]['prog'] / n,
                  'kept': (a[c]['prog'] / n) / max(a[c]['live'] / n, 1e-9), 'n': a[c]['n']}
    return out


def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit[:, :T].reshape(-1).long()] = True
    ncov = int(seen_cpu.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    seen = seen_cpu.to(DEV)
    COV['seen'] = seen
    toks = seen_cpu.nonzero(as_tuple=True)[0]
    tk = toks.to(DEV)
    unc = (~seen).nonzero(as_tuple=True)[0]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    COV['freq'] = torch.bincount(fit[:, 1:T + 1].reshape(-1).long(),
                                 minlength=V).to(DEV)
    print(f'FREQUENCY VS COVERAGE | cells on (held-out rank < {TOPRANK}) x (fit-row count) | '
          f'leave-one-role-out frequency source | DISCOVERY ONLY', flush=True)

    def holdout_rank(exclude):
        """Target-token frequency rank from the OTHER two roles -- never the one being scored."""
        c = torch.zeros(V, dtype=torch.long, device=DEV)
        used = []
        for nm, pth, _ in EVAL_SETS:
            if nm == exclude:
                continue
            r = load(pth)
            c += torch.bincount(r[:, 1:][:, 64:].reshape(-1).long().to(DEV), minlength=V)
            used.append(nm)
            del r
        order = torch.argsort(c, descending=True)
        rk = torch.empty(V, dtype=torch.long, device=DEV)
        rk[order] = torch.arange(V, device=DEV)
        return rk, used, int((c > 0).sum())

    # the settled fallback: output-NN neighbour (§1780/§1781)
    lpc = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        lpc[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
    pcn = torch.softmax(lpc, -1)
    pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
    del lpc
    nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
    nnrow[tk] = torch.arange(ncov, device=DEV)
    for s0 in range(0, unc.numel(), 512):
        u = unc[s0:s0 + 512]
        p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
    del pcn
    torch.cuda.empty_cache()

    tables = {st: torch.zeros(ncov, D, device=DEV) for st in sites}
    cap = {}

    def mk(st):
        def hook(mod, args, out):
            cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
            return None
        return hook
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        forward_logits(t, [(st, mk(st)) for st in sites])
        for st in sites:
            tables[st][i:i + t.shape[0]] = cap[st]
    Ecov = m.transformer.wte.weight.detach()[tk].float().double()
    A = Ecov.T @ Ecov + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (ncov / D)
    Eunc = m.transformer.wte.weight.detach()[unc].float().double()
    print(f'  built the settled fallback and 36 tables ({time.time() - t0:.0f}s)', flush=True)

    def program_rows(r):
        if r is None:
            tc = tables
        else:
            tc = {}
            for st, tbl in tables.items():
                b = tbl.double()
                mu = b.mean(0, keepdim=True)
                U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
                tc[st] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        out = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tc[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mp = (U[:, :MAP_RANK] * S[:MAP_RANK]) @ Vh[:MAP_RANK]
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tc[st]
            fr[unc] = (Eunc @ mp).float()
            out[st] = fr
        return out

    res = {}
    fr = program_rows(None)
    hooks = [(st, row_hook(fr[st])) for st in sites]
    src_used = {}
    for ename, epath, ce_ref in EVAL_SETS:
        hrank, used, ntypes = holdout_rank(ename)
        src_used[ename] = used
        assert ename not in used, f'{ename} leaked into its own frequency source'
        ev = load(epath)
        c = compare_cells(ev, hooks, hrank)
        res[ename] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in c.items()}
        print(f'\n  {ename}: frequency source {used} ({ntypes} target types seen)', flush=True)
        for k in ('all', 'freq_uncovered', 'freq_covered', 'rare_any'):
            x = c[k]
            print(f'    {k:16s} n {x["n"]:6d}  live {x["acc_live"]:6.2%}  prog {x["acc_prog"]:6.2%}'
                  f'  kept {x["kept"]:6.1%}', flush=True)
        del ev, hrank
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    pa = all(res[e]['freq_uncovered']['acc_prog'] < 0.05 for e in roles)
    pb = all(res[e]['freq_uncovered']['acc_live'] > 0.25 for e in roles)
    pc = all(res[e]['freq_covered']['acc_prog'] > 0.30
             and res[e]['freq_covered']['acc_prog']
             >= 6.0 * max(res[e]['freq_uncovered']['acc_prog'], 1e-9) for e in roles)
    pd = (all(abs(res[e]['all']['acc_prog'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['all']['acc_live'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(e not in src_used[e] for e in roles) and ncov == NCOV)

    print(f'\n  on FREQUENT-but-UNCOVERED targets the program is below 5% -> {pa}  ' + '  '.join(
        f'{e} {res[e]["freq_uncovered"]["acc_prog"]:.2%} (n {res[e]["freq_uncovered"]["n"]})'
        for e in roles), flush=True)
    print(f'  ... while the live model exceeds 25% there -> {pb}  ' + '  '.join(
        f'{e} {res[e]["freq_uncovered"]["acc_live"]:.2%}' for e in roles), flush=True)
    print(f'  ... and on FREQUENT-and-COVERED it exceeds 30% and 6x the uncovered cell -> {pc}  '
          + '  '.join(f'{e} {res[e]["freq_covered"]["acc_prog"]:.2%} vs '
                      f'{res[e]["freq_uncovered"]["acc_prog"]:.2%}' for e in roles), flush=True)
    print(f'  overall reproduces §1789, frequency source excludes the scored role, coverage {ncov} '
          f'-> control {pd}', flush=True)

    r2 = {'config': {'table_ranks': ['full' if r is None else str(r) for r in RANKS],
                     'map_rank': MAP_RANK,
                     'program': 'context-free tables, output-NN fallback with a rank-64 '
                                'embedding->row map -- the settled design of §1780-§1786',
                     'instruments': 'top-1 agreement with the live model, top-1 accuracy against the '
                                    'true next token, and KL(live || program). All three are NEW to '
                                    'this thread, which has been CE-only since §1747.',
                     'ROLE_NOTE': 'DISCOVERY ONLY; a second-class confirmation with a DIFFERENT '
                                  'instrument, not a replication of the same one.'},
          'results': res,
          'frequency_source': src_used,
          'predictions': {'pred_a_program_fails_on_frequent_uncovered': bool(pa),
                          'pred_b_live_succeeds_on_frequent_uncovered': bool(pb),
                          'pred_c_program_succeeds_on_frequent_covered': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
