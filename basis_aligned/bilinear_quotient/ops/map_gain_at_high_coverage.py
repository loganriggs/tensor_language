# DOES THE MAP'S RARE-TARGET GAIN SURVIVE AT 16,110? -- §1933's open question.
#
# §1933 made the first measurement of map rank on the accuracy structure: at 5,419 types, raising §1870's
# covered-fit fallback map from rank 64 to 512 at FIXED full table rank moves the unseen-target
# kept-fraction 2.7 -> 4.0, 6.2 -> 6.7, 3.6 -> 4.0 (+1.3 / +0.5 / +0.4pp) while the 125+ bucket moves only
# 63.5 -> 63.1, 62.9 -> 62.1, 63.4 -> 62.8. §1870, §1877 and §1880 had priced map rank purely in CE.
#
# That was one coverage. §1924 is a live example of a map-rank lever measured at 16,110 that did not
# transfer to 5,419, and §1932/§1933 have just shown this family's SIGNS flip between coverages. So the
# gain is not assumable at 16,110, which is where §1931's best-known build lives.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 16,110 coverage. Rung 3.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# LESSON 72 applied throughout: this section's questions are DIRECTIONAL, so every predicate is a SIGNED
# comparison, not a tolerance on |delta|. §1932's pred_a and §1933's pred_c both passed tolerance bars
# while the sign carried the finding, and I am not writing a third.
#   pred_a THE GAIN SURVIVES, SIGNED: at 16,110 the rank-512 map's unseen-bucket kept-fraction is HIGHER
#          than the rank-64 map's, at fixed full table rank, on all three roles. Not "within" anything --
#          strictly higher. If FALSE the rare-target benefit of map rank is a 5,419 phenomenon and §1933's
#          "second reason to spend on map rank" needs the coverage attached, exactly as §1924 forced on
#          §1919.
#   pred_b AND IT IS MATERIAL: the gain is at least 0.2pp on at least 2 of 3 roles, against §1933's
#          PUBLISHED +1.3 / +0.5 / +0.4pp. A one-sided bar on a quantity whose direction pred_a already
#          fixes, so it tests size without pretending to test direction.
#   pred_c AND IT IS CONFINED TO THE RARE END: the 125+ kept-fraction does NOT rise by more than 0.2pp on
#          any role -- one-sided, because §1933 found the map costs a little there (63.5 -> 63.1 etc.) and
#          the claim under test is that its benefit is rare-target-specific. If FALSE the map helps both
#          buckets at this coverage and it is simply a better fallback, which would be a cleaner and more
#          useful statement than the targeted one.
#   pred_d CONTROLS: coverage is exactly 16,110; the rank-64 arm reproduces §1883's PUBLISHED deployed
#          figures at this coverage -- 125+ 53.6 / 54.1 / 53.9% and unseen 2.6 / 4.9 / 3.5% -- within
#          0.5pp; buckets partition; and the LIVE per-bucket accuracy is identical across arms.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = ('map64', 'map512')
MAPRANK_OF = {'map64': 64, 'map512': 512}
# UNIFORM ranks only -- the question is about the table axis, not the allocation.
ALLOC = {'map64': None, 'map512': None}   # FULL table rank in both arms: only the map differs
S1883_TOP = {'skip7000': 0.536, 'skip11000': 0.541, 'skip1200': 0.539}   # §1883 deployed @16,110
S1883_BOT = {'skip7000': 0.026, 'skip11000': 0.049, 'skip1200': 0.035}   # §1883 deployed @16,110
S1933_GAIN = [0.013, 0.005, 0.004]   # §1933 unseen gain at 5,419

RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/map_gain_at_high_coverage_results.json'
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
# live COVERED-CE refs set to None: 3.29205 / 3.09711 / 3.40277 are the 5,419 covered set's and
# this runs at 16,110. This lineage does not assert on them, but a fork that does would inherit
# the trap that cost §1930 a launch and failed §1905's pred_d.
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', None),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', None),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None)]
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'   # 5,419 types at T=256 -- the DEPLOYED coverage
H = m.transformer.h
NCOV = 16110      # high coverage; §1933 measured the same comparison at the deployed 5,419
S1789_COV = 5419  # below are AT 5419 and are printed for context, never used as bars (§1882's trap)
S1788_ACC = {'skip7000': {'prog': 0.1355, 'live': 0.3932},
             'skip11000': {'prog': 0.1425, 'live': 0.4235},
             'skip1200': {'prog': 0.1364, 'live': 0.3888}}
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
def compare_by_bucket(rows, hooks):
    """Top-1 accuracy of live and program, split by the TRUE TARGET token's fit-row frequency.

    The bucket axis is the TARGET, not the current token: the program is keyed on the current token,
    so this asks what it can PRODUCE rather than what it can condition on."""
    a = {b: {'acc_l': 0, 'acc_p': 0, 'n': 0} for b in BUCKETS}
    tot = {'acc_l': 0, 'acc_p': 0, 'n': 0}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        al = forward_logits(idx)[:, 64:].argmax(-1)
        ap = forward_logits(idx, hooks)[:, 64:].argmax(-1)
        cl, cp = (al == tg), (ap == tg)
        f = COV['freq'][tg]
        tot['acc_l'] += int(cl.sum()); tot['acc_p'] += int(cp.sum()); tot['n'] += int(tg.numel())
        for b in BUCKETS:
            msk = (f >= b[0]) & (f <= b[1])
            a[b]['acc_l'] += int(cl[msk].sum()); a[b]['acc_p'] += int(cp[msk].sum())
            a[b]['n'] += int(msk.sum())
    assert sum(a[b]['n'] for b in BUCKETS) == tot['n'], 'buckets do not partition the positions'
    out = {'overall': {'top1_acc_live': tot['acc_l'] / tot['n'],
                       'top1_acc_prog': tot['acc_p'] / tot['n'], 'n': tot['n']}}
    for b in BUCKETS:
        n = max(a[b]['n'], 1)
        out[f'{b[0]}-{b[1]}'] = {'top1_acc_live': a[b]['acc_l'] / n,
                                 'top1_acc_prog': a[b]['acc_p'] / n,
                                 'kept_fraction': (a[b]['acc_p'] / n) / max(a[b]['acc_l'] / n, 1e-9),
                                 'n': a[b]['n']}
    return out


@torch.no_grad()
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
    print(f'MAP RARE-TARGET GAIN AT 16110 | buckets {BUCKETS} on the fit-row count of the TRUE '
          f'target | full-rank settled program | DISCOVERY ONLY', flush=True)

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
        a = ALLOC[r]
        if a is None:
            tc = tables
        else:
            tc = {}
            for st, tbl in tables.items():
                rk = a[st[0]]
                b = tbl.double()
                mu = b.mean(0, keepdim=True)
                U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
                tc[st] = (mu + (U[:, :rk] * S[:rk]) @ Vh[:rk]).float()
        out = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tc[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mr = MAPRANK_OF[r]
            mp = (U[:, :mr] * S[:mr]) @ Vh[:mr]
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tc[st]
            fr[unc] = (Eunc @ mp).float()
            out[st] = fr
        return out

    res = {}
    for r in RANKS:
        key = str(r)
        fr = program_rows(r)
        hooks = [(st, row_hook(fr[st])) for st in sites]
        res[key] = {}
        print(f'\n  === table rank {key}, map rank {MAPRANK_OF[r]} ===', flush=True)
        for ename, epath, ce_ref in EVAL_SETS:
            ev = load(epath)
            c = compare_by_bucket(ev, hooks)
            res[key][ename] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                                   for kk, vv in v.items()} for k, v in c.items()}
            print(f'  {ename}: overall live {c["overall"]["top1_acc_live"]:.2%} '
                  f'prog {c["overall"]["top1_acc_prog"]:.2%}', flush=True)
            for b in BUCKETS:
                k = f'{b[0]}-{b[1]}'
                x = c[k]
                print(f'    target fit-count {k:12s} n {x["n"]:6d}  live {x["top1_acc_live"]:6.2%}  '
                      f'prog {x["top1_acc_prog"]:6.2%}  kept {x["kept_fraction"]:6.1%}', flush=True)
            ev = None
            torch.cuda.empty_cache()
        fr, hooks = None, None
        torch.cuda.empty_cache()
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    top, bot = f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}', f'{BUCKETS[0][0]}-{BUCKETS[0][1]}'
    def kept(arm, e, b):
        return res[arm][e][b]['kept_fraction']
    gain = {e: kept('map512', e, bot) - kept('map64', e, bot) for e in roles}
    topmv = {e: kept('map512', e, top) - kept('map64', e, top) for e in roles}
    livespread = max(abs(res['map512'][e][b]['top1_acc_live'] - res['map64'][e][b]['top1_acc_live'])
                     for e in roles for b in [f'{x}-{y}' for x, y in BUCKETS] + ['overall'])
    partition = all(sum(res[a2][e][f'{x}-{y}']['n'] for x, y in BUCKETS)
                    == res[a2][e]['overall']['n'] for a2 in RANKS for e in roles)
    pa = all(gain[e] > 0 for e in roles)
    pb = sum(1 for e in roles if gain[e] >= 0.002) >= 2
    pc = all(topmv[e] <= 0.002 for e in roles)
    pd = (ncov == NCOV and partition and livespread <= 1e-9
          and all(abs(kept('map64', e, top) - S1883_TOP[e]) <= 0.005 for e in roles)
          and all(abs(kept('map64', e, bot) - S1883_BOT[e]) <= 0.005 for e in roles))

    print(f'\n  kept-fraction at FULL table rank, map 64 vs 512, {ncov} types:', flush=True)
    for i9, e in enumerate(roles):
        print(f'    {e:10s} 125+   map64 {kept("map64", e, top):.1%}  map512 '
              f'{kept("map512", e, top):.1%}  ({topmv[e] * 100:+.2f}pp)', flush=True)
        print(f'    {"":10s} unseen map64 {kept("map64", e, bot):.1%}  map512 '
              f'{kept("map512", e, bot):.1%}  ({gain[e] * 100:+.2f}pp)   '
              f'(§1933 at 5,419 {S1933_GAIN[i9] * 100:+.1f}pp)', flush=True)
    print(f'\n  the GAIN SURVIVES, signed (unseen higher on 3/3) -> {pa}  ' + '  '.join(
        f'{e} {gain[e] * 100:+.2f}pp' for e in roles), flush=True)
    print(f'  and it is MATERIAL (>= 0.2pp on >= 2 roles) -> {pb}  '
          f'{sum(1 for e in roles if gain[e] >= 0.002)}/3', flush=True)
    print(f'  and it is CONFINED to the rare end (125+ does not rise > 0.2pp) -> {pc}  ' + '  '.join(
        f'{e} {topmv[e] * 100:+.2f}pp' for e in roles), flush=True)
    print(f'  coverage {ncov}, map64 arm reproduces §1883, partitions, LIVE identical -> control {pd}',
          flush=True)

    r2 = {'config': {'table_ranks': ['full' if r is None else str(r) for r in RANKS],
                     'map_rank_of_table_rank': {str(k): v for k, v in MAPRANK_OF.items()},
                     'program': 'context-free tables, output-NN fallback with a rank-64 '
                                'embedding->row map -- the settled design of §1780-§1786',
                     'instruments': 'top-1 agreement with the live model, top-1 accuracy against the '
                                    'true next token, and KL(live || program). All three are NEW to '
                                    'this thread, which has been CE-only since §1747.',
                     'ROLE_NOTE': 'DISCOVERY ONLY; a second-class confirmation with a DIFFERENT '
                                  'instrument, not a replication of the same one.'},
          'results': res,
          'predictions': {'pred_a_gain_survives_signed_program_concentrated_on_frequent_targets': bool(pa),
                          'pred_b_material_more_concentrated_than_live': bool(pb),
                          'pred_c_confined_to_rare_end_keeps_60pc_on_the_top_bucket': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
