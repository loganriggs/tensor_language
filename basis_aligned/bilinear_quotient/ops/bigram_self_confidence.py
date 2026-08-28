# DOES THE BIGRAM KNOW WHEN IT IS RIGHT?  -- the question §1796 ended on.
#
# §1796 established two things. The complementarity is real: an oracle taking whichever arm is correct
# reaches 45.97 / 48.22 / 47.52% on the head, +6.96 / +7.80 / +8.61 pp above the better single arm. And
# NONE of it is reachable from the program's own confidence: selecting a logit-margin threshold on
# skip7000 chose tau=0, never defer, and deferring on the program's least-confident 32% of positions
# already costs accuracy. The program is better than the bigram even where it is least sure.
#
# But the deferral signal there came from the arm doing the deferring. The arm being deferred TO has
# its own, equally realizable confidence: the leave-one-out COUNT behind its argmax. A bigram whose
# choice rests on 12 observations is in a different position from one resting on 1, and unlike the
# program's margin that quantity is a direct measure of evidence rather than of a decision boundary.
#
# SELECTOR: take the bigram when the leave-one-out count behind ITS argmax is >= tau, else the program.
# tau ranges over (never-defer, 20, 12, 8, 5, 3, 2, 1), is chosen on skip7000 ALONE by overall accuracy,
# and is applied unchanged to skip11000 and skip1200, so pred_a and pred_c are scored only on roles the
# threshold never saw. The never-defer entry recovers the program alone and is the null.
#
# ROLES. skip7000 (selection), skip11000 + skip1200 (held out); full-rank settled program; leak-free
# LOO bigram per §1795, built PER ROLE. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39:
#   pred_a EVIDENCE BEATS A DECISION BOUNDARY: at the tau chosen on skip7000, the selector beats the
#          program alone on BOTH held-out roles by at least 0.5 percentage points. If FALSE the
#          complementarity is unreachable from EITHER arm's self-assessment, and §1796's oracle
#          headroom is an upper bound that no realizable rule built from these two objects approaches
#          -- a stronger and more useful negative than §1796's alone.
#   pred_b THE PROCEDURE ACTUALLY CHOSE TO DEFER: the selected tau is finite and defers on more than 1%
#          of skip7000's positions. Scored separately because §1796's selection degenerated to
#          never-defer, and a second degeneration would mean pred_a failed for want of any candidate
#          rather than because the signal is uninformative -- a different conclusion from the same
#          boolean.
#   pred_c AND THE GAIN IS ON THE HEAD: the selector beats the program by at least 1pp on the 125+
#          bucket of both held-out roles. The head is where the two arms are closest and where §1796
#          located most of the oracle headroom, so a gain that appears only in the tail would be the
#          fallback improving rather than the arms combining.
#   pred_d CONTROLS, cross-run per LESSON 42: program and live top-1 reproduce §1789's PUBLISHED
#          0.1355 / 0.1425 / 0.1364 and 0.3932 / 0.4235 / 0.3888 within 0.001; the leak-free bigram
#          reproduces §1795's 0.1244 / 0.1288 / 0.1225; the oracle union reproduces §1796's 0.45966 /
#          0.48224 / 0.47517 within 0.001, which checks that the bigram arm here is the same object
#          despite now also returning its count; tau is selected on skip7000 only; coverage 5419.
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
OUT = PT + 'ops/bigram_self_confidence_results.json'
S1795_BG = {'skip7000': 0.12440, 'skip11000': 0.12880, 'skip1200': 0.12250}
TAUS = (10 ** 9, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0)  # DEFER when the bigram's LOO count >= tau
S1796_UNION = {'skip7000': 0.45966, 'skip11000': 0.48224, 'skip1200': 0.47517}
PICK_ROLE = 'skip7000'
ALPHA = 0.01
S1767_FITBIGRAM_CE = {'skip7000': 7.88804, 'skip11000': 7.90729}
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1789_PROG = {'skip7000': 0.1355, 'skip11000': 0.1425, 'skip1200': 0.1364}
S1789_LIVE = {'skip7000': 0.3932, 'skip11000': 0.4235, 'skip1200': 0.3888}
S1790_LOOBG = {'skip7000': 0.1597, 'skip11000': 0.1663, 'skip1200': 0.1800}
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
def bigram_ce(rows, lp_table):
    """Covered-position CE of the fit-row bigram, to control that it is §1767's object."""
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV)[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        cov = COV['seen'][idx]
        r = COV['idmap'][idx].clamp(min=0)
        lp = lp_table[r].gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        tot += float((-lp.double())[cov].sum()); cnt += int(cov.sum())
    return tot / cnt


@torch.no_grad()
def selector_stats(rows, hooks, loo_argmax):
    """Can the program and the bigram be COMBINED, or is their complementarity unusable?

    §1791 measured an oracle union +10 to +12pp above either arm on the head. An oracle is not a
    program. The realizable test is whether the program's OWN confidence -- its top-1 logit margin --
    says when to defer. Every tau in TAUS is evaluated in the same pass; the choice among them is made
    on PICK_ROLE alone and applied unchanged to the other two."""
    a = {'n': 0, 'prog': 0, 'bg': 0, 'live': 0, 'union': 0,
         'sel': {t: 0 for t in TAUS}, 'defer': {t: 0 for t in TAUS}}
    head = {k: (dict(v) if isinstance(v, dict) else v) for k, v in a.items()}
    head['sel'] = {t: 0 for t in TAUS}
    head['defer'] = {t: 0 for t in TAUS}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        cur = idx[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        ph = forward_logits(idx, hooks)[:, 64:].argmax(-1) == tg
        bpred, bcount = loo_argmax(cur, tg)
        bh = bpred == tg
        margin = bcount              # the DEFERRING signal is now the bigram's own evidence
        lh = forward_logits(idx)[:, 64:].argmax(-1) == tg
        hd = COV['freq'][tg] >= 125
        for d, m in ((a, torch.ones_like(hd)), (head, hd)):
            d['n'] += int(m.sum())
            d['prog'] += int(ph[m].sum()); d['bg'] += int(bh[m].sum())
            d['live'] += int(lh[m].sum()); d['union'] += int((ph | bh)[m].sum())
            for t in TAUS:
                use_bg = margin >= t
                d['sel'][t] += int(torch.where(use_bg, bh, ph)[m].sum())
                d['defer'][t] += int(use_bg[m].sum())
        torch.cuda.empty_cache()
    out = {}
    for name, d in (('overall', a), ('head', head)):
        n = max(d['n'], 1)
        out[name] = {'n': d['n'], 'acc_prog': d['prog'] / n, 'acc_bg': d['bg'] / n,
                     'acc_live': d['live'] / n, 'acc_union': d['union'] / n,
                     'sel': {str(t): d['sel'][t] / n for t in TAUS},
                     'defer_share': {str(t): d['defer'][t] / n for t in TAUS}}
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

    # ---- arm 3: the FIT-ROW bigram, the fair floor (same rows the program was fitted on).
    # add-alpha with unigram backoff, alpha 0.01 as selected in §1767; uncovered current tokens
    # fall back to the unigram argmax, which is what a bigram-with-backoff actually predicts there.
    cur = fit[:, :T].reshape(-1).long()
    nxt = fit[:, 1:T + 1].reshape(-1).long()
    idmap = torch.full((V,), -1, dtype=torch.long)
    idmap[seen_cpu.nonzero(as_tuple=True)[0]] = torch.arange(ncov)
    ri = idmap[cur]
    keep = ri >= 0
    cnts = torch.zeros(ncov, V, dtype=torch.float32, device=DEV)
    cnts.index_put_((ri[keep].to(DEV), nxt[keep].to(DEV)),
                    torch.ones(int(keep.sum()), device=DEV), accumulate=True)
    uni = cnts.sum(0)
    back = torch.softmax(torch.log((uni + 1.0) / (uni.sum() + V)), 0).unsqueeze(0)
    pfit = (cnts + ALPHA * V * back) / (cnts.sum(1, keepdim=True) + ALPHA * V)
    COV['fitbg_ce_table'] = torch.log(pfit.clamp_min(1e-30))
    COV['back'] = back
    bg_arg = pfit.argmax(-1)
    fitbg = torch.full((V,), int(uni.argmax()), dtype=torch.long, device=DEV)
    fitbg[seen_cpu.nonzero(as_tuple=True)[0].to(DEV)] = bg_arg
    COV['fitbg'] = fitbg
    COV['idmap'] = idmap.to(DEV)
    print(f'  fit-row bigram: {int(keep.sum())} observations over {ncov} covered current types; '
          f'{int((cnts.sum(1) > 0).sum())} types observed', flush=True)
    print(f'ACCURACY BY TARGET FREQUENCY | buckets {BUCKETS} on the fit-row count of the TRUE '
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

    # ---- arm 4: the EVAL-ROW LEAVE-ONE-OUT bigram -- NOT a fair floor but an upper bound on
    # what any bigram could do here, since it is fitted on the very rows it is scored on. Leave-one-out
    # is exact: take the top-2 counts and, when the top is the target itself, decrement and re-compare.
    loo_state = {}

    def build_loo(rows, ename):
        """LOO bigram counts on THIS role's own rows only -- no borrowing across roles."""
        c = torch.zeros(ncov + 1, V, dtype=torch.float32, device=DEV)
        cu = rows[:, :-1][:, 64:].reshape(-1).long().to(DEV)
        nx = rows[:, 1:][:, 64:].reshape(-1).long().to(DEV)
        r = COV['idmap'][cu]
        r = torch.where(r >= 0, r, torch.full_like(r, ncov))
        c.index_put_((r, nx), torch.ones(r.numel(), device=DEV), accumulate=True)
        loo_state['cnt'] = c
        torch.cuda.empty_cache()
        print(f'  eval-row LOO bigram for {ename}: {r.numel()} observations', flush=True)

    def loo_argmax(cur, tg):
        """LEAK-FREE argmax of the eval-row bigram with this position's own observation removed.

        §1794 found §1790's version held the target on a TIE: it compared `c1 >= v1` after
        decrementing only the top-1 slot, so whenever the target was top-1 and its leave-one-out count
        merely tied the runner-up, the removed observation still decided the prediction. 30.4 / 30.6 /
        41.1% of that arm's hits were held that way. Here the target's own cell is decremented and the
        argmax is taken over the whole row, with the unigram breaking ties (0.5*back < 0.5 can never
        reorder distinct integer counts)."""
        r = COV['idmap'][cur]
        r = torch.where(r >= 0, r, torch.full_like(r, ncov))
        c = loo_state['cnt'][r]
        own = c.gather(-1, tg.unsqueeze(-1))
        c.scatter_(-1, tg.unsqueeze(-1), own - 1.0)
        pred = (c + 0.5 * COV['back']).argmax(-1)
        cnt = c.gather(-1, pred.unsqueeze(-1)).squeeze(-1)   # evidence behind its own choice
        del c
        return pred, cnt

    res = {}
    fr = program_rows(None)
    hooks = [(st, row_hook(fr[st])) for st in sites]
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        build_loo(ev, ename)
        c = selector_stats(ev, hooks, loo_argmax)
        del loo_state['cnt']
        torch.cuda.empty_cache()
        res[ename] = c
        for nm in ('overall', 'head'):
            x = c[nm]
            print(f'\n  {ename} {nm}: n {x["n"]}  prog {x["acc_prog"]:.2%}  bigram '
                  f'{x["acc_bg"]:.2%}  ORACLE UNION {x["acc_union"]:.2%}  live {x["acc_live"]:.2%}',
                  flush=True)
            print('    selector by program margin: ' + '  '.join(
                f'tau {t} {x["sel"][str(t)]:.2%} (defer {x["defer_share"][str(t)]:.0%})'
                for t in TAUS), flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    top, bot = f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}', f'{BUCKETS[0][0]}-{BUCKETS[0][1]}'
    ho = [e for e in roles if e != PICK_ROLE]
    px = res[PICK_ROLE]['overall']['sel']
    tau = max(TAUS, key=lambda t: px[str(t)])
    pa = all(res[e]['overall']['sel'][str(tau)] - res[e]['overall']['acc_prog'] >= 0.005
             for e in ho)
    pb = (tau < 10 ** 9
          and res[PICK_ROLE]['overall']['defer_share'][str(tau)] > 0.01)
    pc = all(res[e]['head']['sel'][str(tau)] - res[e]['head']['acc_prog'] >= 0.01 for e in ho)
    fitce = {e: bigram_ce(load(p), COV['fitbg_ce_table']) for e, p, _ in EVAL_SETS
             if e in S1767_FITBIGRAM_CE}
    pd = (all(abs(res[e]['overall']['acc_prog'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['overall']['acc_live'] - S1789_LIVE[e]) <= 0.001
              and abs(res[e]['overall']['acc_bg'] - S1795_BG[e]) <= 0.001
              and abs(res[e]['head']['acc_union'] - S1796_UNION[e]) <= 0.001 for e in roles)
          and all(abs(fitce[e] - v) <= 0.01 for e, v in S1767_FITBIGRAM_CE.items())
          and ncov == NCOV)

    print(f'\n  count threshold chosen on {PICK_ROLE} = {tau} (defer share '
          f'{res[PICK_ROLE]["overall"]["defer_share"][str(tau)]:.0%}); applied unchanged to {ho}',
          flush=True)
    print(f'  the SELECTOR beats the program alone on held-out roles by >=0.5pp -> {pa}  '
          + '  '.join(
              f'{e} {res[e]["overall"]["sel"][str(tau)]:.2%} vs {res[e]["overall"]["acc_prog"]:.2%} '
              f'({100*(res[e]["overall"]["sel"][str(tau)] - res[e]["overall"]["acc_prog"]):+.2f}pp)'
              for e in ho), flush=True)
    print(f'  the procedure did NOT choose never-defer -> {pb}  tau {tau} defer '
          f'{res[PICK_ROLE]["overall"]["defer_share"][str(tau)]:.1%}', flush=True)
    print(f'  ... and it gains >=1pp on the HEAD held out -> {pc}  ' + '  '.join(
        f'{e} {res[e]["head"]["sel"][str(tau)]:.2%} vs {res[e]["head"]["acc_prog"]:.2%} '
        f'({100*(res[e]["head"]["sel"][str(tau)] - res[e]["head"]["acc_prog"]):+.2f}pp)'
        for e in ho), flush=True)
    print(f'  accuracies reproduce §1789/§1790, fit-bigram CE reproduces §1767 ' + '  '.join(
        f'{e} {v:.5f}' for e, v in fitce.items()) + f', coverage {ncov} -> control {pd}', flush=True)

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
          'fit_bigram_covered_ce': {e: round(v, 5) for e, v in fitce.items()},
          'count_threshold_chosen_on_' + PICK_ROLE: tau,
          'predictions': {'pred_a_bigram_confidence_beats_program_heldout': bool(pa),
                          'pred_b_procedure_did_not_choose_never_defer': bool(pb),
                          'pred_c_gain_on_the_head_heldout': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
