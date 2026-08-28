# HOW MUCH IS LEFT INSIDE THE CLASS CODEX JUST PRUNED?
#
# §1765/§1766 established, and Codex independently derived, that a fully-installed table+linear
# program is a pure function of the current token: every cross-position Jacobian is zero, an
# earlier-position poke moves later covered loss by 0.118/0.072 nat in the live model and EXACTLY 0
# in the installed program. Codex froze the consequence as CONTEXTUAL_COMPILER_NO_GO_AND_GATE.md and
# pruned "rank or degree sweeps inside the old position-wise grammar" as mathematically pruned.
#
# That pruning is correct about CONTEXT and it is a derivation. What it does not say is a NUMBER:
# how much fidelity is still unclaimed inside the position-wise class, and how much of the gap to the
# live model is context that no such program can ever reach. Those are different quantities and the
# second is the one that justifies the prune quantitatively.
#
# The ceiling of the class is the best possible map from the current token to a next-token
# distribution. Two estimates, both computed on the EVAL rows themselves so they are optimistic --
# which is what a ceiling should be:
#   LOO       leave-one-out bigram: every position is predicted from every OTHER position in the same
#             eval role, with unigram backoff where the current token occurs once. Optimistic but not
#             degenerate.
#   2FOLD     fit on half the rows, score the other half, both ways. Honestly achievable from ~24k
#             tokens of the eval distribution, so a looser and more realistic ceiling.
#   MLE       in-sample maximum likelihood, reported ONLY as the degenerate extreme: a token seen once
#             gets probability 1 and contributes zero loss, so this is not a usable bound and is shown
#             to make the optimism of the others legible.
#
# NO MODEL IS LOADED. This is counting, runs on CPU in seconds, and takes no GPU while Codex holds it.
#
# ROLES. Both eval roles reported, scored on covered positions from 64 to match every published
# figure. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a THE CLASS IS NEARLY EXHAUSTED: the best 36-site program's covered CE on skip11000,
#          6.57289, is within 1.0 nat of the LOO position-wise ceiling. If TRUE, Codex's prune costs
#          almost nothing in fidelity and is justified by measurement as well as by derivation. If
#          FALSE there is real headroom left inside the position-wise grammar and the prune, though
#          right about context, leaves fidelity on the table -- which someone should know before
#          abandoning the class.
#   pred_b CONTEXT IS THE DOMINANT TERM: the LOO ceiling sits at least 2.0 nats above the live
#          model's 3.09711. Scored independently of pred_a. If FALSE, a per-token function can get
#          most of the way to the model and "attention is where the work is" is overstated.
#   pred_c POSITIVE CONTROL ON THE CEILING: the LOO ceiling is BELOW the best program's 6.57289. An
#          oracle fitted on the eval role must beat an out-of-sample program; if it does not, the two
#          numbers are not on the same scale and neither pred_a nor pred_b means anything.
#   pred_d CONTROLS: the fit-row bigram at alpha 0.01 reproduces §1766's 7.88804 (skip7000) and
#          7.90729 (skip11000) within 0.001 -- the same object rebuilt in a second script with no
#          model in the process -- and coverage is exactly 5419 of 50257.
import json, time, sys, os, torch

T = 256; V = 50257
ALPHA = 0.01
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/position_wise_ceiling_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
NCOV = 5419
LIVE_CE = {'skip7000': 3.29205, 'skip11000': 3.09711}
BEST_PROGRAM_CE = {'skip7000': 7.35114 - 0.77602, 'skip11000': 7.35825 - 0.78536}
ALL_TABLED_CE = {'skip7000': 7.35114, 'skip11000': 7.35825}
S1766_FIT_BIGRAM = {'skip7000': 7.88804, 'skip11000': 7.90729}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def pairs(rows, seen):
    """(current, next) over SCORED positions only: index >= 64 and the current token covered."""
    cur = rows[:, 64:T].reshape(-1).long()
    nxt = rows[:, 65:T + 1].reshape(-1).long()
    keep = seen[cur]
    return cur[keep], nxt[keep]


def bigram_ce(fit_cur, fit_nxt, ev_cur, ev_nxt, alpha):
    """CE of an add-alpha bigram with unigram backoff, fitted on (fit_*) and scored on (ev_*)."""
    uni = torch.zeros(V, dtype=torch.float64).index_add_(
        0, fit_nxt, torch.ones(fit_nxt.numel(), dtype=torch.float64))
    back = (uni + 1.0) / (uni.sum() + V)
    key = fit_cur * V + fit_nxt
    uniq, inv = torch.unique(key, return_inverse=True)
    cnt = torch.zeros(uniq.numel(), dtype=torch.float64).index_add_(
        0, inv, torch.ones(inv.numel(), dtype=torch.float64))
    ctx = torch.zeros(V, dtype=torch.float64).index_add_(
        0, fit_cur, torch.ones(fit_cur.numel(), dtype=torch.float64))
    lut = {int(k): float(c) for k, c in zip(uniq, cnt)}
    ec = ctx[ev_cur]
    joint = torch.tensor([lut.get(int(a) * V + int(b), 0.0)
                          for a, b in zip(ev_cur, ev_nxt)], dtype=torch.float64)
    p = (joint + alpha * V * back[ev_nxt]) / (ec + alpha * V)
    return float(-torch.log(p.clamp_min(1e-300)).mean())


def loo_ce(cur, nxt):
    """Leave-one-out bigram: each position predicted from every OTHER position in the same role.

    A ceiling on ANY function of the current token, estimated on the eval role itself, so optimistic
    by construction -- which is what a ceiling is for. Where the current token occurs once there is
    no leave-one-out conditional, so the position backs off to the leave-one-out UNIGRAM.
    """
    uni = torch.zeros(V, dtype=torch.float64).index_add_(
        0, nxt, torch.ones(nxt.numel(), dtype=torch.float64))
    ctx = torch.zeros(V, dtype=torch.float64).index_add_(
        0, cur, torch.ones(cur.numel(), dtype=torch.float64))
    key = cur * V + nxt
    uniq, inv = torch.unique(key, return_inverse=True)
    cnt = torch.zeros(uniq.numel(), dtype=torch.float64).index_add_(
        0, inv, torch.ones(inv.numel(), dtype=torch.float64))
    n = nxt.numel()
    j_loo = cnt[inv] - 1.0                      # joint count excluding this position
    c_loo = ctx[cur] - 1.0                      # context count excluding this position
    u_loo = (uni[nxt] - 1.0 + 1.0) / (n - 1.0 + V)   # LOO unigram, add-one smoothed
    has_ctx = c_loo > 0
    p = torch.where(has_ctx,
                    (j_loo + ALPHA * V * u_loo) / (c_loo + ALPHA * V),
                    u_loo)
    return float(-torch.log(p.clamp_min(1e-300)).mean()), int(has_ctx.sum()), n


def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen = torch.zeros(V, dtype=torch.bool)
    seen[fit[:, :T].reshape(-1).long()] = True
    ncov = int(seen.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    fc, fn = fit[:, :T].reshape(-1).long(), fit[:, 1:T + 1].reshape(-1).long()
    print(f'POSITION-WISE CEILING | no model loaded, CPU only | coverage {ncov} of {V} | '
          f'DISCOVERY ONLY', flush=True)

    out = {}
    for ename, epath in EVAL_SETS:
        ev = load(epath)
        cur, nxt = pairs(ev, seen)
        loo, n_ctx, n = loo_ce(cur, nxt)
        half = ev.shape[0] // 2
        a_c, a_n = pairs(ev[:half], seen)
        b_c, b_n = pairs(ev[half:], seen)
        two = 0.5 * (bigram_ce(a_c, a_n, b_c, b_n, ALPHA) + bigram_ce(b_c, b_n, a_c, a_n, ALPHA))
        mle = bigram_ce(cur, nxt, cur, nxt, 1e-12)
        fitb = bigram_ce(fc, fn, cur, nxt, ALPHA)
        out[ename] = {'scored_positions': n, 'with_loo_context': n_ctx,
                      'LOO_ceiling': round(loo, 5), 'twofold_ceiling': round(two, 5),
                      'in_sample_MLE_degenerate': round(mle, 5),
                      'fit_row_bigram': round(fitb, 5),
                      'live_ce': LIVE_CE[ename], 'best_program_ce': round(BEST_PROGRAM_CE[ename], 5),
                      'all_tabled_ce': ALL_TABLED_CE[ename]}
        o = out[ename]
        print(f'\n  {ename}: {n} scored positions, {n_ctx} ({n_ctx / n:.1%}) have a '
              f'leave-one-out conditional', flush=True)
        print(f'    live model            {o["live_ce"]:.5f}', flush=True)
        print(f'    in-sample MLE         {o["in_sample_MLE_degenerate"]:.5f}  '
              f'(degenerate, not a bound)', flush=True)
        print(f'    LOO ceiling           {o["LOO_ceiling"]:.5f}', flush=True)
        print(f'    2-fold ceiling        {o["twofold_ceiling"]:.5f}', flush=True)
        print(f'    best 36-site program  {o["best_program_ce"]:.5f}', flush=True)
        print(f'    all-tabled baseline   {o["all_tabled_ce"]:.5f}', flush=True)
        print(f'    fit-row bigram        {o["fit_row_bigram"]:.5f}  '
              f'(§1766 {S1766_FIT_BIGRAM[ename]})', flush=True)
        del ev

    ho = 'skip11000'
    gap_to_ceiling = out[ho]['best_program_ce'] - out[ho]['LOO_ceiling']
    gap_to_live = out[ho]['LOO_ceiling'] - LIVE_CE[ho]
    pa = gap_to_ceiling <= 1.0
    pb = gap_to_live >= 2.0
    pc = out[ho]['LOO_ceiling'] < out[ho]['best_program_ce']
    pd = (all(abs(out[e]['fit_row_bigram'] - v) <= 0.001 for e, v in S1766_FIT_BIGRAM.items())
          and ncov == NCOV)

    print(f'\n  best program is {gap_to_ceiling:+.4f} from the LOO position-wise ceiling '
          f'-> class nearly exhausted (<=1.0) {pa}', flush=True)
    print(f'  the ceiling is {gap_to_live:+.4f} above the live model -> context is the dominant '
          f'term (>=2.0) {pb}', flush=True)
    print(f'  the eval-fitted oracle beats the out-of-sample program -> scales agree {pc}',
          flush=True)
    print(f'  §1766 fit-row bigram reproduced with no model in the process + coverage {ncov} -> '
          f'control {pd}', flush=True)

    r = {'config': {'alpha': ALPHA, 'scoring': 'covered positions from 64, matching every published '
                                               'figure', 'model_loaded': False,
                    'ceilings': 'LOO and 2-fold are estimated ON the eval role, so both are '
                                'OPTIMISTIC -- a ceiling on what any function of the current token '
                                'can achieve there. The in-sample MLE is degenerate (a token seen '
                                'once gets probability 1) and is reported only to make that optimism '
                                'legible.',
                    'WHY': 'Codex pruned rank and degree sweeps inside the position-wise grammar as '
                           'mathematically pruned (CONTEXTUAL_COMPILER_NO_GO_AND_GATE.md). Correct '
                           'about context, but silent on how much fidelity remains unclaimed inside '
                           'the class. This measures it.',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'results': out,
         'gap_best_program_to_LOO_ceiling': round(gap_to_ceiling, 5),
         'gap_LOO_ceiling_to_live': round(gap_to_live, 5),
         'predictions': {'pred_a_class_nearly_exhausted': bool(pa),
                         'pred_b_context_dominates': bool(pb),
                         'pred_c_oracle_beats_program': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush()


if __name__ == '__main__':
    main()
