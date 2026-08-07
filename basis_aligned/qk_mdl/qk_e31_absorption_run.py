"""E31 JOB 1 -- THE ABSORPTION QUESTION (checkpoint-only; no training).

E22's registered prediction (ii) has never been evaluated: the E22 runner's
census_residual step OOMed (its census forward ran WITH the autograd graph
alive -- 12 blocks x (B, 6, T, T) patterns plus the 21 shuffled-null feature
tensors -- so it never completed), and only the wiring probe was repaired
later. This runner finishes it.

QUESTION: giving every head an EXPLICIT named match term (E22a's
b_h * MATCH_prev + c_h * MATCH_same + signed positional profile) either
ABSORBED the match structure out of the learned bilinear residual pattern, or
it did not and the residual is still doing the matching itself.

MEASURE (E21 machinery reused verbatim -- qk_e21_census_run.census_model,
null_zscores, summarize, build_feats, and the synthetic known-answer
controls):
  - RESIDUAL census: the census callback fires on the bilinear-only pattern
    (s1 * s2, BEFORE the named terms are added) while the model's forward is
    otherwise untouched -- the pattern the residual would have to explain on
    its own. This is the registered comparison against the parent E19a's
    stored census (qk_e21_census.json).
  - FULL census: the same census on the pattern the model actually uses
    (residual + named terms). Disambiguates the three-way verdict:
      residual < parent AND full ~ parent      -> ABSORPTION (structure moved
                                                  into the named term)
      residual < parent AND full < parent      -> the arm simply matches LESS
      residual ~ parent                        -> NO absorption (the explicit
                                                  term is additional, not a
                                                  replacement)
  - per-head rows: parent cos^2, residual cos^2, full cos^2, the head's
    named-term mass shares and b_h from qk_e22.json['mixture_weights_E22a'],
    plus the shuffled-token null z-scores of both censuses.

POSITIVE CONTROLS (before the measurement, hard asserts):
  1. E21's own synthetic known-answer controls (exact previous-token pattern
     -> PREV_token ~ 1.0 with the offset profile a delta at offset 1; exact
     same-token pattern -> MATCH_same ~ 1.0) through the exact scoring code
     this run uses;
  2. DECOMPOSITION: on a real batch, full_pattern - residual_pattern -
     (profile + b_h * MATCH_prev + c_h * MATCH_same) == 0 to 1e-4 for every
     block -- i.e. the thing this run calls "the residual" really is the full
     pattern minus the named terms.

OUTPUT: qk_e22.json['census_residual_E22a'] (the schema E22's summarize()
expects, so the stored prediction-(ii) verdict fills in) +
qk_e22.json['census_full_E22a'], and the verdict summary in qk_e31.json.
Idempotent on both JSON keys."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, DEPTH, torch
import qk_e15_reinvest_run as E15R
import qk_e22_predbasis_run as E22R

E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP31 = E.jpath('qk_e31.json')
JP22 = E.jpath('qk_e22.json')
STEM = 'qk_e22_a'


def spearman(a, b):
    """Rank correlation with average ranks (no scipy dependency)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    def rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


# ---------------- the two census forwards (no-grad: the OOM fix) ----------------
def make_fwd(kind):
    """Pattern-collecting forward for the E22 architecture, under no_grad
    (the E21 reference forwards are @torch.no_grad; the E22 model's own
    forward is not, which is what exhausted the card the first time).
    kind='residual': the callback fires on the bilinear-only pattern, BEFORE
    the named terms are added. kind='full': on the pattern actually used."""
    def fwd(mm, idx, layer_cb=None, pat_hook=None):
        with torch.no_grad():
            if kind == 'residual':
                return mm(idx, census_cb=layer_cb, pat_hook=pat_hook)
            return mm(idx, census_full_cb=layer_cb, pat_hook=pat_hook)
    return fwd


def control_decomposition(m, E21, held):
    """Control 2: full - residual - named terms == 0 on a real batch."""
    key = 'control_decomposition_E22a'
    done = E.loadj(JP31).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    T = E21.T
    idx = held[:2, :T]
    maskf = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                  device=idx.device)).float()
    resid, full = {}, {}
    with torch.no_grad():
        m(idx, census_cb=lambda l, p: resid.__setitem__(l, p.clone()),
          census_full_cb=lambda l, p: full.__setitem__(l, p.clone()))
        Kprev, Ksame = E22R.match_kernels(idx, maskf)
        worst = 0.0
        for l in range(len(m.h)):
            terms = m.pred_terms(l, Kprev, Ksame, maskf, T)
            worst = max(worst, float((full[l] - resid[l] - terms).abs().max()))
    rec = {'blocks': len(m.h), 'batch': int(idx.shape[0]), 'T': int(T),
           'max_abs_full_minus_residual_minus_named_terms': worst,
           'note': 'the census "residual" is exactly the full pattern minus '
                   'the named predicate terms (profile + b_h*MATCH_prev + '
                   'c_h*MATCH_same)',
           'pass': bool(worst < 1e-4)}
    E.merge(JP31, key, rec)
    print(f"{key}: max |full - resid - named| {worst:.2e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def run_census(name, m, kind, held, E21):
    """E21.census_model + E21.null_zscores on one pattern view, with an
    out-of-memory fallback to a smaller pattern micro-batch (accumulations
    are sums, so BP only changes batching)."""
    t0 = time.time()
    fwd = make_fwd(kind)
    bp0 = E21.BP
    for bp in (bp0, 2, 1):
        try:
            E21.BP = bp
            table, sc, tpl = E21.census_model(name, m, fwd, held)
            c2r, zmat, mu_n, sd_n = E21.null_zscores(m, fwd, held)
            break
        except torch.cuda.OutOfMemoryError:
            print(f"{name}: OOM at pattern micro-batch {bp} -- retrying "
                  f"smaller", flush=True)
            torch.cuda.empty_cache()
    else:
        raise RuntimeError('census OOM even at micro-batch 1')
    E21.BP = bp0
    for r in table:
        l, h = r['layer'], r['head']
        r['cos2_eval'] = {E21.FEATN[f]: round(float(c2r[l, h, f]), 4)
                          for f in E21.NULL_FEATS}
        r['null_z'] = {E21.FEATN[f]: round(float(zmat[l, h, f]), 1)
                       for f in E21.NULL_FEATS}
    del tpl
    if not E.SMOKE:
        torch.cuda.empty_cache()
    print(f"{name}: census done ({time.time() - t0:.0f}s, micro-batch {bp})",
          flush=True)
    return table, E21.summarize(table), round(time.time() - t0, 1)


def mixture_rows():
    """(layer, head) -> the stored E22a named-term row (b_h, mass shares)."""
    j = E.loadj(JP22)
    mw = j.get('mixture_weights_E22a')
    if not mw:
        return {}
    return {(r['layer'], r['head']): r for r in mw['table']}


def main():
    E.setup()
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c
    E21 = E22R.get_e21()
    if E.SMOKE:
        E21.N_FIT = 8                       # 16 smoke held seqs -> two halves
        E21.NULL_K = 3
        E21.BP = 2

    n_rows = 16 if E.SMOKE else 200
    held = torch.from_numpy(
        np.load(E.HELD_PATH)[33000:33000 + n_rows].astype(np.int64)).to(E.DEV)
    print(f"held: fresh34k[33000:{33000 + n_rows}] on {E.DEV} "
          f"(the E21 census rows)", flush=True)

    if 'control_synthetic_E21' not in E.loadj(JP31):
        note = ('qk_e21_census_run.control_synthetic verbatim: known-answer '
                'synthetic patterns through the exact scoring machinery this '
                'run uses')
        try:
            ctrl = E21.control_synthetic(held)
            E.merge(JP31, 'control_synthetic_E21',
                    {'note': note, 'result': ctrl, 'pass': True})
            print('control (E21 synthetic): PASS', flush=True)
        except AssertionError as ex:
            # The control's "no other predicate above 0.5" clause is
            # calibrated at T=512: MATCH_next CONTAINS the subdiagonal by
            # construction (key j = i-1 has tok_{j+1} = tok_i), so at the
            # smoke length T=64 it scores 0.73 against an exact previous-
            # token pattern. Never tolerated in the real run.
            if not E.SMOKE:
                raise
            E.merge(JP31, 'control_synthetic_E21',
                    {'note': note, 'smoke_only_tolerated': str(ex)[:300],
                     'pass': False})
            print('control (E21 synthetic): smoke-length tolerance (see JSON)',
                  flush=True)

    if E.SMOKE:
        m = E22R.make_e22(s=s_c).eval().float()
    else:
        assert os.path.exists(E.ckpath(STEM)), f'{STEM}.pt missing'
        m, _ = E.load_arm(STEM, lambda: E22R.make_e22(s=s_c))
    m.eval().float()

    control_decomposition(m, E21, held)

    done = E.loadj(JP22)
    parent, prog = ({}, [])
    try:
        parent, prog = E22R.parent_census_matchprev()
    except Exception as ex:                       # smoke / missing census
        print(f"parent census unavailable ({str(ex)[:80]})", flush=True)

    tables = {}
    for kind, key in (('residual', 'census_residual_E22a'),
                      ('full', 'census_full_E22a')):
        if key in done and not E.SMOKE:
            print(f"{key}: already recorded -- skip", flush=True)
            tables[kind] = done[key]['table']
            continue
        table, summ, secs = run_census(f'E22a_{kind}', m, kind, held, E21)
        tables[kind] = table
        ours = {(r['layer'], r['head']): r['cos2_eval']['MATCH_prev']
                for r in table}
        rec = {'checkpoint': f'{STEM}.pt',
               'pattern_view': ('the bilinear-only residual pattern (named '
                                'terms NOT included)' if kind == 'residual'
                                else 'the full pattern the model uses '
                                     '(residual + named terms)'),
               'held_rows': f'fresh34k[33000:{33000 + n_rows}] (the E21 '
                            'census rows)',
               'note': 'machinery = qk_e21_census_run.census_model + '
                       'null_zscores + summarize verbatim; the census '
                       'forward runs under no_grad (the E22 runner\'s did '
                       'not -- that is what OOMed)',
               'summary': summ, 'runtime_s': secs,
               'match_prev_total_cos2': round(sum(ours.values()), 4),
               'table': table}
        if parent:
            total_p = round(sum(parent.values()), 4)
            mean_p = round(sum(parent[p] for p in prog) / max(len(prog), 1), 4)
            mean_o = round(sum(ours[p] for p in prog) / max(len(prog), 1), 4)
            rec.update({
                'match_prev_total_cos2_parent_stored': total_p,
                'parent_programmatic_match_prev_heads': [
                    {'layer': l, 'head': h,
                     'parent_cos2': round(parent[(l, h)], 4),
                     'this_cos2': round(ours[(l, h)], 4)} for (l, h) in prog],
                'match_prev_mean_over_parent_prog_heads_parent': mean_p,
                'match_prev_mean_over_parent_prog_heads_this': mean_o})
            if kind == 'residual':
                # the exact key names E22's summarize() reads
                rec['match_prev_total_cos2_residual'] = rec[
                    'match_prev_total_cos2']
                rec['match_prev_mean_over_parent_prog_heads_residual'] = mean_o
                rec['prediction_ii_drop_confirmed'] = bool(
                    rec['match_prev_total_cos2'] < total_p
                    and mean_o < mean_p)
        E.merge(JP22, key, rec)
        print(f"{key}: MATCH_prev total cos2 "
              f"{rec['match_prev_total_cos2']}"
              + (f" vs parent {rec.get('match_prev_total_cos2_parent_stored')}"
                 if parent else ''), flush=True)

    # ---------------- the absorption verdict ----------------
    mx = mixture_rows()
    res = {(r['layer'], r['head']): r['cos2_eval']['MATCH_prev']
           for r in tables['residual']}
    ful = {(r['layer'], r['head']): r['cos2_eval']['MATCH_prev']
           for r in tables['full']}
    zres = {(r['layer'], r['head']): r['null_z']['MATCH_prev']
            for r in tables['residual']}
    keys = sorted(res)
    rows = []
    for (l, h) in keys:
        row = {'layer': l, 'head': h,
               'parent_cos2': round(parent.get((l, h), float('nan')), 4)
               if parent else None,
               'residual_cos2': round(res[(l, h)], 4),
               'full_cos2': round(ful[(l, h)], 4),
               'residual_null_z': zres[(l, h)]}
        r22 = mx.get((l, h))
        if r22:
            row.update({'b_match_prev': r22['b_match_prev'],
                        'mass_share_match_prev': r22['mass_share_match_prev'],
                        'mass_share_residual': r22['mass_share_residual']})
        rows.append(row)
    tot_res = round(sum(res.values()), 4)
    tot_ful = round(sum(ful.values()), 4)
    tot_par = round(sum(parent.values()), 4) if parent else None
    verdict = {}
    if parent:
        prog_res = round(sum(res[p] for p in prog) / max(len(prog), 1), 4)
        prog_ful = round(sum(ful[p] for p in prog) / max(len(prog), 1), 4)
        prog_par = round(sum(parent[p] for p in prog) / max(len(prog), 1), 4)
        drop = tot_res < tot_par and prog_res < prog_par
        full_holds = tot_ful >= 0.9 * tot_par
        verdict = {
            'prediction_ii_registered':
                'E21 census re-run on the RESIDUAL patterns: MATCH_prev '
                'total eval-half cos^2 over all heads AND the mean over the '
                'parent\'s programmatic MATCH_prev heads both DROP vs the '
                'parent\'s stored census',
            'prediction_ii_verdict': 'CONFIRMED' if drop else 'REFUTED',
            'match_prev_total_cos2': {'parent_E19a': tot_par,
                                      'E22a_residual': tot_res,
                                      'E22a_full': tot_ful},
            'match_prev_mean_over_parent_programmatic_heads': {
                'parent_E19a': prog_par, 'E22a_residual': prog_res,
                'E22a_full': prog_ful},
            'three_way_reading': (
                'ABSORPTION (structure moved out of the residual into the '
                'named term)' if drop and full_holds else
                'LESS MATCHING OVERALL (both residual and full pattern carry '
                'less MATCH_prev mass than the parent)' if drop else
                'NO ABSORPTION (the residual still carries the parent\'s '
                'match structure; the named term is additional, not a '
                'replacement)')}
        if mx:
            b = [abs(mx[k]['b_match_prev']) for k in keys if k in mx]
            rr = [res[k] for k in keys if k in mx]
            ss = [mx[k]['mass_share_match_prev'] for k in keys if k in mx]
            verdict['spearman_abs_b_vs_residual_match_prev_cos2'] = round(
                spearman(b, rr), 4)
            verdict['spearman_named_match_mass_share_vs_residual_cos2'] = \
                round(spearman(ss, rr), 4)
            verdict['spearman_note'] = (
                'negative = heads that lean harder on the explicit named '
                'match term have less MATCH_prev structure left in their '
                'residual pattern (per-head absorption)')
    E.merge(JP31, 'absorption_E22a', {
        'question': 'did the explicit per-head match term ABSORB the match '
                    'structure out of the learned residual attention '
                    'pattern, or is the residual still doing it?',
        'checkpoint': f'{STEM}.pt',
        'parent': 'qk_e19_a.pt census stored in qk_e21_census.json',
        'stored_under': ['qk_e22.json::census_residual_E22a',
                         'qk_e22.json::census_full_E22a'],
        'verdict': verdict,
        'per_head': rows})
    print(json.dumps({'absorption_verdict': verdict}, indent=2), flush=True)

    del m
    if not E.SMOKE:
        torch.cuda.empty_cache()
    try:
        E22R.summarize()
    except Exception as ex:
        print(f"E22 summarize skipped ({str(ex)[:100]})", flush=True)
    print('e31 absorption run done', flush=True)


if __name__ == '__main__':
    main()
