"""E31b FACTORED CODE TABLES (checkpoint-only; no training).

THE WALL E24 HIT: modelling a module as a JOINT lookup table from its input
code tuple to its output code tuple is defeated by coverage, not by
determinism. On qk_e20_a the 8-code input tuples of 20 of the 22 modules
covered only 0.04% of audit tokens (at 100% top-1 accuracy on that sliver),
and the one module with real coverage (mlp0, 37%) scored 15% top-1. Joint
tuples are simply too sparse: the table is right where it exists and absent
everywhere else.

THIS RUN asks the printability question in the factored form instead: model
P(output code | input codes) as a SUM OF PER-INPUT-SLOT CONTRIBUTIONS -- one
small table per (input slot, input code) giving a vector of output-code
logits, summed over the input slots, one such model per output code position:

    logits(output position p) = bias_p + sum_j  W[p][j][ code_j ]

Two fits of exactly that form, both on the estimation rows only:
  * COUNTING (naive Bayes): W[p][j][c][o] = log P(code_j = c | out_p = o)
    with Laplace smoothing, bias = log P(out_p = o). Closed form, one pass of
    counts.
  * LOGISTIC REGRESSION: the same parameterization fit by a few Adam passes
    of multinomial cross-entropy (a factored/embedding-bag softmax).
  * SPARSE variant: keep only the top-8 output logits in each (input slot,
    input code) row (the rest replaced by the dropped entries' mean) -- the
    literal "small printable table" version, and its accuracy cost.

Because the factored model always produces a prediction it has FULL coverage
by construction, so the comparison against the joint table is:
  - top-1 accuracy at FULL audit coverage (exact output tuple, and per output
    code position), against the marginal-majority baseline (the floor: always
    print the most common output tuple);
  - accuracy-vs-coverage when abstaining on low-confidence predictions
    (confidence = product of the per-position max softmax probabilities),
    reported at coverage 100/75/50/25/10/5/1%;
  - the joint table's own accuracy-vs-coverage curve, obtained by sweeping
    its minimum-count threshold (min_count 1/2/5/20/100) through
    qk_e24_transitions_run.contingency_stats VERBATIM;
  - description size in numbers for both (joint: rows x tuple length;
    factored: features x codes x kept logits).

SPLITS: identical to E24 -- tables FIT on fresh34k[33200:34500] (held,
audit-disjoint, never trained on), EVALUATED on the fixed audit slice
fresh34k[33000:33200]. Inputs = the top-4 read-norm slots at the module's
read interface (qk_e24_transitions_run.module_specs verbatim); outputs = the
module's own written slot's code tuple.

CONTROLS (hard gates, before any model pass):
  1. PLANTED FACTORED structure: outputs generated as argmax over a planted
     sum of per-slot tables must be recovered at ~1.0 by the logistic fit
     (the counting fit is reported alongside);
  2. PLANTED SINGLE-SLOT deterministic map: both fits ~1.0;
  3. SHUFFLED-OUTPUT null: accuracy must fall to the marginal-majority rate;
  4. E24's own planted/independent joint-table gate re-asserted (the joint
     baseline here runs through that exact code).

MODELS: qk_e20_a.pt (the E20a codebook arm, k=2 everywhere, the arm with
code dictionaries) and, if it exists, the E31a composition arm (variable-k:
4 codes on attention slots, 2 on MLP slots -- the factored machinery is
generic over tuple lengths).

OUTPUT qk_e31b.json (idempotent per model/key)."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, DEPTH, torch
import qk_e15_reinvest_run as E15R
import qk_e20_codebook_run as E20R
import qk_e24_transitions_run as E24R
import qk_deeproute_train_2 as R2

E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e31b.json')
QZ_N = E20R.QZ_N
MIN_COUNTS = (1, 2, 5, 20, 100)
COV_GRID = (1.0, 0.75, 0.5, 0.25, 0.10, 0.05, 0.01)
SPARSE_R = 8
LR_STEPS = 40 if E.SMOKE else 800         # fixed step budget (cosine-decayed)
LR_BATCH = 256 if E.SMOKE else 8192
LR_LR = 0.05
LAPLACE = 0.5


# ---------------- the factored table machinery ----------------
def _dev():
    return 'cpu' if E.SMOKE else 'cuda'


def fit_counting(in_est, out_col, n_codes, n_out):
    """Naive-Bayes tables: W[j][c][o] = log P(code_j = c | out = o), plus the
    log-prior bias. Exactly the factored (per input slot, per code) ->
    output-logit-vector form, fit by counting."""
    m = in_est.shape[1]
    cnt_out = np.bincount(out_col, minlength=n_out).astype(np.float64)
    W = np.zeros((m, n_codes, n_out), dtype=np.float32)
    for j in range(m):
        key = in_est[:, j].astype(np.int64) * n_out + out_col
        cnt = np.bincount(key, minlength=n_codes * n_out).astype(np.float64)
        cnt = cnt.reshape(n_codes, n_out) + LAPLACE
        W[j] = np.log(cnt / (cnt_out[None, :] + LAPLACE * n_codes)).astype(
            np.float32)
    bias = np.log((cnt_out + LAPLACE) / (cnt_out.sum() + LAPLACE * n_out))
    return torch.from_numpy(W), torch.from_numpy(bias.astype(np.float32))


def factored_logits(W, bias, in_rows, chunk=65536):
    """logits = bias + sum_j W[j][code_j], streamed over rows."""
    dev = W.device
    outs = []
    for i in range(0, len(in_rows), chunk):
        idx = in_rows[i:i + chunk].to(dev)
        lg = bias[None, :].expand(idx.shape[0], -1).clone()
        for j in range(W.shape[0]):
            lg += W[j][idx[:, j]]
        outs.append(lg)
    return torch.cat(outs, 0)


def fit_logistic(in_est, out_col, n_codes, n_out, seed=0, steps=None):
    """The same factored parameterization fit by multinomial logistic
    regression (a fixed cosine-decayed Adam step budget)."""
    dev = _dev()
    LR_STEPS_ = LR_STEPS if steps is None else steps
    g = torch.Generator(device='cpu').manual_seed(seed)
    m = in_est.shape[1]
    W = (0.01 * torch.randn(m, n_codes, n_out, generator=g)).to(dev)
    bias = torch.zeros(n_out, device=dev)
    W.requires_grad_(True)
    bias.requires_grad_(True)
    opt = torch.optim.Adam([W, bias], lr=LR_LR)
    X = torch.from_numpy(in_est.astype(np.int64))
    Y = torch.from_numpy(out_col.astype(np.int64))
    n = len(X)
    bs = min(LR_BATCH, n)
    sg = torch.Generator().manual_seed(seed + 1)
    for t in range(LR_STEPS_):
        for gp in opt.param_groups:               # cosine-decayed lr
            gp['lr'] = LR_LR * 0.5 * (1 + math.cos(math.pi * t / LR_STEPS_))
        sel = torch.randint(0, n, (bs,), generator=sg)
        xb = X[sel].to(dev)
        yb = Y[sel].to(dev)
        lg = bias[None, :].expand(bs, -1).clone()
        for j in range(m):
            lg = lg + W[j][xb[:, j]]
        loss = torch.nn.functional.cross_entropy(lg, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return W.detach(), bias.detach()


def sparsify(W, r=SPARSE_R):
    """Keep the top-r output logits in each (input slot, input code) row; the
    dropped entries are replaced by their mean (a single number per row)."""
    Ws = W.clone()
    top = Ws.topk(r, dim=-1)
    mask = torch.zeros_like(Ws, dtype=torch.bool)
    mask.scatter_(-1, top.indices, True)
    dropped_sum = (Ws * (~mask)).sum(-1, keepdim=True)
    n_dropped = (~mask).sum(-1, keepdim=True).clamp_min(1)
    fill = dropped_sum / n_dropped
    return torch.where(mask, Ws, fill.expand_as(Ws))


def evaluate_factored(models, in_aud, out_aud, n_out):
    """models: list (one per output position) of (W, bias). Returns exact-
    tuple and per-position accuracy at full coverage, the confidence-ordered
    accuracy-vs-coverage curve, and the majority-class floor."""
    N, p = out_aud.shape
    correct = torch.ones(N, dtype=torch.bool)
    logconf = torch.zeros(N, dtype=torch.float64)
    per_pos = []
    for c, (W, bias) in enumerate(models):
        lg = factored_logits(W, bias, torch.from_numpy(in_aud.astype(np.int64)))
        pr = torch.softmax(lg.float(), dim=1)
        mx, am = pr.max(1)
        hit = (am.cpu() == torch.from_numpy(out_aud[:, c].astype(np.int64)))
        per_pos.append(round(float(hit.float().mean()), 4))
        correct &= hit
        logconf += torch.log(mx.double().cpu().clamp_min(1e-30))
        del lg, pr
    order = torch.argsort(logconf, descending=True)
    corr_sorted = correct[order].double()
    cum = torch.cumsum(corr_sorted, 0)
    curve = []
    for cov in COV_GRID:
        k = max(1, int(round(cov * N)))
        curve.append({'coverage': cov,
                      'accuracy': round(float(cum[k - 1] / k), 4),
                      'n_tokens': int(k)})
    return {'exact_tuple_top1_full_coverage': round(float(
        correct.float().mean()), 4),
        'per_position_top1_full_coverage': per_pos,
        'accuracy_vs_coverage': curve}


def majority_floor(out_est, out_aud):
    """Always print the most common output tuple seen in estimation."""
    uniq, inv = np.unique(out_est, axis=0, return_inverse=True)
    top = uniq[np.bincount(inv).argmax()]
    return round(float((out_aud == top[None, :]).all(1).mean()), 4)


def fit_and_eval(in_est, out_est, in_aud, out_aud, n_codes=QZ_N,
                 with_logistic=True, lr_steps=None):
    """Full factored fit + evaluation for one module (all output positions).
    Returns the record dict."""
    p = out_est.shape[1]
    n_out = n_codes
    rec = {'n_input_features': int(in_est.shape[1]),
           'n_output_positions': int(p),
           'majority_tuple_floor_audit': majority_floor(out_est, out_aud)}
    cnt_models = [fit_counting(in_est, out_est[:, c], n_codes, n_out)
                  for c in range(p)]
    cnt_models = [(W.to(_dev()), b.to(_dev())) for W, b in cnt_models]
    rec['counting'] = evaluate_factored(cnt_models, in_aud, out_aud, n_out)
    if with_logistic:
        lr_models = [fit_logistic(in_est, out_est[:, c], n_codes, n_out,
                                  seed=c, steps=lr_steps) for c in range(p)]
        rec['logistic'] = evaluate_factored(lr_models, in_aud, out_aud, n_out)
        sp_models = [(sparsify(W), b) for W, b in lr_models]
        rec['logistic_sparse_top8'] = evaluate_factored(sp_models, in_aud,
                                                        out_aud, n_out)
        rec['table_size_numbers'] = {
            'factored_dense': int(in_est.shape[1] * n_codes * n_out * p),
            'factored_sparse_top8': int(in_est.shape[1] * n_codes
                                        * (SPARSE_R * 2 + 1) * p),
            'note': 'sparse counts each kept (output code, logit) pair plus '
                    'one fill value per row'}
        del lr_models, sp_models
    del cnt_models
    if not E.SMOKE:
        torch.cuda.empty_cache()
    return rec


# ---------------- controls ----------------
def control_planted():
    key = 'control_planted_factored'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    rng = np.random.default_rng(11)
    n_codes, m, n_out = 32, 4, 8
    # full-size settings even in smoke: this control validates the FITTING
    # machinery, so it must run at the real fit budget (it is a tiny toy --
    # 4 x 32 -> 8 tables -- and costs seconds on CPU)
    n, n_a, ctrl_steps = 60000, 10000, 800
    X = rng.integers(0, n_codes, size=(n, m))
    # (1) planted FACTORED map: out = argmax_o sum_j Wtrue[j][c_j][o]
    Wt = rng.normal(size=(m, n_codes, n_out))
    sc = np.zeros((n, n_out))
    for j in range(m):
        sc += Wt[j][X[:, j]]
    y_fac = sc.argmax(1)[:, None]
    # (2) planted SINGLE-SLOT deterministic map: out = g(c_0)
    g = rng.integers(0, n_out, size=n_codes)
    y_one = g[X[:, 0]][:, None]
    # (3) shuffled-output null on the factored stream
    y_shuf = y_fac.copy()
    rng.shuffle(y_shuf)
    out = {}
    for tag, y in (('planted_factored', y_fac), ('planted_single_slot', y_one),
                   ('shuffled_null', y_shuf)):
        out[tag] = fit_and_eval(X[:-n_a], y[:-n_a], X[-n_a:], y[-n_a:],
                                n_codes=n_codes, lr_steps=ctrl_steps)
    floor = out['shuffled_null']['majority_tuple_floor_audit']
    rec = {'setup': f'{m} input slots x {n_codes} codes -> {n_out} output '
                    f'codes, {n - n_a} estimation / {n_a} audit rows, '
                    f'{ctrl_steps} logistic steps (the real fit budget)',
           'results': out,
           'pass': bool(
               out['planted_factored']['logistic'][
                   'exact_tuple_top1_full_coverage'] >= 0.95
               and out['planted_single_slot']['logistic'][
                   'exact_tuple_top1_full_coverage'] >= 0.99
               and out['planted_single_slot']['counting'][
                   'exact_tuple_top1_full_coverage'] >= 0.99
               and out['shuffled_null']['logistic'][
                   'exact_tuple_top1_full_coverage'] <= floor + 0.02
               and out['shuffled_null']['counting'][
                   'exact_tuple_top1_full_coverage'] <= floor + 0.02)}
    E.merge(JP, key, rec)
    print(f"{key}: planted factored logistic "
          f"{out['planted_factored']['logistic']['exact_tuple_top1_full_coverage']}"
          f", single-slot "
          f"{out['planted_single_slot']['logistic']['exact_tuple_top1_full_coverage']}"
          f", shuffled null "
          f"{out['shuffled_null']['logistic']['exact_tuple_top1_full_coverage']}"
          f" (majority floor {floor}) -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


# ---------------- joint baseline (E24 machinery verbatim) ----------------
def joint_curve(in_est, out_est, in_aud, out_aud):
    """The joint tuple-lookup table's own coverage/accuracy curve, swept over
    its minimum-count threshold through E24's contingency_stats."""
    pts = []
    for mc in ([1, 2] if E.SMOKE else MIN_COUNTS):
        r, _ = E24R.contingency_stats(in_est, out_est, in_aud, out_aud, mc)
        pts.append({'min_count': mc,
                    'audit_coverage_frac': r['audit_coverage_frac'],
                    'audit_top1_acc_covered': r['audit_top1_acc_covered'],
                    'n_tuples_qualifying': r['n_tuples_qualifying'],
                    'H_marginal_bits': r['H_marginal_bits'],
                    'H_conditional_bits': r['H_conditional_bits']})
    return pts


# ---------------- per-model pass ----------------
def run_model(name, stem, factory, s_c):
    key = f'modules_{name}'
    if key in E.loadj(JP):
        print(f"{key}: already done -- skip", flush=True)
        return
    if E.SMOKE:
        m = factory().eval().float()
    else:
        if not os.path.exists(E.ckpath(stem)):
            print(f"{name}: {stem}.pt missing -- skipped", flush=True)
            return
        m, _ = E.load_arm(stem, factory)
    m.eval().float()
    specs = E24R.module_specs(m)
    rows = Q.HELD
    if E.SMOKE:
        aud_lo, aud_hi, est_lo, est_hi = 0, 4, 4, len(rows)
    else:
        aud_lo, aud_hi = E24R.AUD_LO, E24R.AUD_HI
        est_lo, est_hi = E24R.EST_LO, E24R.EST_HI
    t0 = time.time()
    codes_est = E24R.collect_codes(m, rows[est_lo:est_hi], f'{name}-est')
    codes_aud = E24R.collect_codes(m, rows[aud_lo:aud_hi], f'{name}-audit')
    del m
    if not E.SMOKE:
        torch.cuda.empty_cache()
    out_mod = {}
    for sp in specs:
        if sp['skip']:
            out_mod[sp['module']] = {'excluded': sp['skip']}
            continue
        ins = sp['in_slots']
        in_est = np.concatenate([codes_est[j] for j in ins], 1).astype(np.int64)
        in_aud = np.concatenate([codes_aud[j] for j in ins], 1).astype(np.int64)
        o_est = codes_est[sp['out_slot']].astype(np.int64)
        o_aud = codes_aud[sp['out_slot']].astype(np.int64)
        rec = fit_and_eval(in_est, o_est, in_aud, o_aud)
        rec['out_slot'] = sp['out_slot']
        rec['in_slots'] = ins
        rec['joint_baseline_curve'] = joint_curve(in_est, o_est, in_aud, o_aud)
        j20 = [p for p in rec['joint_baseline_curve'] if p['min_count'] == 20]
        rec['joint_at_min_count_20'] = j20[0] if j20 else None
        out_mod[sp['module']] = rec
        fa = rec['logistic']['exact_tuple_top1_full_coverage'] \
            if 'logistic' in rec else None
        print(f"  {sp['module']}: factored(LR) {fa} at FULL coverage "
              f"(counting "
              f"{rec['counting']['exact_tuple_top1_full_coverage']}, floor "
              f"{rec['majority_tuple_floor_audit']}); joint@20 "
              f"{rec['joint_at_min_count_20']}", flush=True)
    E.merge(JP, key, {
        'checkpoint': f'{stem}.pt',
        'estimation_rows': f'fresh34k[{33000 + est_lo}:{33000 + est_hi}] '
                           '(held, audit-disjoint, never trained on)',
        'audit_rows': f'fresh34k[{33000 + aud_lo}:{33000 + aud_hi}] '
                      '(the fixed audit slice)',
        'input_definition': 'concatenated code tuples of the top-4 read-norm '
                            'slots at the module read interface '
                            '(qk_e24_transitions_run.module_specs), same '
                            'token position',
        'factored_form': 'logits(output position) = bias + sum over input '
                         'features of W[feature][code] (one small table per '
                         '(input slot, code)); counting = naive Bayes, '
                         'logistic = the same form fit by Adam',
        'runtime_s': round(time.time() - t0, 1),
        'per_module': out_mod})
    summarize_model(name, out_mod)


def summarize_model(name, out_mod):
    rows = [(mod, r) for mod, r in out_mod.items() if 'excluded' not in r]
    if not rows:
        return
    def g(r, k):
        return r['logistic'][k] if 'logistic' in r else None
    fac = [g(r, 'exact_tuple_top1_full_coverage') for _, r in rows]
    cnt = [r['counting']['exact_tuple_top1_full_coverage'] for _, r in rows]
    flo = [r['majority_tuple_floor_audit'] for _, r in rows]
    spa = [r['logistic_sparse_top8']['exact_tuple_top1_full_coverage']
           for _, r in rows if 'logistic_sparse_top8' in r]
    jcov = [r['joint_at_min_count_20']['audit_coverage_frac'] for _, r in rows
            if r.get('joint_at_min_count_20')]
    jacc = [r['joint_at_min_count_20']['audit_top1_acc_covered'] for _, r in
            rows if r.get('joint_at_min_count_20')
            and r['joint_at_min_count_20']['audit_top1_acc_covered']
            is not None]
    best = sorted(rows, key=lambda x: -(g(x[1],
                  'exact_tuple_top1_full_coverage') or 0))[:5]
    # accuracy at 10% coverage (the abstaining regime), mean over modules
    a10 = [c['accuracy'] for _, r in rows if 'logistic' in r
           for c in r['logistic']['accuracy_vs_coverage']
           if c['coverage'] == 0.10]
    summ = {
        'n_modules': len(rows),
        'factored_logistic_exact_tuple_top1_full_coverage': {
            'mean': round(float(np.mean([x for x in fac if x is not None])), 4),
            'max': round(float(np.max([x for x in fac if x is not None])), 4)},
        'factored_counting_mean': round(float(np.mean(cnt)), 4),
        'majority_tuple_floor_mean': round(float(np.mean(flo)), 4),
        'sparse_top8_mean': round(float(np.mean(spa)), 4) if spa else None,
        'factored_accuracy_at_10pct_coverage_mean':
            round(float(np.mean(a10)), 4) if a10 else None,
        'joint_baseline_at_min_count_20': {
            'mean_audit_coverage': round(float(np.mean(jcov)), 4)
            if jcov else None,
            'mean_top1_on_covered': round(float(np.mean(jacc)), 4)
            if jacc else None},
        'top5_modules_by_factored_accuracy': [
            {'module': mod,
             'factored_logistic': g(r, 'exact_tuple_top1_full_coverage'),
             'majority_floor': r['majority_tuple_floor_audit'],
             'per_position': g(r, 'per_position_top1_full_coverage'),
             'accuracy_vs_coverage': g(r, 'accuracy_vs_coverage')}
            for mod, r in best],
        'reading': 'the factored model has FULL coverage by construction, so '
                   'its full-coverage accuracy is the printability number to '
                   'compare against the majority-tuple floor; the joint '
                   'table\'s accuracy applies only on its covered sliver'}
    E.merge(JP, f'summary_{name}', summ)
    print(json.dumps({f'summary_{name}': summ}, indent=2)[:3000], flush=True)


if __name__ == '__main__':
    E.setup()
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c
    control_planted()
    E24R.control_planted()                    # E24's joint-table gate (reused)

    if 'E31b_design' not in E.loadj(JP):
        E.merge(JP, 'E31b_design', {
            'question': 'are modules printable as SMALL FACTORED tables even '
                        'though they are not printable as joint lookup '
                        'tables? (E24: joint 8-code tuples covered 0.04% of '
                        'audit tokens at 100% accuracy, or 37% coverage at '
                        '15% accuracy)',
            'factored_model': 'P(output code | inputs) as a sum of '
                              'per-(input slot, input code) logit tables, '
                              'one model per output code position; fits: '
                              'counting (naive Bayes) and logistic '
                              'regression; sparse variant keeps the top-8 '
                              'output logits per row',
            'coverage': 'the factored model always predicts -> full '
                        'coverage; abstention curve by confidence',
            'baselines': ['marginal-majority output tuple (the floor)',
                          'the E24 joint lookup table, min_count swept '
                          '1/2/5/20/100 (its own coverage/accuracy curve)'],
            'splits': 'estimation fresh34k[33200:34500], audit '
                      'fresh34k[33000:33200] -- identical to E24'})

    run_model('E20a', 'qk_e20_a', lambda: E20R.make_e20(s=s_c), s_c)

    try:
        import qk_e31a_compose_run as E31A
        if E.SMOKE or os.path.exists(E.ckpath('qk_e31a_a')):
            run_model('E31a', 'qk_e31a_a', lambda: E31A.make_e31(s=s_c), s_c)
        else:
            print('E31a checkpoint not present yet -- skipped (rerun after '
                  'the composition arm finishes)', flush=True)
    except Exception as ex:
        print(f'E31a pass skipped: {str(ex)[:200]}', flush=True)

    print('e31b factored run done', flush=True)
