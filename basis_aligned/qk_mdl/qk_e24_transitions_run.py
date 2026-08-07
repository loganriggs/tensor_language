"""E24 CODE-TO-CODE TRANSITION TABLES (checkpoint-only; no training).

QUESTION (the discrete-substitutability question from the E20 error
decomposition, BRAINSTORM_STATE 2026-08-07): now that E20a's slot contents
are discrete code pairs, which modules are already PRINTABLE LOOKUP TABLES --
i.e. how deterministic is the map from a module's INPUT codes (the k=2 code
pairs in the slots it reads most strongly) to its OUTPUT codes (the pair it
writes)?

ON qk_e20_a.pt (the trained E20a codebook arm, n=256/k=2 per slot).

PER MODULE (attention module l writes slot 2l, MLP module l writes slot
2l+1):
  INPUTS: the top-4 input slots by the arm's own wiring table = the module's
    read-group Frobenius norms (this module's read matrices restricted to
    each visible written slot's columns: attention c_q/c_k/c_q2/c_k2/c_v,
    MLP Left/Right -- the literal read-group-norm wiring table the lasso
    acts on), among slots already written at its read interface (attention
    at hn: slots 0..2l-1; MLP at xn: slots 0..2l). The input tuple at token
    position t = the concatenation of those slots' (code1, code2) pairs at
    position t (slot-index order, fixed).
  OUTPUT: the (code1, code2) pair the module's own written slot quantizes to
    at position t.
  EXCLUDED: attn0 (no quantized input slots at its interface) and mlp11
    (slot 23 is read only by the unquantized readout -- codebook 23 is
    causally inert, documented E20 exemption). 22 modules remain.
  CAVEAT (documented, expected): attention output at position t mixes VALUE
    content from all positions <= t, so a same-position input tuple is an
    incomplete predictor for attention modules by construction; MLP modules
    are per-position (bilinear in xn at t) so their tables are the fair
    test. Reported per module either way.

SPLITS (NO train rows anywhere): tables are FIT on the audit-slice-DISJOINT
held rows fresh34k[33200:34500] (1300 seqs, held, never trained on) and
EVALUATED on the fixed audit slice fresh34k[33000:33200] (200 seqs).

MEASURES per module:
  - marginal output entropy H(out pair) on the estimation split (bits);
  - conditional entropy H(out pair | input tuple) over estimation tuples
    seen >= 20 times (token-weighted over qualifying tuples);
  - determinism gain = H_marg - H_cond, and relative gain / H_marg;
  - fraction of qualifying input tuples with a single dominant output pair
    (>= 0.9 mass), both unweighted over tuples and token-weighted;
  - coverage: fraction of AUDIT tokens whose input tuple was seen >= 20
    times in the estimation split;
  - audit top-1 accuracy: on covered audit tokens, how often the table's
    argmax output pair is the realized one (the printable-lookup-table
    substitution score, on rows the table never saw).

CONTROLS (hard gates, run before the model pass):
  1. PLANTED determinism: a synthetic stream where the output IS a
     deterministic function of the input tuple must give relative gain
     > 0.95 through the exact table/entropy code;
  2. INDEPENDENT null: outputs drawn independently of inputs must give
     relative gain < 0.1 (small-sample entropy bias floor, quantified);
  3. SHUFFLED-PAIRING null per module: permuting the output pairs across
     tokens (within the module) must give ~zero determinism gain -- reported
     next to every real gain as its bias floor.

OUTPUT qk_e24.json: per-module table stats, the 3 most deterministic
modules with 3 example rows each ("reads code-X ('the'-detector) + code-Y
-> writes code-Z"), code names derived from qk_e20_code_dictionaries.json
where available (slots 0, 1, 3, 15, 22; a code is named by the majority
firing token of its dictionary examples). Idempotent on JSON keys."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time
from collections import Counter

import numpy as np

import qk_e_common as E
from qk_e_common import Q, DEPTH, torch
import qk_e15_reinvest_run as E15R
import qk_e20_codebook_run as E20R
import qk_deeproute_train_2 as R2

E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e24.json')
QZ_N = E20R.QZ_N
MIN_COUNT = 2 if E.SMOKE else 20
TOP_SLOTS = 4
DOM_MASS = 0.9
# Q.HELD row indices: audit = fixed audit slice fresh34k[33000:33200];
# estimation = the remaining held rows (audit-disjoint, never trained on).
AUD_LO, AUD_HI = 0, 200
EST_LO, EST_HI = 200, 1500


def entropy_bits(counts):
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


# ---------------- the table/entropy machinery (shared with the controls) ----------------
def contingency_stats(in_est, out_est, in_aud, out_aud, min_count,
                      want_examples=0):
    """Fit the input-tuple -> output contingency on the estimation rows,
    evaluate coverage/top-1 on the audit rows. in_*: (N, m) int arrays
    (concatenated input code tuples); out_*: (N, p) int arrays (output code
    tuple). Returns the stats dict (+ example rows if requested)."""
    n_est = len(in_est)
    all_in = np.concatenate([in_est, in_aud], 0)
    iuniq, tid_all = np.unique(all_in, axis=0, return_inverse=True)
    tid_e, tid_a = tid_all[:n_est], tid_all[n_est:]
    all_out = np.concatenate([out_est, out_aud], 0)
    ouniq, oid_all = np.unique(all_out, axis=0, return_inverse=True)
    oid_e, oid_a = oid_all[:n_est], oid_all[n_est:]
    n_out, n_t = len(ouniq), len(iuniq)

    h_marg = entropy_bits(np.bincount(oid_e, minlength=n_out))
    tc = np.bincount(tid_e, minlength=n_t)           # est count per tuple

    # unique (tuple, output) pairs with counts; keys sorted => ptid sorted
    key = tid_e.astype(np.int64) * n_out + oid_e
    uk, uc = np.unique(key, return_counts=True)
    ptid = (uk // n_out).astype(np.int64)
    poid = (uk % n_out).astype(np.int64)
    bnd = np.nonzero(np.r_[True, ptid[1:] != ptid[:-1]])[0]
    tids_g = ptid[bnd]                               # tuples with >=1 est row
    tc_g = tc[tids_g].astype(np.float64)
    p = uc / tc[ptid]
    ent_g = np.add.reduceat(-p * np.log2(np.maximum(p, 1e-300)), bnd)
    dom_g = np.maximum.reduceat(p, bnd)
    # argmax output per tuple (last element after a (tuple, count) lexsort)
    order = np.lexsort((uc, ptid))
    ptid_s, poid_s = ptid[order], poid[order]
    last = np.nonzero(np.r_[ptid_s[1:] != ptid_s[:-1], True])[0]
    pred_oid = np.full(n_t, -1, dtype=np.int64)
    pred_oid[ptid_s[last]] = poid_s[last]

    qual = tc_g >= min_count
    wq = tc_g[qual]
    if wq.sum() > 0:
        h_cond = float((ent_g[qual] * wq).sum() / wq.sum())
        frac_dom_tuples = float((dom_g[qual] >= DOM_MASS).mean())
        frac_dom_tokens = float(
            (wq * (dom_g[qual] >= DOM_MASS)).sum() / wq.sum())
    else:
        h_cond, frac_dom_tuples, frac_dom_tokens = float('nan'), 0.0, 0.0
    covered_est_mass = float(wq.sum() / max(n_est, 1))
    cov_a = tc[tid_a] >= min_count
    coverage = float(cov_a.mean()) if len(tid_a) else 0.0
    hits = (pred_oid[tid_a] == oid_a)[cov_a]
    top1 = float(hits.mean()) if len(hits) else float('nan')

    rec = {'n_est_tokens': int(n_est), 'n_audit_tokens': int(len(in_aud)),
           'n_tuples_est': int(len(tids_g)),
           'n_tuples_qualifying': int(qual.sum()),
           'H_marginal_bits': round(h_marg, 4),
           'H_conditional_bits': round(h_cond, 4),
           'determinism_gain_bits': round(h_marg - h_cond, 4),
           'relative_gain': round((h_marg - h_cond) / h_marg, 4)
           if h_marg > 0 else 0.0,
           'frac_qual_tuples_dominant_ge0.9': round(frac_dom_tuples, 4),
           'frac_qual_tokens_dominant_ge0.9': round(frac_dom_tokens, 4),
           'covered_est_mass': round(covered_est_mass, 4),
           'audit_coverage_frac': round(coverage, 4),
           'audit_top1_acc_covered': round(top1, 4)
           if not math.isnan(top1) else None}
    examples = []
    if want_examples:
        sel = np.nonzero(qual & (dom_g >= DOM_MASS))[0]
        sel = sel[np.argsort(-tc_g[sel])][:want_examples]
        for gi in sel:
            t = tids_g[gi]
            examples.append({'in_tuple': iuniq[t].tolist(),
                             'out': ouniq[pred_oid[t]].tolist(),
                             'count_est': int(tc_g[gi]),
                             'dominant_mass': round(float(dom_g[gi]), 3)})
    return rec, examples


# ---------------- controls ----------------
def control_planted():
    """Gates 1+2: planted deterministic map -> relative gain > 0.95;
    independent outputs -> relative gain < 0.1 (the bias floor), through the
    exact contingency_stats code."""
    key = 'control_planted_and_independent'
    done = E.loadj(JP).get(key)
    if done and done.get('pass'):
        print(f"{key}: already passed -- skip", flush=True)
        return
    rng = np.random.default_rng(7)
    n, pool, n_out = 20000, 30, 16
    tuples = rng.integers(0, 50, size=(pool, 8))
    lab = rng.integers(0, pool, size=n)
    in_arr = tuples[lab]
    # planted: output = a fixed random function of the tuple id
    fmap = rng.integers(0, n_out, size=(pool, 2))
    out_det = fmap[lab]
    # independent: outputs drawn without looking at the input
    out_ind = rng.integers(0, n_out, size=(n, 2))
    n_a = 2000
    det, _ = contingency_stats(in_arr[:-n_a], out_det[:-n_a],
                               in_arr[-n_a:], out_det[-n_a:], MIN_COUNT)
    ind, _ = contingency_stats(in_arr[:-n_a], out_ind[:-n_a],
                               in_arr[-n_a:], out_ind[-n_a:], MIN_COUNT)
    # shuffled-pairing of the PLANTED stream must also kill the gain
    perm = np.random.default_rng(0).permutation(n - n_a)
    shuf, _ = contingency_stats(in_arr[:-n_a], out_det[:-n_a][perm],
                                in_arr[-n_a:], out_det[-n_a:], MIN_COUNT)
    rec = {'planted': det, 'independent': ind, 'shuffled_planted': shuf,
           'pass': bool(det['relative_gain'] > 0.95
                        and det['audit_top1_acc_covered'] > 0.99
                        and ind['relative_gain'] < 0.1
                        and shuf['relative_gain'] < 0.1)}
    E.merge(JP, key, rec)
    print(f"{key}: planted gain {det['relative_gain']} (top1 "
          f"{det['audit_top1_acc_covered']}), independent "
          f"{ind['relative_gain']}, shuffled planted "
          f"{shuf['relative_gain']} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


# ---------------- model pass: code streams + wiring table ----------------
def collect_codes(m, rows, tag):
    """One eval fp32 forward pass per batch; returns {slot: (Ntok, 2) int
    array} for every quantized slot (codes cached once per forward -- the
    documented E20 single-quantization semantics)."""
    per = {}
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            b = rows[i:i + 8]
            col = {'codes': {}}
            m(b[:, :Q.T], collect=col)
            for k, chunks in col['codes'].items():
                per.setdefault(k, []).append(torch.cat(chunks, 0))
            if (i // 8) % 40 == 0:
                print(f"  codes[{tag}]: {i + b.shape[0]}/{len(rows)} seqs "
                      f"({time.time() - t0:.0f}s)", flush=True)
    return {k: torch.cat(v, 0).long().numpy().astype(np.int32)
            for k, v in per.items()}


def module_specs(m):
    """Per module: output slot + top-4 input slots by the module's own
    read-group Frobenius norms (the literal wiring table)."""
    s = m.s
    specs = []
    for l in range(DEPTH):
        blk = m.h[l]
        for kind in ('attn', 'mlp'):
            out_slot = 2 * l + (0 if kind == 'attn' else 1)
            if kind == 'attn':
                mats = [blk.c_q.weight, blk.c_k.weight, blk.c_q2.weight,
                        blk.c_k2.weight, blk.c_v.weight]
                avail = list(range(2 * l))
            else:
                mats = [blk.Left.weight, blk.Right.weight]
                avail = list(range(2 * l + 1))
            skip = None
            if out_slot == E20R.READOUT_ONLY_SLOT:
                skip = ('output slot 23 is never quantized (readout-only, '
                        'documented E20 exemption)')
            elif not avail:
                skip = 'no quantized input slots at its read interface'
            norms = {}
            for j in avail:
                norms[j] = float(sum(
                    M[:, s * j:s * (j + 1)].detach().float().pow(2).sum()
                    .item() for M in mats)) ** 0.5
            top = sorted(avail, key=lambda j: -norms[j])[:TOP_SLOTS]
            top = sorted(top)                       # fixed slot-index order
            specs.append({
                'module': f'{kind}{l}', 'out_slot': out_slot,
                'skip': skip, 'in_slots': top,
                'in_slot_norms': {str(j): round(norms[j], 3) for j in top},
                'all_read_norms': {str(j): round(norms[j], 3)
                                   for j in avail}})
    return specs


def code_names():
    """(slot, code) -> short name from qk_e20_code_dictionaries.json: the
    majority firing token over the code's dictionary examples."""
    p = E.jpath('qk_e20_code_dictionaries.json')
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    names = {}
    for slot_s, rec in d['per_slot'].items():
        for e_ in rec['codes']:
            toks = [ex['ctx'].split('==>')[-1] for ex in e_['examples']]
            if not toks:
                continue
            top, cnt = Counter(toks).most_common(1)[0]
            if cnt >= max(2, (len(toks) + 1) // 2):
                names[(int(slot_s), int(e_['code']))] = f'{top!r}-detector'
    return names


def fmt_code(slot, code, names):
    nm = names.get((slot, int(code)))
    return f'{int(code)} ({nm})' if nm else str(int(code))


def main_tables(m):
    key = 'modules_E24'
    j = E.loadj(JP)
    if key in j and 'top_deterministic_modules' in j:
        print(f"{key}: already done -- skip", flush=True)
        return
    rows = Q.HELD
    if E.SMOKE:
        aud_lo, aud_hi, est_lo, est_hi = 0, 4, 4, len(rows)
    else:
        aud_lo, aud_hi, est_lo, est_hi = AUD_LO, AUD_HI, EST_LO, EST_HI
    m.eval().float()
    codes_est = collect_codes(m, rows[est_lo:est_hi], 'est')
    codes_aud = collect_codes(m, rows[aud_lo:aud_hi], 'audit')
    del m
    torch.cuda.empty_cache()
    specs = module_specs_cache['specs']
    names = code_names()
    rng_null = np.random.default_rng(0)
    out_mod, out_ex = {}, {}
    for sp in specs:
        if sp['skip']:
            out_mod[sp['module']] = {'excluded': sp['skip']}
            continue
        ins = sp['in_slots']
        in_est = np.concatenate([codes_est[j2] for j2 in ins], 1)
        in_aud = np.concatenate([codes_aud[j2] for j2 in ins], 1)
        o_est, o_aud = codes_est[sp['out_slot']], codes_aud[sp['out_slot']]
        rec, ex = contingency_stats(in_est, o_est, in_aud, o_aud,
                                    MIN_COUNT, want_examples=6)
        perm = rng_null.permutation(len(o_est))
        null, _ = contingency_stats(in_est, o_est[perm], in_aud, o_aud,
                                    MIN_COUNT)
        rec.update({
            'out_slot': sp['out_slot'], 'in_slots': ins,
            'in_slot_read_norms': sp['in_slot_norms'],
            'null_shuffled_gain_bits': null['determinism_gain_bits'],
            'null_shuffled_relative_gain': null['relative_gain'],
            'gain_minus_null_bits': round(
                rec['determinism_gain_bits']
                - null['determinism_gain_bits'], 4)})
        out_mod[sp['module']] = rec
        out_ex[sp['module']] = ex
        print(f"  {sp['module']}: H_marg {rec['H_marginal_bits']} -> "
              f"H_cond {rec['H_conditional_bits']} (null gain "
              f"{null['determinism_gain_bits']}), audit coverage "
              f"{rec['audit_coverage_frac']} top1 "
              f"{rec['audit_top1_acc_covered']}", flush=True)
    E.merge(JP, key, {
        'checkpoint': 'qk_e20_a.pt',
        'estimation_rows': f'fresh34k[{33000 + est_lo}:{33000 + est_hi}] '
                           '(held, audit-disjoint, never trained on)',
        'audit_rows': f'fresh34k[{33000 + aud_lo}:{33000 + aud_hi}] '
                      '(the fixed audit slice)',
        'min_count': MIN_COUNT, 'top_input_slots': TOP_SLOTS,
        'input_definition': 'concatenated (code1, code2) pairs of the '
                            'top-4 read-norm slots, same token position; '
                            'attention modules mix values across positions '
                            'so this is an incomplete predictor for them '
                            'by construction (documented caveat)',
        'per_module': out_mod})

    # ---- the 3 most deterministic modules + printable example rows ----
    scored = [(mod, r) for mod, r in out_mod.items() if 'excluded' not in r
              and r['relative_gain'] is not None
              and not math.isnan(r['H_conditional_bits'])]
    well = [x for x in scored if x[1]['covered_est_mass'] >= 0.2]
    pool = well if len(well) >= 3 else scored
    pool.sort(key=lambda x: -(x[1]['relative_gain']
                              - x[1]['null_shuffled_relative_gain']))
    top3 = []
    for mod, r in pool[:3]:
        rows_out = []
        for ex in out_ex.get(mod, [])[:3]:
            ins = out_mod[mod]['in_slots']
            parts = []
            for pi, j2 in enumerate(ins):
                c1, c2 = ex['in_tuple'][2 * pi], ex['in_tuple'][2 * pi + 1]
                parts.append(
                    f"{R2.stream_name(j2 + 1)}[{j2}] codes "
                    f"({fmt_code(j2, c1, names)}, {fmt_code(j2, c2, names)})")
            o1, o2 = ex['out']
            os_ = out_mod[mod]['out_slot']
            rows_out.append(
                f"reads {' + '.join(parts)} -> writes "
                f"({fmt_code(os_, o1, names)}, {fmt_code(os_, o2, names)}) "
                f"[{ex['dominant_mass'] * 100:.0f}% of {ex['count_est']} "
                f"estimation occurrences]")
        top3.append({'module': mod,
                     'relative_gain_minus_null': round(
                         r['relative_gain']
                         - r['null_shuffled_relative_gain'], 4),
                     'audit_top1_acc_covered': r['audit_top1_acc_covered'],
                     'audit_coverage_frac': r['audit_coverage_frac'],
                     'example_rows': rows_out})
    E.merge(JP, 'top_deterministic_modules', {
        'ranking': 'relative determinism gain minus its shuffled null, '
                   'preferring modules with covered estimation mass >= 0.2',
        'code_names_source': 'qk_e20_code_dictionaries.json (majority '
                             'firing token; only slots 0/1/3/15/22 have '
                             'dictionaries)',
        'top3': top3})
    for t in top3:
        print(f"TOP {t['module']} (gain-null {t['relative_gain_minus_null']}"
              f", audit top1 {t['audit_top1_acc_covered']}):", flush=True)
        for rrow in t['example_rows']:
            print(f"    {rrow}", flush=True)


module_specs_cache = {}

if __name__ == '__main__':
    E.setup()
    control_planted()
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c
        assert os.path.exists(E.ckpath('qk_e20_a')), 'qk_e20_a.pt missing'
        m, _ = E.load_arm('qk_e20_a', lambda: E20R.make_e20(s=s_c))
    else:
        m = E20R.make_e20(s=s_c).eval().float()
    module_specs_cache['specs'] = module_specs(m)
    if 'wiring_table_note' not in E.loadj(JP):
        E.merge(JP, 'wiring_table_note', {
            'definition': 'per-module read-group Frobenius norms from the '
                          'checkpoint weights (attention: c_q/c_k/c_q2/'
                          'c_k2/c_v slot columns; MLP: Left/Right slot '
                          'columns) -- the literal wiring table the '
                          'group-lasso prices',
            'excluded_modules': {
                'attn0': 'no quantized input slots at its interface',
                'mlp11': 'writes slot 23, never quantized (readout-only '
                         'exemption)'}})
    main_tables(m)
    print('e24 transitions run done', flush=True)
