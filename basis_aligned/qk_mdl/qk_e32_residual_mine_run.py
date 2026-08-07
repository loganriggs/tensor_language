"""E32 RESIDUAL PATTERN MINING (checkpoint-only, no training; -> qk_e32.json).

QUESTION (the predicate-library growth step). E31 showed ABSORPTION: giving
every head an explicit named MATCH_prev term drained the match structure out of
the learned bilinear residual pattern (residual MATCH_prev total cos^2 0.5036 ->
0.0951; ZERO programmatic heads left in the residual, against 42 in the full
pattern). So what is the residual pattern -- which is still the majority of the
per-head pattern mass -- actually DOING? Its structure is the evidence for the
NEXT predicate to add to the library.

MODELS: the predicate-basis arm at all three seeds (qk_e22_a = seed 0, plus the
E29 replicates qk_e29_predbasis_s1 / qk_e29_predbasis_s2, checkpoint paths from
qk_e29.json). The residual pattern is extracted exactly as in
qk_e31_absorption_run: the census callback fires on the bilinear-only pattern
BEFORE the named terms are added, so full = residual + named is exact by
construction (control 1 re-checks it per seed at 1e-4).

(a) EXPANDED PREDICATE LIBRARY. qk_e21_census_run's scoring machinery is reused
verbatim -- census_model / null_zscores / score_heads / summarize / Stats and
its held-out uncentered-R^2 + shuffled-token-null z-score protocol -- with its
module-level library MONKEYPATCHED to a superset (E21's build_feats allocates
(B, NF, T, T) from the module-global NF, so widening NF and wrapping build_feats
extends every downstream statistic with no other change; control 0 asserts the
first ten channels are bit-identical to the E21 library). Added, on top of E21's
{MATCH_same, MATCH_prev, MATCH_next, PREV_token, FIRST_token, KEY_newline,
KEY_punct, KEY_func, KEY_cap, KEY_digit, POSITIONAL_decay}:
  MATCH_prev2      1[tok_{j-2} == tok_i]        (two-back induction)
  KEY_repeat       1[tok_j occurred before j]   (key-side repetition)
  SAME_WORD_PIECE  1[leadspace(tok_j) == leadspace(tok_i)]  (word-piece class)
  DUPLICATE_pair   1[tok_j == tok_i AND i - j > 1]          (non-adjacent dup)
  Q<c1>_x_K<c2>    25 query-class x key-class contingencies over
                   {newline, punct, func, cap, digit}
(MATCH_next is already in the E21 library and is kept.)

(b) UNSUPERVISED STRUCTURE. Per head, the residual pattern of each audit
sequence is SVD'd: the fraction of squared (Frobenius) mass in rank 1 / 2 / 4,
and held-out R^2 of the top singular vectors regressed on (i) TOKEN identity
(the sequence's token embeddings projected on the top-32 embedding principal
components) versus (ii) POSITION (a 32-element cosine basis) -- same feature
count, same even-position fit / odd-position eval split, so the two are
comparable. This says whether what is left is low-rank-and-nameable (and of
which kind) or genuinely diffuse.

(c) CAUSAL WEIGHT. On the same audit slice: held CE and the induction advantage
(E28's repeated-prefix probe: CE(first copy) - CE(second copy) on identical
targets) with (i) the residual zeroed on every head (named terms only), (ii) the
named terms zeroed (residual only), (iii) untouched. This is the honest split of
what the named library has captured against what is left, and the ceiling on
what any next predicate can buy. Finally, for the top-3 candidate predicates,
the residual of their programmatic heads is REPLACED by the fitted
a * predicate + b * profile (and by profile-only as the control) and the
recovered fraction of the residual-zeroing cost is measured.

CONTROLS (hard asserts, before the measurements):
  0. library back-compat: the expanded build_feats reproduces the E21 library
     bit-exactly on its first ten channels;
  1. E21's own synthetic known-answer patterns (exact previous-token, exact
     same-token) scored through the EXPANDED library;
  2. new-predicate known answers: each new predicate's own pattern scores
     R^2 ~ 1.0 on itself and is the library argmax;
  3. exact decomposition, per seed: full - residual - named == 0 (pattern
     level) and the subtract-and-re-add hook reproduces the logits;
  4. SVD/regression known answers: an exact rank-1 pattern is 100% rank-1; a
     pattern built as g(tok_i) h(tok_j) scores token >> position; a pattern
     built as f(i) f(j) scores position >> token;
  5. untrained-init floor: the same expanded census on a fresh-init model of
     the same architecture (the floor every new predicate must clear).

OUTPUT qk_e32.json, ending in a RANKED "next predicate to add" recommendation
backed by three-seed z-scores, programmatic-head counts over the untrained
floor, and the measured causal weight. Idempotent on the top-level summary key;
smoke-gated via QK_SMOKE=1."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json
import math
import time

import numpy as np

import qk_e_common as E
from qk_e_common import Q, DEPTH, F, torch
import qk_e15_reinvest_run as E15R
import qk_e22_predbasis_run as E22R
import qk_e31_absorption_run as E31R          # make_fwd (residual view), spearman

E.DEV = 'cpu' if E.SMOKE else 'cuda'

JP = E.jpath('qk_e32.json')
ROWS = (33000, 33200)                          # the fixed E21/E31 audit slice
GAIN_THR = 0.05                                # bilin18 programmatic criterion
Z_THR = 3.0
ALREADY_NAMED = ('MATCH_prev', 'MATCH_same', 'PREV_token')

MODELS = [('s0', 'qk_e22_a'),
          ('s1', 'qk_e29_predbasis_s1'),
          ('s2', 'qk_e29_predbasis_s2')]

NEWN = ['MATCH_prev2', 'KEY_repeat', 'SAME_WORD_PIECE', 'DUPLICATE_pair']
CLS_NAMES = ['newline', 'punct', 'func', 'cap', 'digit']
QXK = [f'Q{a}_x_K{b}' for a in CLS_NAMES for b in CLS_NAMES]

FEATN_EXT = None                               # set by install_library
NF0 = None
_ORIG_BUILD = None


# ============================ the expanded library ==========================
def lead_class():
    """1 for gpt2 tokens carrying a leading space (word start), 0 for
    continuation word pieces -- the SAME_WORD_PIECE class."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    L = torch.zeros(Q.V, dtype=torch.bool)
    for i in range(Q.V):
        s = tok.convert_ids_to_tokens(i)
        if s is not None and s.startswith('Ġ'):
            L[i] = True
    return L.to(E.DEV)


def install_library(E21):
    """Widen qk_e21_census_run's predicate library in place. E21.build_feats
    allocates (B, NF, T, T) from the module-global NF and fills channels 0..9,
    so raising NF and wrapping the original gives the expanded feature tensor
    with the E21 channels bit-identical (control 0)."""
    global FEATN_EXT, NF0, _ORIG_BUILD
    if getattr(E21, '_e32_installed', False):
        return
    _ORIG_BUILD = E21.build_feats
    NF0 = len(E21.FEATN)
    FEATN_EXT = list(E21.FEATN) + NEWN + QXK
    E21.FEATN = FEATN_EXT
    E21.NF = len(FEATN_EXT)
    # PREV_token / FIRST_token are position-only: invariant under the
    # within-sequence token shuffle, so no z is defined for them (E21 rule).
    E21.NULL_FEATS = [i for i, n in enumerate(FEATN_EXT)
                      if n not in ('PREV_token', 'FIRST_token')]
    LEAD = lead_class()
    KCLS = E21.KCLS

    def build_feats_ext(idx, mask):
        Fs = _ORIG_BUILD(idx, mask)               # channels 0..NF0-1, masked
        B, Tq = idx.shape
        i0 = NF0
        qtok = idx.unsqueeze(2)                   # (B, T, 1): query on dim 1
        prev2 = torch.roll(idx, 2, 1)
        prev2[:, :2] = -1
        Fs[:, i0] = (prev2.unsqueeze(1) == qtok).float() * mask
        eq = (idx.unsqueeze(2) == idx.unsqueeze(1))          # [b, a, c]
        strict = torch.tril(torch.ones(Tq, Tq, dtype=torch.bool,
                                       device=idx.device), -1)
        rep = (eq & strict).any(-1).float()                  # (B, T) key-side
        Fs[:, i0 + 1] = rep.unsqueeze(1).expand(B, Tq, Tq) * mask
        ld = LEAD[idx]                                       # (B, T)
        Fs[:, i0 + 2] = (ld.unsqueeze(2) == ld.unsqueeze(1)).float() * mask
        ar = torch.arange(Tq, device=idx.device)
        far = (ar[:, None] - ar[None, :]) > 1                 # i - j > 1
        Fs[:, i0 + 3] = ((idx.unsqueeze(1) == qtok) & far).float() * mask
        for a in range(len(CLS_NAMES)):
            qa = KCLS[a][idx].float().unsqueeze(2)            # query axis
            for b in range(len(CLS_NAMES)):
                kb = KCLS[b][idx].float().unsqueeze(1)        # key axis
                Fs[:, i0 + 4 + len(CLS_NAMES) * a + b] = qa * kb * mask
        return Fs

    E21.build_feats = build_feats_ext
    E21._e32_installed = True
    print(f"library: {NF0} E21 predicates + {len(NEWN)} new + {len(QXK)} "
          f"class contingencies = {len(FEATN_EXT)}", flush=True)


# ============================ controls ======================================
def control_library(E21, held):
    key = 'control0_library_backcompat'
    if E.loadj(JP).get(key, {}).get('pass'):
        print(f'{key}: cached', flush=True)
        return
    T = E21.T
    idx = held[:2, :T]
    maskf = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                  device=idx.device)).float()
    new = E21.build_feats(idx, maskf)
    E21.NF = NF0
    try:
        old = _ORIG_BUILD(idx, maskf)
    finally:
        E21.NF = len(FEATN_EXT)
    d = float((new[:, :NF0] - old).abs().max())
    kp, ks = E22R.match_kernels(idx, maskf)
    d_prev = float((new[:, 1] - kp).abs().max())
    d_same = float((new[:, 0] - ks).abs().max())
    mass = new.pow(2).sum((0, 2, 3))
    dead = [FEATN_EXT[f] for f in range(len(FEATN_EXT))
            if float(mass[f]) == 0.0]
    rec = {'n_features': len(FEATN_EXT), 'e21_channels': NF0,
           'max_abs_diff_on_e21_channels': d,
           'match_prev_vs_e22_kernel': d_prev,
           'match_same_vs_e22_kernel': d_same,
           'features_with_zero_mass_on_this_batch': dead,
           'note': 'the expanded library is a strict superset: channels 0..9 '
                   'are the E21 features unchanged, and channels 0/1 still '
                   'equal qk_e22_predbasis_run.match_kernels',
           'pass': bool(d == 0.0 and d_prev == 0.0 and d_same == 0.0)}
    E.merge(JP, key, rec)
    print(f"{key}: e21 channels {d:.1e}, kernels {d_prev:.1e}/{d_same:.1e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


def control_synthetic_e21(E21, held):
    key = 'control1_synthetic_e21_through_expanded_library'
    if key in E.loadj(JP):
        print(f'{key}: cached', flush=True)
        return
    note = ('qk_e21_census_run.control_synthetic verbatim, scored through the '
            'EXPANDED library: exact previous-token and exact same-token '
            'patterns must still name themselves')
    try:
        ctrl = E21.control_synthetic(held)
        E.merge(JP, key, {'note': note, 'result': ctrl, 'pass': True})
        print(f'{key}: PASS', flush=True)
    except AssertionError as ex:
        # E21's "no other predicate above 0.5" clause is calibrated at T=512;
        # at the smoke length T=64 MATCH_next contains the subdiagonal by
        # construction. Never tolerated in the real run.
        if not E.SMOKE:
            raise
        E.merge(JP, key, {'note': note, 'smoke_only_tolerated': str(ex)[:300],
                          'pass': False})
        print(f'{key}: smoke-length tolerance (see JSON)', flush=True)


def control_synthetic_new(E21, held):
    """Known answers for the NEW predicates: feed each predicate's own pattern
    through the exact scoring machinery; it must score ~1.0 on itself and be
    the library argmax."""
    key = 'control2_synthetic_new_predicates'
    if E.loadj(JP).get(key, {}).get('pass'):
        print(f'{key}: cached', flush=True)
        return
    T, NF = E21.T, len(FEATN_EXT)
    tgt = [FEATN_EXT.index(n) for n in NEWN] + \
          [FEATN_EXT.index(n) for n in ('Qpunct_x_Kpunct', 'Qcap_x_Kdigit',
                                        'Qfunc_x_Kfunc')]
    maskf = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                  device=E.DEV)).float()
    ar = torch.arange(T, device=E.DEV)
    off_flat = (ar[:, None] - ar[None, :]).clamp(min=0).reshape(-1)
    halves = []
    for lo, hi in ((0, E21.N_FIT), (E21.N_FIT, held.shape[0])):
        st = E21.Stats(1, len(tgt))
        for i in range(lo, hi, E21.BP):
            idx = held[i:i + E21.BP, :T]
            Fs = E21.build_feats(idx, maskf)
            st.update(0, Fs[:, tgt], Fs, off_flat)
        halves.append(st)
    sc = E21.score_heads(*halves)
    rows, ok = [], True
    for k, f in enumerate(tgt):
        s = sc['s_pred'][0, k].float().cpu().numpy()
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        best = int(np.argmax(s))
        mass = float(halves[1].bb[f])
        row = {'predicate': FEATN_EXT[f], 'self_score': round(float(s[f]), 4),
               'library_argmax': FEATN_EXT[best],
               'runner_up': FEATN_EXT[int(np.argsort(-s)[1])],
               'runner_up_score': round(float(np.sort(-s)[1] * -1), 4),
               'eval_half_squared_mass': round(mass, 1)}
        row['ok'] = bool(mass == 0.0 or
                         (s[f] > 0.99 and s[f] >= s[best] - 1e-6))
        ok = ok and row['ok']
        rows.append(row)
    rec = {'rows': rows, 'n_features': NF,
           'note': 'each new predicate\'s own causal-masked pattern, scored '
                   'through qk_e21_census_run.Stats/score_heads; features '
                   'with zero mass on the audit slice are exempt',
           'pass': bool(ok)}
    E.merge(JP, key, rec)
    for r in rows:
        print(f"  {r['predicate']}: self {r['self_score']} argmax "
              f"{r['library_argmax']} (runner-up {r['runner_up']} "
              f"{r['runner_up_score']})", flush=True)
    assert ok, f'{key} FAILED'
    print(f'{key}: PASS', flush=True)


def control_decomposition(tag, m, E21, held):
    """full - residual - named == 0 (the E31 control, per seed) plus the
    logit-level check that subtract-and-re-add through pat_hook is a no-op."""
    key = f'control3_decomposition_{tag}'
    if E.loadj(JP).get(key, {}).get('pass'):
        print(f'{key}: cached', flush=True)
        return
    T = E21.T
    idx = held[:2, :T]
    maskf = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                  device=idx.device)).float()
    resid, full = {}, {}
    with torch.no_grad():
        base = m(idx)
        m(idx, census_cb=lambda l, p: resid.__setitem__(l, p.clone()),
          census_full_cb=lambda l, p: full.__setitem__(l, p.clone()))
        Kprev, Ksame = E22R.match_kernels(idx, maskf)
        worst = 0.0
        for l in range(len(m.h)):
            terms = m.pred_terms(l, Kprev, Ksame, maskf, T)
            worst = max(worst, float((full[l] - resid[l] - terms).abs().max()))
        rt = m(idx, pat_hook=make_hook(m, idx, 'identity'))
        d_logit = float((rt - base).abs().max())
    rec = {'blocks': len(m.h), 'batch': int(idx.shape[0]), 'T': int(T),
           'max_abs_full_minus_residual_minus_named': worst,
           'max_abs_logit_diff_subtract_and_readd': d_logit,
           'note': 'the census "residual" is exactly the full pattern minus '
                   'the named terms (profile + b_h MATCH_prev + c_h '
                   'MATCH_same); qk_e31_absorption_run.control_decomposition '
                   'per seed, plus the hook round-trip used by the causal '
                   'measurements',
           'pass': bool(worst < 1e-4 and d_logit < 1e-4)}
    E.merge(JP, key, rec)
    print(f"{key}: pattern {worst:.2e}, logits {d_logit:.2e} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


# ============================ census over one model =========================
def census(name, m, E21, held):
    """qk_e21_census_run.census_model + null_zscores on the RESIDUAL view (the
    bilinear pattern before the named terms are added), through the expanded
    library. Returns (table, sc, c2r, zmat, seconds)."""
    t0 = time.time()
    fwd = E31R.make_fwd('residual')
    bp0 = E21.BP
    for bp in (bp0, 2, 1):
        try:
            E21.BP = bp
            table, sc, tpl = E21.census_model(name, m, fwd, held)
            c2r, zmat, mu_n, sd_n = E21.null_zscores(m, fwd, held)
            break
        except torch.cuda.OutOfMemoryError:
            print(f'{name}: OOM at pattern micro-batch {bp} -- retrying',
                  flush=True)
            torch.cuda.empty_cache()
    else:
        raise RuntimeError('census OOM even at micro-batch 1')
    E21.BP = bp0
    zmat = torch.nan_to_num(zmat, nan=0.0, posinf=0.0, neginf=0.0)
    for r in table:
        l, h = r['layer'], r['head']
        r['cos2_eval'] = {E21.FEATN[f]: round(float(c2r[l, h, f]), 4)
                          for f in E21.NULL_FEATS}
        r['null_z'] = {E21.FEATN[f]: round(float(zmat[l, h, f]), 1)
                       for f in E21.NULL_FEATS}
    del tpl
    if not E.SMOKE:
        torch.cuda.empty_cache()
    print(f'{name}: census done ({time.time() - t0:.0f}s, micro-batch {bp})',
          flush=True)
    return table, sc, c2r, zmat, round(time.time() - t0, 1)


def aggregate(E21, sc, c2r, zmat):
    """Per-predicate aggregation over the model's heads."""
    tokdep = set(E21.NULL_FEATS)
    out = {}
    for f, name in enumerate(FEATN_EXT):
        gain = np.nan_to_num(sc['gain'][:, :, f].float().cpu().numpy())
        cos2 = np.nan_to_num(c2r[:, :, f].float().cpu().numpy())
        z = np.nan_to_num(zmat[:, :, f].float().cpu().numpy())
        im = np.unravel_index(int(np.argmax(gain)), gain.shape)
        prog = gain >= GAIN_THR
        out[name] = {
            'token_dependent': bool(f in tokdep),
            'n_heads_gain_ge_005': int(prog.sum()),
            'n_heads_gain_ge_005_and_z_ge_3': (int((prog & (z >= Z_THR)).sum())
                                               if f in tokdep else None),
            'mean_gain': round(float(gain.mean()), 5),
            'max_gain': round(float(gain.max()), 4),
            'max_gain_head': [int(im[0]), int(im[1])],
            'total_cos2_eval': round(float(cos2.sum()), 4),
            'max_cos2_eval': round(float(cos2.max()), 4),
            'n_heads_cos2_ge_002': int((cos2 >= 0.02).sum()),
            'n_heads_z_ge_3': int((z >= Z_THR).sum()) if f in tokdep else None,
            'max_z': round(float(z.max()), 1) if f in tokdep else None}
    return out


# ============================ (b) unsupervised structure ====================
def emb_pcs(m, npc):
    W = m.wte.weight.detach().float()
    Wc = W - W.mean(0, keepdim=True)
    q = min(npc, Wc.shape[1] - 1)
    _, _, V = torch.pca_lowrank(Wc, q=q, niter=4)
    P = Wc @ V                                     # (V, q)
    P = P / P.std(0, keepdim=True).clamp(min=1e-6)
    return P


def pos_basis(T, k, dev):
    t = (torch.arange(T, device=dev).float() + 0.5) / T
    return torch.stack([torch.cos(math.pi * j * t) for j in range(k)], 1)


def heldout_r2(y, X):
    """R^2 of y ~ X on ODD positions, fit on EVEN positions (X carries no
    intercept; one is added here). Ridge-stabilized least squares."""
    T = y.shape[0]
    ones = torch.ones(T, 1, device=y.device)
    Xd = torch.cat([ones, X], 1)
    ev = torch.arange(0, T, 2, device=y.device)
    od = torch.arange(1, T, 2, device=y.device)
    A, b = Xd[ev], y[ev]
    G = A.t() @ A
    lam = 1e-6 * float(torch.diagonal(G).mean().clamp(min=1e-12))
    beta = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=y.device),
                              A.t() @ b)
    pred = Xd[od] @ beta
    yo = y[od]
    ss_res = float(((yo - pred) ** 2).sum())
    ss_tot = float(((yo - yo.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def svd_stats(pat, Xtok, Xpos):
    """pat: (B, T, T) causal-masked patterns for ONE head, one per sequence.
    Xtok: list of (T, k) token-identity regressors, one per sequence."""
    U, S, Vh = torch.linalg.svd(pat.double(), full_matrices=False)
    s2 = S ** 2
    tot = s2.sum(-1).clamp(min=1e-30)
    fr = {f'rank{r}_mass_fraction': float((s2[:, :r].sum(-1) / tot).mean())
          for r in (1, 2, 4)}
    reg = {k: [] for k in ('token_left1', 'pos_left1', 'token_right1',
                           'pos_right1', 'token_left2', 'pos_left2')}
    for b in range(pat.shape[0]):
        for tag, y in (('left1', U[b, :, 0].float()),
                       ('right1', Vh[b, 0, :].float()),
                       ('left2', U[b, :, 1].float())):
            if tag == 'left2':
                reg['token_left2'].append(heldout_r2(y, Xtok[b]))
                reg['pos_left2'].append(heldout_r2(y, Xpos))
            else:
                reg[f'token_{tag}'].append(heldout_r2(y, Xtok[b]))
                reg[f'pos_{tag}'].append(heldout_r2(y, Xpos))
    out = dict(fr)
    out.update({k: round(float(np.mean(v)), 4) for k, v in reg.items()})
    return out


def structure_model(tag, m, E21, held, nseq):
    key = f'structure_{tag}'
    if key in E.loadj(JP) and not E.SMOKE:
        print(f'{key}: cached', flush=True)
        return
    T = E21.T
    npc = 32 if not E.SMOKE else 4
    P = emb_pcs(m, npc)
    Xpos = pos_basis(T, npc, E.DEV)
    rows, t0 = [], time.time()
    bs = 2
    acc = {}
    for i in range(0, nseq, bs):
        idx = held[i:i + bs, :T]
        Xtok = [P[idx[b]] for b in range(idx.shape[0])]
        pats = {}
        with torch.no_grad():
            m(idx, census_cb=lambda l, p: pats.__setitem__(l, p.float()))
        for l, p in pats.items():
            for h in range(p.shape[1]):
                acc.setdefault((l, h), []).append(
                    svd_stats(p[:, h], Xtok, Xpos))
        del pats
    for (l, h), lst in sorted(acc.items()):
        r = {'layer': l, 'head': h}
        for k in lst[0]:
            r[k] = round(float(np.mean([d[k] for d in lst])), 4)
        tokmax = max(r['token_left1'], r['token_right1'], r['token_left2'])
        posmax = max(r['pos_left1'], r['pos_right1'], r['pos_left2'])
        r['top_vectors_explained_by'] = (
            'diffuse' if r['rank4_mass_fraction'] < 0.5 else
            'token_identity' if tokmax > posmax else 'position')
        r['token_minus_pos_r2'] = round(tokmax - posmax, 4)
        rows.append(r)
    from collections import Counter
    summ = {
        'n_heads': len(rows),
        'mean_rank1_mass_fraction': round(
            float(np.mean([r['rank1_mass_fraction'] for r in rows])), 4),
        'mean_rank2_mass_fraction': round(
            float(np.mean([r['rank2_mass_fraction'] for r in rows])), 4),
        'mean_rank4_mass_fraction': round(
            float(np.mean([r['rank4_mass_fraction'] for r in rows])), 4),
        'n_heads_rank4_ge_050': int(sum(r['rank4_mass_fraction'] >= 0.5
                                        for r in rows)),
        'n_heads_rank1_ge_050': int(sum(r['rank1_mass_fraction'] >= 0.5
                                        for r in rows)),
        'verdict_hist': dict(Counter(r['top_vectors_explained_by']
                                     for r in rows)),
        'mean_token_r2_left1': round(
            float(np.mean([r['token_left1'] for r in rows])), 4),
        'mean_pos_r2_left1': round(
            float(np.mean([r['pos_left1'] for r in rows])), 4)}
    E.merge(JP, key, {
        'checkpoint': tag, 'n_sequences': int(nseq), 'T': int(T),
        'regressors': f'{npc} embedding principal components (token identity) '
                      f'vs a {npc}-element cosine basis (position); held-out '
                      f'R^2, fit on even positions, scored on odd',
        'note': 'SVD of the RESIDUAL (bilinear-only) causal-masked pattern of '
                'each audit sequence, per head; mass fractions are of the '
                'squared Frobenius norm',
        'summary': summ, 'runtime_s': round(time.time() - t0, 1),
        'per_head': rows})
    print(f'{key}: {json.dumps(summ)}', flush=True)


def control_svd(E21, m, held):
    key = 'control4_svd_known_answers'
    if E.loadj(JP).get(key, {}).get('pass'):
        print(f'{key}: cached', flush=True)
        return
    T = E21.T
    npc = 32 if not E.SMOKE else 4
    P = emb_pcs(m, npc)
    Xpos = pos_basis(T, npc, E.DEV)
    idx = held[:2, :T]
    Xtok = [P[idx[b]] for b in range(idx.shape[0])]
    g = torch.Generator(device='cpu').manual_seed(4242)
    a = torch.randn(2, T, 1, generator=g).to(E.DEV)
    b = torch.randn(2, 1, T, generator=g).to(E.DEV)
    r1 = svd_stats(a * b, Xtok, Xpos)                      # exact rank 1
    tokpat = torch.stack([torch.outer(Xtok[i][:, 0], Xtok[i][:, 1])
                          for i in range(2)])              # g(tok_i) h(tok_j)
    tk = svd_stats(tokpat, Xtok, Xpos)
    f = Xpos[:, 3]
    pospat = torch.outer(f, f)[None].expand(2, T, T).contiguous()
    ps = svd_stats(pospat, Xtok, Xpos)
    rec = {'rank1_pattern': r1, 'token_outer_pattern': tk,
           'positional_outer_pattern': ps,
           'note': 'exact rank-1 outer product -> all mass at rank 1; a '
                   'pattern built as g(tok_i) h(tok_j) from two embedding '
                   'principal components -> token regression wins; a pattern '
                   'built as f(i) f(j) from one cosine basis element -> '
                   'positional regression wins',
           'pass': bool(r1['rank1_mass_fraction'] > 0.999
                        and tk['token_left1'] > 0.9
                        and tk['token_left1'] > tk['pos_left1']
                        and ps['pos_left1'] > 0.9
                        and ps['pos_left1'] > ps['token_left1'])}
    E.merge(JP, key, rec)
    print(f"{key}: rank1 {r1['rank1_mass_fraction']:.4f}; token pattern "
          f"tok {tk['token_left1']} vs pos {tk['pos_left1']}; pos pattern "
          f"pos {ps['pos_left1']} vs tok {ps['token_left1']} -> "
          f"{'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    assert rec['pass'], f'{key} FAILED'


# ============================ (c) causal weight =============================
def make_hook(m, x, mode, heads=None, repl=None):
    """pat_hook over the FULL pattern (residual + named).
    mode 'named_only'     -> pattern = named terms (residual deleted)
         'residual_only'  -> pattern = residual (named terms deleted)
         'identity'       -> subtract and re-add (control)
         'heads_zero'     -> residual deleted on `heads` only
         'heads_replace'  -> residual replaced by repl(l, h, x) on `heads`"""
    if mode == 'full':
        return None
    Tq = x.shape[1]
    maskf = torch.tril(torch.ones(Tq, Tq, dtype=torch.bool,
                                  device=x.device)).float()
    Kp, Ks = E22R.match_kernels(x, maskf)
    byl = {}
    for (l, h) in (heads or ()):
        byl.setdefault(l, []).append(h)

    def hook(l, pat):
        terms = m.pred_terms(l, Kp, Ks, maskf, Tq).to(pat.dtype)
        if mode == 'named_only':
            return terms.expand_as(pat).clone()
        if mode == 'residual_only':
            return pat - terms
        if mode == 'identity':
            return (pat - terms) + terms
        if l not in byl:
            return pat
        out = pat.clone()
        for h in byl[l]:
            base = terms[:, h] if terms.shape[0] > 1 else terms[0, h]
            out[:, h] = base if mode == 'heads_zero' else base + repl(l, h, x)
        return out
    return hook


@torch.no_grad()
def cond_ce(m, held, mode, T, bs=4, **kw):
    out = []
    for i in range(0, held.shape[0], bs):
        idx = held[i:i + bs]
        x = idx[:, :T]
        lg = m(x, pat_hook=make_hook(m, x, mode, **kw))
        ce = F.cross_entropy(lg.float().reshape(-1, Q.V),
                             idx[:, 1:T + 1].reshape(-1), reduction='none')
        out.append(ce.view(x.shape[0], T).mean(1).cpu())
    return torch.cat(out)


def induction_probe(P, nseq):
    rows = np.asarray(np.load(E.HELD_PATH, mmap_mode='r')
                      [ROWS[0]:ROWS[0] + nseq]).astype(np.int64)
    pref = torch.from_numpy(rows[:, 1:1 + P]).to(E.DEV)
    ev = torch.cat([pref, pref], 1)
    idx, tgt = ev[:, :-1], ev[:, 1:]
    fir = torch.arange(1, P - 1, device=E.DEV)
    sec = torch.arange(P + 1, 2 * P - 1, device=E.DEV)
    assert torch.equal(tgt[:, fir], tgt[:, sec])
    return idx, tgt, fir, sec


@torch.no_grad()
def cond_induction(m, probe, mode, bs=8, **kw):
    idx, tgt, fir, sec = probe
    ces = []
    for i in range(0, idx.shape[0], bs):
        x, t = idx[i:i + bs], tgt[i:i + bs]
        lg = m(x, pat_hook=make_hook(m, x, mode, **kw)).float()
        lsm = torch.log_softmax(lg, -1)
        ces.append(-lsm.gather(-1, t[..., None])[..., 0])
    ce = torch.cat(ces)
    per_seq = ce[:, fir].mean(1) - ce[:, sec].mean(1)
    return {'ce_first': round(float(ce[:, fir].mean()), 4),
            'ce_second': round(float(ce[:, sec].mean()), 4),
            'induction_advantage': round(float(per_seq.mean()), 4),
            'se_seq': round(float(per_seq.std(unbiased=True)
                                  / math.sqrt(len(per_seq))), 4)}, per_seq


def paired(a, b):
    d = a - b
    return (round(float(d.mean()), 5),
            round(float(d.std(unbiased=True) / math.sqrt(len(d))), 5))


def causal_weight(tag, m, E21, held):
    key = f'causal_weight_{tag}'
    if key in E.loadj(JP) and not E.SMOKE:
        print(f'{key}: cached', flush=True)
        return E.loadj(JP)[key]
    T = E21.T
    P = 64 if not E.SMOKE else 8
    nseq = 96 if not E.SMOKE else 8
    probe = induction_probe(P, nseq)
    ce = {mode: cond_ce(m, held, mode, T) for mode in
          ('full', 'named_only', 'residual_only')}
    ind, inds = {}, {}
    for mode in ('full', 'named_only', 'residual_only'):
        ind[mode], inds[mode] = cond_induction(m, probe, mode)
    d_res, se_res = paired(ce['named_only'], ce['full'])
    d_nam, se_nam = paired(ce['residual_only'], ce['full'])
    di_res, sei_res = paired(inds['named_only'], inds['full'])
    di_nam, sei_nam = paired(inds['residual_only'], inds['full'])
    rec = {
        'checkpoint': tag,
        'audit_slice': f'fresh34k[{ROWS[0]}:{ROWS[1]}], T={T}',
        'induction_probe': f'{nseq} held prefixes of {P} tokens, repeated '
                           'once; advantage = CE(first copy) - CE(second '
                           'copy) on identical targets (qk_e28 convention)',
        'held_ce': {k: round(float(v.mean()), 4) for k, v in ce.items()},
        'delta_ce_residual_zeroed': d_res, 'delta_ce_residual_zeroed_se': se_res,
        'delta_ce_named_zeroed': d_nam, 'delta_ce_named_zeroed_se': se_nam,
        'induction': ind,
        'delta_induction_residual_zeroed': di_res,
        'delta_induction_residual_zeroed_se': sei_res,
        'delta_induction_named_zeroed': di_nam,
        'delta_induction_named_zeroed_se': sei_nam,
        'named_share_of_ce_cost': round(d_nam / max(d_nam + d_res, 1e-9), 4),
        'e28_reference_all_b_zeroed_induction_delta': -1.593,
        'note': 'residual zeroed = every head\'s pattern is exactly its named '
                'terms (signed positional profile + b_h MATCH_prev + c_h '
                'MATCH_same); named zeroed = every head\'s pattern is exactly '
                'the learned bilinear residual. The E28 reference zeroed only '
                'the b_h coefficients, so it is a subset of "named zeroed".'}
    E.merge(JP, key, rec)
    print(f"{key}: dCE residual-zeroed {d_res:+.4f} (se {se_res}), "
          f"named-zeroed {d_nam:+.4f} (se {se_nam}); induction "
          f"{ind['full']['induction_advantage']} -> "
          f"{ind['named_only']['induction_advantage']} (residual zeroed) / "
          f"{ind['residual_only']['induction_advantage']} (named zeroed)",
          flush=True)
    return rec


def candidate_causal(tag, m, E21, sc, zmat, held, cands, base_rec):
    """For each candidate predicate: on its programmatic heads, (i) zero the
    residual, (ii) replace it with the fitted a * predicate + b * profile,
    (iii) replace it with profile only. Recovery = the fraction of the
    residual-zeroing CE cost that the fitted predicate buys back."""
    key = f'candidate_causal_{tag}'
    if key in E.loadj(JP) and not E.SMOKE:
        print(f'{key}: cached', flush=True)
        return
    T = E21.T
    ar = torch.arange(T, device=E.DEV)
    offmat = (ar[:, None] - ar[None, :]).clamp(min=0)
    ce0 = cond_ce(m, held, 'full', T)
    probe = induction_probe(64 if not E.SMOKE else 8, 96 if not E.SMOKE else 8)
    _, ind0 = cond_induction(m, probe, 'full')
    _fc, _pc = {}, {}                            # length-generic small caches

    def maskq(x):
        Tq = x.shape[1]
        return torch.tril(torch.ones(Tq, Tq, dtype=torch.bool,
                                     device=x.device)).float()

    def featcol(x, f):
        k = (x.data_ptr(), int(x.shape[0]), int(x.shape[1]), f)
        if k not in _fc:
            _fc.clear()
            _fc[k] = E21.build_feats(x, maskq(x))[:, f]
        return _fc[k]

    def profpat(l, h, x):
        Tq = x.shape[1]
        k = (l, h, Tq)
        if k not in _pc:
            _pc[k] = (sc['prof'][l, h].float()[offmat[:Tq, :Tq]]
                      * maskq(x))
        return _pc[k]

    rows = []
    for name in cands:
        f = FEATN_EXT.index(name)
        gain = np.nan_to_num(sc['gain'][:, :, f].float().cpu().numpy())
        z = np.nan_to_num(zmat[:, :, f].float().cpu().numpy())
        cand = [(int(l), int(h)) for l in range(gain.shape[0])
                for h in range(gain.shape[1])
                if gain[l, h] >= GAIN_THR and z[l, h] >= Z_THR]
        cand.sort(key=lambda lh: -gain[lh])
        heads = cand[:8]
        if not heads:
            rows.append({'predicate': name, 'n_heads': 0,
                         'note': 'no head clears gain >= 0.05 with z >= 3'})
            continue
        A = {lh: float(sc['joint_a'][lh[0], lh[1], f]) for lh in heads}
        B = {lh: float(sc['joint_b'][lh[0], lh[1], f]) for lh in heads}

        def repl_fit(l, h, x, f=f, A=A, B=B):
            return (A[(l, h)] * featcol(x, f)
                    + B[(l, h)] * profpat(l, h, x))

        def repl_prof(l, h, x):
            return profpat(l, h, x)

        out = {'predicate': name, 'n_heads': len(heads),
               'heads': [[l, h] for l, h in heads],
               'gain': [round(float(gain[lh]), 4) for lh in heads],
               'z': [round(float(z[lh]), 1) for lh in heads],
               'joint_coef_predicate': [round(A[lh], 5) for lh in heads],
               'joint_coef_profile': [round(B[lh], 5) for lh in heads]}
        for mode, kw, lab in (
                ('heads_zero', {}, 'zero'),
                ('heads_replace', {'repl': repl_fit}, 'fit'),
                ('heads_replace', {'repl': repl_prof}, 'profile_only')):
            ce = cond_ce(m, held, mode, T, heads=heads, **kw)
            d, se = paired(ce, ce0)
            out[f'dce_{lab}'] = d
            out[f'dce_{lab}_se'] = se
            _, ps = cond_induction(m, probe, mode, heads=heads, **kw)
            di, sei = paired(ps, ind0)
            out[f'dinduction_{lab}'] = di
        dz = out['dce_zero']
        out['recovery_fraction_of_residual_cost'] = (
            round(1.0 - out['dce_fit'] / dz, 4) if abs(dz) > 1e-6 else None)
        out['beats_profile_only'] = bool(
            out['dce_fit'] < out['dce_profile_only']
            - 2 * (out['dce_fit_se'] + out['dce_profile_only_se']))
        rows.append(out)
        print(f"  {name} ({len(heads)} heads): zero {out['dce_zero']:+.4f} "
              f"fit {out['dce_fit']:+.4f} profile-only "
              f"{out['dce_profile_only']:+.4f} -> recovery "
              f"{out['recovery_fraction_of_residual_cost']}", flush=True)
    E.merge(JP, key, {
        'checkpoint': tag,
        'note': 'per candidate predicate, on the heads where it is '
                'programmatic in the RESIDUAL census (gain >= 0.05 and '
                'shuffled-null z >= 3, top 8 by gain): the named terms are '
                'kept throughout and only the residual is manipulated; '
                'recovery = 1 - dCE(fit)/dCE(zero)',
        'baseline_ce': base_rec['held_ce']['full'], 'rows': rows})


# ============================ the recommendation ============================
def recommend(per_seed_agg, floor_agg, causal, cand_causal):
    seeds = sorted(per_seed_agg)
    rows = []
    for name in FEATN_EXT:
        if name in ALREADY_NAMED:
            continue
        agg = [per_seed_agg[s][name] for s in seeds]
        fl = floor_agg.get(name, {})
        nprog = [a['n_heads_gain_ge_005_and_z_ge_3']
                 if a['n_heads_gain_ge_005_and_z_ge_3'] is not None
                 else a['n_heads_gain_ge_005'] for a in agg]
        tot = [a['total_cos2_eval'] for a in agg]
        rows.append({
            'predicate': name,
            'n_programmatic_heads_per_seed': nprog,
            'n_programmatic_heads_min_over_seeds': int(min(nprog)),
            'n_programmatic_heads_mean': round(float(np.mean(nprog)), 2),
            'total_residual_cos2_per_seed': tot,
            'total_residual_cos2_mean': round(float(np.mean(tot)), 4),
            'untrained_floor_total_cos2': fl.get('total_cos2_eval'),
            'untrained_floor_n_programmatic': fl.get('n_heads_gain_ge_005'),
            'over_floor_total_cos2': round(
                float(np.mean(tot)) - float(fl.get('total_cos2_eval') or 0.0),
                4),
            'max_gain_per_seed': [a['max_gain'] for a in agg],
            'max_z_per_seed': [a['max_z'] for a in agg],
            'replicated_all_seeds': bool(min(nprog) > 0)})
    rows.sort(key=lambda r: (-r['n_programmatic_heads_min_over_seeds'],
                             -r['n_programmatic_heads_mean'],
                             -r['over_floor_total_cos2']))
    for i, r in enumerate(rows):
        r['rank'] = i + 1
    cc = {r['predicate']: r for r in (cand_causal or {}).get('rows', [])}
    top = []
    for r in rows[:3]:
        e = {k: r[k] for k in ('rank', 'predicate',
                               'n_programmatic_heads_per_seed',
                               'total_residual_cos2_mean',
                               'over_floor_total_cos2', 'max_z_per_seed',
                               'replicated_all_seeds')}
        c = cc.get(r['predicate'])
        if c:
            e['causal'] = {k: c.get(k) for k in
                           ('n_heads', 'dce_zero', 'dce_fit',
                            'dce_profile_only',
                            'recovery_fraction_of_residual_cost',
                            'beats_profile_only')}
        top.append(e)
    return rows, top


def summarize(E21, per_seed_agg, floor_agg, causal, cand_causal, structure):
    rows, top = recommend(per_seed_agg, floor_agg, causal, cand_causal)
    seeds = sorted(causal)
    cw = {s: {'delta_ce_residual_zeroed': causal[s]['delta_ce_residual_zeroed'],
              'delta_ce_named_zeroed': causal[s]['delta_ce_named_zeroed'],
              'delta_induction_residual_zeroed':
                  causal[s]['delta_induction_residual_zeroed'],
              'delta_induction_named_zeroed':
                  causal[s]['delta_induction_named_zeroed'],
              'named_share_of_ce_cost': causal[s]['named_share_of_ce_cost']}
          for s in seeds}
    out = {
        'question': 'after the named terms (signed positional profile + '
                    'MATCH_prev + MATCH_same) have absorbed what they can, '
                    'what is the learned bilinear residual pattern still '
                    'doing, and which predicate should the library grow next?',
        'models': dict(MODELS), 'library_size': len(FEATN_EXT),
        'library': FEATN_EXT,
        'already_named': list(ALREADY_NAMED),
        'causal_weight_per_seed': cw,
        'causal_weight_reading':
            'delta CE with the residual zeroed is what the learned residual '
            'is worth on top of the named library; delta CE with the named '
            'terms zeroed is what the library is worth on top of the '
            'residual. Their ratio is the honest split.',
        'structure_summary': {s: (structure.get(s) or {}).get('summary')
                              for s in structure},
        'next_predicate_ranked': rows,
        'top3_recommendation': top,
        'ranking_rule': 'primary: the number of heads where the predicate is '
                        'programmatic (joint-fit gain over the positional '
                        'profile >= 0.05) AND clears the shuffled-token null '
                        'at z >= 3, taken as the MINIMUM over the three '
                        'seeds (replication first); ties broken by the mean '
                        'over seeds and then by total residual cos^2 above '
                        'the untrained-init floor'}
    E.merge(JP, 'summary_E32', out)
    print(json.dumps({'top3_recommendation': top,
                      'causal_weight_per_seed': cw}, indent=2), flush=True)


# ============================ main ==========================================
def main():
    t0 = time.time()
    if 'summary_E32' in E.loadj(JP) and not E.SMOKE:
        print('summary_E32 already recorded -- nothing to do', flush=True)
        print('e32 residual mine run done', flush=True)
        return
    E.setup()
    E.DEV = 'cpu' if E.SMOKE else 'cuda'
    s_c, _ = E15R.solve_slot_c(4 * Q.D)
    if not E.SMOKE:
        assert s_c == 15, s_c
    E21 = E22R.get_e21()
    if E.SMOKE:
        E21.N_FIT = 8
        E21.NULL_K = 3
        E21.BP = 2
    else:
        E21.BP = 2                              # 39 features x 21 null copies
    install_library(E21)

    n_rows = 16 if E.SMOKE else (ROWS[1] - ROWS[0])
    held = torch.from_numpy(
        np.load(E.HELD_PATH)[ROWS[0]:ROWS[0] + n_rows].astype(np.int64)
    ).to(E.DEV)
    print(f"held: fresh34k[{ROWS[0]}:{ROWS[0] + n_rows}] on {E.DEV} "
          f"(the fixed E21/E31 audit slice)", flush=True)

    if 'E32_config' not in E.loadj(JP):
        E.merge(JP, 'E32_config', {
            'models': dict(MODELS),
            'library': FEATN_EXT, 'n_features': len(FEATN_EXT),
            'e21_library_size': NF0,
            'added': NEWN + QXK,
            'audit_slice': f'fresh34k[{ROWS[0]}:{ROWS[1]}]',
            'fit_eval_split': f'fit seqs 0:{E21.N_FIT}, eval rest',
            'null': f'{E21.NULL_K} within-sequence key-token shuffles '
                    '(qk_e21_census_run.null_zscores verbatim)',
            'programmatic_criterion': f'joint-fit gain over the positional '
                                      f'profile >= {GAIN_THR}, z >= {Z_THR}',
            'device': E.DEV})

    # ---- controls ----
    control_library(E21, held)
    control_synthetic_e21(E21, held)
    control_synthetic_new(E21, held)

    mu0 = E22R.make_e22(s=s_c).eval().float()
    control_svd(E21, mu0, held)

    # ---- untrained-init floor (cheap, and every candidate must clear it) ----
    floor_agg = None
    if 'untrained_floor' in E.loadj(JP) and not E.SMOKE:
        floor_agg = E.loadj(JP)['untrained_floor']['per_predicate']
        del mu0
    else:
        t, sc_u, c2_u, z_u, secs = census('UNTRAINED_init', mu0, E21, held)
        floor_agg = aggregate(E21, sc_u, c2_u, z_u)
        E.merge(JP, 'untrained_floor', {
            'architecture': 'qk_e22_predbasis_run.make_e22 fresh init '
                            '(predicate parameters are exact zeros, so the '
                            'residual view IS the full pattern)',
            'summary': E21.summarize(t), 'runtime_s': secs,
            'per_predicate': floor_agg})
        print(f"untrained floor: {json.dumps(E21.summarize(t))}", flush=True)
        del mu0, sc_u, c2_u, z_u
        if not E.SMOKE:
            torch.cuda.empty_cache()

    # ---- per-seed mining ----
    per_seed_agg, causal, structure, cand_causal = {}, {}, {}, None
    for tag, stem in MODELS:
        if not E.SMOKE and not os.path.exists(E.ckpath(stem)):
            print(f'{tag}: {stem}.pt missing -- skipped', flush=True)
            continue
        print(f'==== {tag} ({stem}) ====', flush=True)
        if E.SMOKE:
            m = E22R.make_e22(s=s_c).eval().float()
        else:
            m, _ = E.load_arm(stem, lambda: E22R.make_e22(s=s_c))
        m.eval().float()
        control_decomposition(tag, m, E21, held)
        table, sc, c2r, zmat, secs = census(f'residual_{tag}', m, E21, held)
        agg = aggregate(E21, sc, c2r, zmat)
        per_seed_agg[tag] = agg
        E.merge(JP, f'residual_census_{tag}', {
            'checkpoint': f'{stem}.pt',
            'pattern_view': 'the bilinear-only residual pattern (named terms '
                            'NOT included) -- qk_e31_absorption_run.make_fwd '
                            '"residual"',
            'held_rows': f'fresh34k[{ROWS[0]}:{ROWS[0] + n_rows}]',
            'summary': E21.summarize(table), 'runtime_s': secs,
            'per_predicate': agg, 'table': table})
        print(f"{tag} residual census summary: "
              f"{json.dumps(E21.summarize(table))}", flush=True)
        structure_model(tag, m, E21, held, 16 if not E.SMOKE else 4)
        structure[tag] = E.loadj(JP).get(f'structure_{tag}')
        causal[tag] = causal_weight(tag, m, E21, held)
        if tag == 's0':
            rows, top = recommend(per_seed_agg, floor_agg, causal, None)
            cands = [r['predicate'] for r in rows[:3]]
            print(f'seed-0 provisional candidates: {cands}', flush=True)
            candidate_causal(tag, m, E21, sc, zmat, held, cands, causal[tag])
            cand_causal = E.loadj(JP).get(f'candidate_causal_{tag}')
        del m, sc, c2r, zmat, table
        if not E.SMOKE:
            torch.cuda.empty_cache()

    # the top-3 are re-derived over all seeds; the causal rows are seed 0's
    summarize(E21, per_seed_agg, floor_agg, causal, cand_causal, structure)
    E.merge(JP, 'E32_runtime_s', round(time.time() - t0, 1))
    print(f'e32 residual mine run done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
