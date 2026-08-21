"""FRONT COMPONENT FUNCTION v2 -- fix 710's flawed null and properly test
CATEGORY SPECIALIZATION beyond the easy-vs-hard difficulty profile. The
concern: hard categories (high baseline CE) show bigger absolute deltas
just because there's more room. To test whether a component SPECIALIZES,
normalize each component's per-category delta by that category's baseline
CE (fractional CE increase = delta_c / baseline_c), then compare the
PROFILE SHAPES across components via pairwise correlation.

If all components share ONE shape (high pairwise correlation), there is NO
specialization beyond difficulty -- the front is broad, differing only in
magnitude. If some component's shape deviates (low/negative correlation),
that is real category specialization.

REGISTERED PREDICTIONS:
  (0) SANITY: reproduce 710's absolute deltas;
  (a) TEST (no strong prior): report pairwise profile correlations. If mean
      pairwise corr >= 0.8 -> conclude BROAD (no specialization beyond
      difficulty); if some pair < 0.3 -> real specialization. Register the
      expectation (from 710) that attention vs mlp MAY differ but the big
      components share the hard-open-vocab shape;
  (b) report normalized profiles + correlation matrix;
  NULL (consistent this time): split eval positions into two random halves,
      compute each component's profile on each half; the SAME component's
      two-half profiles must correlate MUCH higher than different components
      (within-component >> between-component) for any between-difference to
      be real."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_component_function_v2_results.json'
NEVAL = 48
FUNC = {'the', 'a', 'an', 'of', 'to', 'in', 'and', 'is', 'was', 'are', 'for',
        'with', 'that', 'as', 'on', 'at', 'by', 'it', 'be', 'or', 'this', 'from',
        'his', 'her', 'their', 'they', 'he', 'she', 'i', 'you', 'we', 'but', 'not',
        'have', 'has', 'had', 'were', 'been', 'which', 'who', 'will', 'would'}
CATS = ['newline', 'punct', 'digit', 'cap_word', 'func_word', 'content', 'subword']


def categorize(t):
    if '\n' in t: return 'newline'
    s = t.strip()
    if s == '': return 'punct'
    if not t.startswith(' ') and not t.startswith('\n'):
        if s[0].isdigit(): return 'digit'
        if not s[0].isalnum(): return 'punct'
        return 'subword'
    core = t.lstrip()
    if core == '': return 'punct'
    if core[0].isdigit(): return 'digit'
    if not core[0].isalnum(): return 'punct'
    if core[0].isupper(): return 'cap_word'
    if core.lower() in FUNC: return 'func_word'
    return 'content'


def hook_zero(mo, i_, o_):
    return torch.zeros_like(o_)


@torch.no_grad()
def per_pos_ce(rows, n):
    ces = []
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1).reshape(-1, logits.shape[-1])
        ces.append(F.nll_loss(lp, tgt.reshape(-1), reduction='none').cpu().numpy())
    return np.concatenate(ces)


def cat_means(ce, cats, mask=None):
    out = {}
    for c in CATS:
        sel = np.array([cats[i] == c and (mask is None or mask[i]) for i in range(len(cats))])
        out[c] = float(ce[sel].mean()) if sel.any() else np.nan
    return out


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    cats = []
    for i in range(NEVAL):
        for tt in rows[i, 1:257].tolist():
            cats.append(categorize(cl.d1(int(tt))))
    cats = cats[:NEVAL * 256]
    rng = np.random.default_rng(0)
    half = rng.random(len(cats)) < 0.5

    base_ce = per_pos_ce(rows, NEVAL)
    base = cat_means(base_ce, cats)
    comps = [('block0.attn', m.transformer.h[0].attn.c_proj),
             ('block0.mlp', m.transformer.h[0].mlp),
             ('block1.attn', m.transformer.h[1].attn.c_proj),
             ('block1.mlp', m.transformer.h[1].mlp),
             ('block2.attn', m.transformer.h[2].attn.c_proj),
             ('block2.mlp', m.transformer.h[2].mlp)]
    prof = {}; prof_h0 = {}; prof_h1 = {}
    for name, mod in comps:
        h = mod.register_forward_hook(hook_zero)
        ace = per_pos_ce(rows, NEVAL)
        h.remove()
        # fractional CE increase per category (normalize out difficulty)
        full = cat_means(ace, cats)
        prof[name] = np.array([(full[c] - base[c]) / base[c] for c in CATS])
        a0 = cat_means(ace, cats, half); a1 = cat_means(ace, cats, ~half)
        b0 = cat_means(base_ce, cats, half); b1 = cat_means(base_ce, cats, ~half)
        prof_h0[name] = np.array([(a0[c] - b0[c]) / b0[c] for c in CATS])
        prof_h1[name] = np.array([(a1[c] - b1[c]) / b1[c] for c in CATS])
        print(f'{name:12s} frac-CE profile: ' +
              '  '.join(f'{c[:4]} {prof[name][k]:+.2f}' for k, c in enumerate(CATS)), flush=True)

    names = [n for n, _ in comps]
    # pairwise correlation of normalized profiles (between-component)
    M = np.array([prof[n] for n in names])
    corr = np.corrcoef(M)
    off = corr[np.triu_indices(len(names), 1)]
    mean_between = float(np.nanmean(off))
    # within-component reliability (two halves)
    within = [float(np.corrcoef(prof_h0[n], prof_h1[n])[0, 1]) for n in names]
    mean_within = float(np.nanmean(within))
    print('\nbetween-component profile corr (mean off-diag):', round(mean_between, 3), flush=True)
    print('within-component split-half reliability:', [round(w, 2) for w in within], flush=True)

    broad = mean_between >= 0.8
    reliable = mean_within >= 0.7
    print(f'\n(a) components share ONE shape (mean between-corr {mean_between:.2f} >=0.8 '
          f'-> BROAD, no specialization): {broad}', flush=True)
    print(f'NULL within>>between meaningful (within {mean_within:.2f}): {reliable}', flush=True)

    out = {'baseline_ce': {c: round(base[c], 3) for c in CATS},
           'frac_profiles': {n: [round(float(x), 4) for x in prof[n]] for n in names},
           'between_corr_mean': round(mean_between, 4),
           'within_reliability': [round(w, 4) for w in within],
           'within_mean': round(mean_within, 4), 'cats': CATS,
           'conclusion_broad': bool(broad), 'null_reliable': bool(reliable),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
