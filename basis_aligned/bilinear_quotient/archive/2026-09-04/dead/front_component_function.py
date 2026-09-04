"""FRONT COMPONENT FUNCTION -- 'understand the first few layers entirely'
(user goal), in plain language. For each component of blocks 0-2 (attn
c_proj output + mlp output = 6), zero its output and measure which
next-token CATEGORY loses the most CE. The category a component most helps
= what that component is FOR. Complements the rank work (699-709) with a
functional 'what does it do' map, in ordinary words.

Target-token categories (what is being predicted at each position):
  newline, punctuation, digit, capitalized-word (leading-space + Upper),
  function-word (the/a/of/to/is/and/...), content-word (leading-space +
  lower, not function), subword-continuation (no leading space).

REGISTERED PREDICTIONS:
  (0) SANITY: baseline CE reproduces; category buckets are non-empty;
  (a) DIFFERENTIATED ROLES: different components most-help DIFFERENT
      categories (the top category is not identical across all 6) -- the
      front is a division of labor, not redundant copies. Expectation from
      634/701/703: attention components help boundary/newline+continuation,
      mlp0 helps the broad class/content decision;
  (b) report the component x category CE-increase table;
  NULL: SHUFFLING the category labels flattens the per-category spread (the
      real spread across categories >> shuffled spread) -- the
      differentiation is real structure, not noise."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_component_function_results.json'
NEVAL = 48
FUNC = {'the', 'a', 'an', 'of', 'to', 'in', 'and', 'is', 'was', 'are', 'for',
        'with', 'that', 'as', 'on', 'at', 'by', 'it', 'be', 'or', 'this', 'from',
        'his', 'her', 'their', 'they', 'he', 'she', 'i', 'you', 'we', 'but', 'not',
        'have', 'has', 'had', 'were', 'been', 'which', 'who', 'will', 'would'}
CATS = ['newline', 'punct', 'digit', 'cap_word', 'func_word', 'content', 'subword']


def categorize(tokstr):
    if tokstr in ('\n', '\n\n') or '\n' in tokstr:
        return 'newline'
    s = tokstr.strip()
    if s == '':
        return 'punct'
    if not tokstr.startswith(' ') and not tokstr.startswith('\n'):
        # no leading space -> continuation, unless pure punctuation/digit
        if s[0].isdigit():
            return 'digit'
        if not s[0].isalnum():
            return 'punct'
        return 'subword'
    core = tokstr.lstrip()
    if core == '':
        return 'punct'
    if core[0].isdigit():
        return 'digit'
    if not core[0].isalnum():
        return 'punct'
    if core[0].isupper():
        return 'cap_word'
    if core.lower() in FUNC:
        return 'func_word'
    return 'content'


def hook_zero(mo, i_, o_):
    return torch.zeros_like(o_)


@torch.no_grad()
def per_cat_ce(rows, n, cats_by_pos):
    """Return dict cat -> mean CE, using the currently-registered hooks."""
    sums = {c: 0.0 for c in CATS}; cnts = {c: 0 for c in CATS}
    p = 0
    for i in range(0, n, 4):
        bb = rows[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        lp = F.log_softmax(logits.float(), -1).reshape(-1, logits.shape[-1])
        tt = tgt.reshape(-1)
        ce = F.nll_loss(lp, tt, reduction='none').cpu().numpy()
        for k in range(len(ce)):
            c = cats_by_pos[p + k]
            sums[c] += float(ce[k]); cnts[c] += 1
        p += len(ce)
    return {c: (sums[c] / cnts[c] if cnts[c] else float('nan')) for c in CATS}, cnts


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    # categorize all eval target tokens
    cats = []
    for i in range(NEVAL):
        for tt in rows[i, 1:257].tolist():
            cats.append(categorize(cl.d1(int(tt))))
    cats = cats[:NEVAL * 256]
    print('category counts:', {c: cats.count(c) for c in CATS}, flush=True)

    base, cnts = per_cat_ce(rows, NEVAL, cats)
    print('baseline per-cat CE:', {c: round(base[c], 2) for c in CATS}, flush=True)

    comps = [('block0.attn', m.transformer.h[0].attn.c_proj),
             ('block0.mlp', m.transformer.h[0].mlp),
             ('block1.attn', m.transformer.h[1].attn.c_proj),
             ('block1.mlp', m.transformer.h[1].mlp),
             ('block2.attn', m.transformer.h[2].attn.c_proj),
             ('block2.mlp', m.transformer.h[2].mlp)]
    table = {}
    for name, mod in comps:
        h = mod.register_forward_hook(hook_zero)
        abl, _ = per_cat_ce(rows, NEVAL, cats)
        h.remove()
        delta = {c: round(abl[c] - base[c], 3) for c in CATS}
        top = max(CATS, key=lambda c: delta[c] if cnts[c] > 20 else -1)
        table[name] = {'delta': delta, 'top_category': top}
        print(f'{name:12s} top={top:9s}  ' +
              '  '.join(f'{c[:4]} {delta[c]:+.2f}' for c in CATS), flush=True)

    tops = [table[n]['top_category'] for n in table]
    pa = len(set(tops)) >= 2
    # null: shuffle category labels, recompute spread for one component
    rng = np.random.default_rng(0)
    csh = cats.copy(); rng.shuffle(csh)
    h = comps[1][1].register_forward_hook(hook_zero)   # block0.mlp
    abl_sh, _ = per_cat_ce(rows, NEVAL, csh)
    h.remove()
    real_spread = max(table['block0.mlp']['delta'].values()) - min(table['block0.mlp']['delta'].values())
    sh_delta = {c: abl_sh[c] - base[c] for c in CATS}   # base uses real cats; approx null
    sh_spread = max(sh_delta.values()) - min(sh_delta.values())
    null_ok = real_spread > 1.5 * sh_spread
    print(f'\n(a) differentiated roles (tops {tops}): {pa}', flush=True)
    print(f'NULL real spread {real_spread:.2f} > 1.5x shuffled {sh_spread:.2f}: {null_ok}',
          flush=True)

    out = {'baseline_ce': {c: round(base[c], 3) for c in CATS},
           'category_counts': {c: cnts[c] for c in CATS}, 'table': table,
           'tops': tops, 'pred_a_differentiated': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
