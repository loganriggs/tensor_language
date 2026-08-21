"""WRITE CLASS DEPTH MAP -- a causal "who writes what, where" map. For
each token CLASS and each of the 18 blocks, how much does that block
write that class? Opens a new thread after the calibration line (624-
628): what do the frequency-neutral MIDDLE blocks (10-16) actually do?

624 gave the causal writer profile for newline+article (early writers,
block 17 calibrates); 627 found the middle blocks 10-16 are frequency-
neutral. This generalizes 624's causal method (mean-ablate a block,
measure the P(class) drop at class-target positions) to a panel of
token classes, producing a (classes x 18) map of writing depth. It
answers: are different token TYPES written at different depths? And do
the middle blocks write anything specific (e.g. content/rare classes),
or are they genuinely idle for next-token identity?

REGISTERED PREDICTIONS:
  (0) SANITY: for each class the aggregate |P-drop| is larger at class-
      target positions than at non-class positions (the map is class-
      specific / trustworthy) -- classes failing this are flagged, not
      trusted (the 623 rule);
  (a) FUNCTION CLASSES EARLY: high-frequency function classes
      (determiner, punct) have their largest positive writer in the
      front blocks (0-3), as 614/624 found;
  (b) BLOCK 17 SUPPRESSES BROADLY: block 17's effect is negative
      (suppressor) for the high-frequency function classes, consistent
      with 624-628 (it calibrates them down);
  (c) MIDDLE BLOCKS: report whether any class has its top writer in the
      middle band (blocks 10-16). Registered guess: content-ish classes
      (capitalized, subword) draw more on middle/late blocks than the
      function classes do -- if so, the middle writes content identity,
      not function tokens;
  NULL: per class, ablating a block changes P(class) at class-target
      positions more than it changes P(class) at random non-target
      positions (already encoded in the sanity specificity check)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'write_class_depth_map_results.json'
NFRESH = 48
NB = 18

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}


def classify(s):
    if chr(10) in s:
        return 'newline'
    t = s.strip().lower()
    if not t:
        return 'space'
    if t in DET:
        return 'determiner'
    if t in PREP:
        return 'preposition'
    if t in PRON:
        return 'pronoun'
    if t[0].isdigit():
        return 'digit'
    if all(not c.isalnum() for c in t):
        return 'punct'
    if s.strip()[:1].isupper():
        return 'capitalized'
    if s.startswith(' '):
        return 'space_word'
    return 'subword'


CLASSES = ['newline', 'determiner', 'preposition', 'pronoun', 'digit',
           'punct', 'capitalized', 'space_word', 'subword']


@torch.no_grad()
def run(fresh, ablate_block, vocab_mask, target_mask):
    """Return (C,) mean P(class) at class-target positions and (C,) mean
    P(class) at non-target positions."""
    C = vocab_mask.shape[1]
    sum_at = np.zeros(C); n_at = np.zeros(C)
    sum_ot = np.zeros(C); n_ot = np.zeros(C)
    pos0 = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x_in = x
            x, v1 = blk(x, v1, x0)
            if ablate_block is not None and li == ablate_block:
                delta = x - x_in
                x = x_in + delta.mean(dim=(0, 1), keepdim=True)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])   # (B*T, V)
        pc = (p @ vocab_mask).cpu().numpy()                   # (B*T, C)
        npos = pc.shape[0]
        tm = target_mask[pos0:pos0 + npos]                    # (npos, C) bool
        sum_at += (pc * tm).sum(0); n_at += tm.sum(0)
        sum_ot += (pc * ~tm).sum(0); n_ot += (~tm).sum(0)
        pos0 += npos
    return (sum_at / np.maximum(n_at, 1), sum_ot / np.maximum(n_ot, 1))


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    C = len(CLASSES)
    cidx = {c: k for k, c in enumerate(CLASSES)}

    # vocab class membership mask (V, C)
    vmask = torch.zeros(V, C)
    for t in range(V):
        c = classify(cl.d1(t))
        if c in cidx:
            vmask[t, cidx[c]] = 1.0
    vmask = vmask.to(DEV)

    # per-position target class mask (Npos, C)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    tmask = np.zeros((len(nxt), C), dtype=bool)
    for j, t in enumerate(nxt):
        c = classify(cl.d1(int(t)))
        if c in cidx:
            tmask[j, cidx[c]] = True
    counts = tmask.sum(0)
    print('class counts: '
          + ', '.join(f'{c}={int(counts[cidx[c]])}' for c in CLASSES), flush=True)

    base_at, base_ot = run(fresh, None, vmask, tmask)
    drops_at = np.zeros((NB, C)); drops_ot = np.zeros((NB, C))
    for L in range(NB):
        ab_at, ab_ot = run(fresh, L, vmask, tmask)
        drops_at[L] = base_at - ab_at        # >0 = block writes class
        drops_ot[L] = base_ot - ab_ot
        top = CLASSES[int(np.argmax(np.abs(drops_at[L])))]
        print(f'  block {L:2d}: top-|effect| class {top:12s} '
              f'({drops_at[L][cidx[top]]:+.4f})', flush=True)

    out = {'classes': CLASSES, 'baseline_P_at_class':
           {c: round(float(base_at[cidx[c]]), 4) for c in CLASSES}, 'per_class': {}}
    for c in CLASSES:
        k = cidx[c]
        col = drops_at[:, k]
        order = np.argsort(-np.abs(col))
        top3 = [(int(b), round(float(col[b]), 4)) for b in order[:3]]
        top_writer = int(np.argmax(col))        # most positive = biggest writer
        agg_at = float(np.abs(col).sum())
        agg_ot = float(np.abs(drops_ot[:, k]).sum())
        specific = agg_at > agg_ot
        out['per_class'][c] = {
            'top3_by_abs': top3, 'top_writer_block': top_writer,
            'top_writer_val': round(float(col[top_writer]), 4),
            'block17': round(float(col[17]), 4),
            'drops': [round(float(v), 4) for v in col],
            'specific': bool(specific)}
        print(f'{c:12s} top writer block {top_writer:2d} '
              f'({col[top_writer]:+.4f}); block17 {col[17]:+.4f}; '
              f'specific {specific}', flush=True)

    # summary predictions
    func_early = all(out['per_class'][c]['top_writer_block'] <= 3
                     for c in ['determiner', 'punct']
                     if out['per_class'][c]['specific'])
    b17_supp = all(out['per_class'][c]['block17'] < 0
                   for c in ['determiner', 'punct', 'newline']
                   if out['per_class'][c]['specific'])
    mid_writers = {c: out['per_class'][c]['top_writer_block'] for c in CLASSES
                   if 10 <= out['per_class'][c]['top_writer_block'] <= 16
                   and out['per_class'][c]['specific']}
    print(f'\n(a) function classes early: {func_early}', flush=True)
    print(f'(b) block17 suppresses function classes: {b17_supp}', flush=True)
    print(f'(c) classes with a MIDDLE (10-16) top writer: {mid_writers}', flush=True)
    out.update({'pred_a_function_early': bool(func_early),
                'pred_b_block17_suppresses': bool(b17_supp),
                'middle_writer_classes': mid_writers,
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
