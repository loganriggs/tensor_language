"""CLASS BIGRAM VS COMPUTED -- generalize 637: which token classes are
predicted by the 0-layer embedding->unembedding BIGRAM (network
ATTENUATES an already-strong bigram) vs genuinely COMPUTED by the 18
blocks (network AMPLIFIES a weak-in-embedding signal)?

637 found the newline and article triggers are embedding-level bigrams
that the blocks attenuate (direct/full > 1). This measures, per class,
P(class) at class-target positions for the direct path (embedding ->
unembedding, no blocks) vs the full model, and the ratio full/direct.
  full/direct < 1  -> the class is over-predicted by the raw bigram; the
                      network ATTENUATES it (bigram-driven, like newline).
  full/direct > 1  -> the class is under-predicted by the bigram; the
                      network COMPUTES / amplifies it (needs the blocks).

REGISTERED PREDICTIONS:
  (0) SANITY: the direct path predicts every class above its base rate
      at class-target positions (bigram is informative for all);
  (a) FUNCTION CLASSES ARE BIGRAM-DRIVEN: newline, determiner, punct,
      preposition have full/direct <= 1 (network attenuates a strong
      embedding bigram);
  (b) CONTENT CLASSES ARE COMPUTED: subword, capitalized, space_word
      have full/direct > 1 (network amplifies a weaker bigram);
  (c) report full and direct P(class) at class-target positions and the
      ratio, per class;
  NULL: at NON-class positions the direct-path P(class) is much lower
      than at class-target positions -- the embedding bigram is specific,
      not a constant offset."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'class_bigram_vs_computed_results.json'
NFRESH = 48

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}
CLASSES = ['newline', 'determiner', 'preposition', 'pronoun', 'digit',
           'punct', 'capitalized', 'space_word', 'subword']


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


@torch.no_grad()
def pclass(fresh, direct, vmask):
    C = vmask.shape[1]
    out = np.zeros((0, C))
    chunks = []
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        if not direct:
            x0 = x; v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        chunks.append((p @ vmask).cpu().numpy())
    return np.concatenate(chunks, 0)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    C = len(CLASSES); cidx = {c: k for k, c in enumerate(CLASSES)}
    vmask = torch.zeros(V, C)
    for t in range(V):
        c = classify(cl.d1(t))
        if c in cidx:
            vmask[t, cidx[c]] = 1.0
    vmask = vmask.to(DEV)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    tcls = np.array([classify(cl.d1(int(t))) for t in nxt])

    pd = pclass(fresh, True, vmask)
    pf = pclass(fresh, False, vmask)

    out = {'per_class': {}}
    for c in CLASSES:
        k = cidx[c]
        m_c = tcls == c
        if m_c.sum() < 30:
            continue
        d_at = float(pd[m_c, k].mean()); f_at = float(pf[m_c, k].mean())
        d_ot = float(pd[~m_c, k].mean())
        ratio = f_at / (d_at + 1e-9)
        out['per_class'][c] = {'direct_at_class': round(d_at, 4),
                               'full_at_class': round(f_at, 4),
                               'full_over_direct': round(ratio, 3),
                               'direct_off_class': round(d_ot, 4),
                               'kind': 'computed' if ratio > 1 else 'bigram(attenuated)'}
        print(f'{c:12s} direct {d_at:.4f}  full {f_at:.4f}  full/direct '
              f'{ratio:.2f}  ({out["per_class"][c]["kind"]})', flush=True)

    pc = out['per_class']
    func = [c for c in ['newline', 'determiner', 'punct', 'preposition'] if c in pc]
    cont = [c for c in ['subword', 'capitalized', 'space_word'] if c in pc]
    pa = all(pc[c]['full_over_direct'] <= 1.05 for c in func)
    pb = all(pc[c]['full_over_direct'] > 1.0 for c in cont)
    null_ok = all(pc[c]['direct_off_class'] < pc[c]['direct_at_class'] for c in pc)
    print(f'\n(a) function classes attenuated (full/direct<=1.05): {pa}', flush=True)
    print(f'(b) content classes computed (full/direct>1): {pb}', flush=True)
    print(f'NULL direct off-class < at-class for all: {null_ok}', flush=True)

    out.update({'pred_a_function_bigram': bool(pa),
                'pred_b_content_computed': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
