"""WITHIN CLASS DEPTH PROFILE -- where in depth does the open-content-
word refinement (632) happen? Per-block within-class sparing across all
18 blocks, overall and for space_word (the class the middle refines most,
632).

632 found the middle refines the open content-word slot (space_word,
sparing +0.67 under the whole middle band). This locates it block by
block: single-block mean-ablation, measuring the within-class sparing
(relative P(token)-drop minus relative P(class)-drop) at each block, all
positions and space_word-target positions.

REGISTERED PREDICTIONS:
  (0) SANITY: block 17's overall sparing is small/negative -- it
      calibrates, it does not refine within-class (628/629);
  (a) MIDDLE REFINES: the per-block space_word sparing is larger in the
      middle band (6-16) than in the front (0-2) on average -- the
      content-word choice is refined in the middle;
  (b) report the per-block space_word sparing profile and the overall
      sparing profile;
  NULL: front blocks (0-2), which decide the CLASS, show low within-class
      sparing (they drop class and token together, 631)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'within_class_depth_profile_results.json'
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


@torch.no_grad()
def measure(fresh, ablate_block, tok_of_pos, tgt_class, class_to_vmask):
    ptok = np.zeros(len(tok_of_pos)); pcls = np.zeros(len(tok_of_pos))
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
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        npos = p.shape[0]
        tk = tok_of_pos[pos0:pos0 + npos]
        ptok[pos0:pos0 + npos] = p[np.arange(npos), tk].cpu().numpy()
        cm = torch.stack([class_to_vmask[c]
                          for c in tgt_class[pos0:pos0 + npos]]).to(DEV)
        pcls[pos0:pos0 + npos] = (p * cm).sum(-1).cpu().numpy()
        pos0 += npos
    return ptok, pcls


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    vocab_class = np.array([classify(cl.d1(t)) for t in range(V)])
    tok_of_pos = nxt.astype(np.int64)
    tgt_class = np.array([classify(cl.d1(int(t))) for t in nxt])
    class_to_vmask = {c: torch.tensor(vocab_class == c, dtype=torch.float32)
                      for c in set(tgt_class.tolist())}
    sw = (tgt_class == 'space_word')
    allm = np.ones(len(nxt), bool)

    base_t, base_c = measure(fresh, None, tok_of_pos, tgt_class, class_to_vmask)

    def sparing(ab_t, ab_c, mask):
        rt = 1 - ab_t[mask].mean() / (base_t[mask].mean() + 1e-9)
        rc = 1 - ab_c[mask].mean() / (base_c[mask].mean() + 1e-9)
        return float(rt - rc)

    prof_all = []; prof_sw = []
    for L in range(NB):
        ab_t, ab_c = measure(fresh, L, tok_of_pos, tgt_class, class_to_vmask)
        prof_all.append(round(sparing(ab_t, ab_c, allm), 4))
        prof_sw.append(round(sparing(ab_t, ab_c, sw), 4))
        print(f'  block {L:2d}: sparing all {prof_all[-1]:+.4f}  '
              f'space_word {prof_sw[-1]:+.4f}', flush=True)

    front_sw = float(np.mean([prof_sw[L] for L in range(3)]))
    mid_sw = float(np.mean([prof_sw[L] for L in range(6, 17)]))
    p0 = prof_all[17] < 0.05
    pa = mid_sw > front_sw
    top_block = int(np.argmax(prof_sw))
    print(f'\n(0) block17 overall sparing small: {p0} ({prof_all[17]})', flush=True)
    print(f'(a) middle space_word sparing {mid_sw:.4f} > front {front_sw:.4f}: {pa}',
          flush=True)
    print(f'(b) top space_word-refining block: {top_block} ({prof_sw[top_block]})',
          flush=True)

    out = {'sparing_all': prof_all, 'sparing_space_word': prof_sw,
           'front_sw_sparing': round(front_sw, 4), 'mid_sw_sparing': round(mid_sw, 4),
           'top_sw_block': top_block, 'pred_0': bool(p0),
           'pred_a_middle_refines': bool(pa), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
