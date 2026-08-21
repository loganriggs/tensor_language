"""MIDDLE REFINES WHICH CLASS -- localize the middle's within-class
refinement (631) to specific token classes. The middle refines the
specific token within an already-decided class; for which classes does
it do the most work?

631 showed the middle [6-16] spares the class while degrading the
specific token. This measures the within-class sparing PER CLASS:
ablate the middle, and for each token class compute the relative drop
in P(correct token) vs P(correct class) at that class's target
positions. Large sparing (token lost >> class lost) = the middle does
real within-class work for that class; small sparing = little to refine
(the class has few members, or is token-determined up front).

REGISTERED PREDICTIONS:
  (0) SANITY: for every class, P(class) >= P(token) at baseline;
  (a) CONTENT CLASSES REFINED MOST: the large open classes (subword,
      space_word, capitalized) show the largest within-class sparing
      (token-drop minus class-drop) under middle ablation -- the middle
      picks which content word;
  (b) FUNCTION CLASSES REFINED LEAST: small closed classes (determiner,
      punct, pronoun) show little sparing -- few members, so class ~
      token, nothing to refine within;
  (c) report per-class sparing under middle vs front ablation;
  NULL: the ordering (content > function sparing) holds for the MIDDLE
      but is weaker/absent for the FRONT -- refinement is a middle
      specialization, not a generic ablation ordering."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_refines_which_class_results.json'
NFRESH = 48

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}
BANDS = {'front': [0, 1, 2], 'middle': list(range(6, 17))}
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
def measure(fresh, ablate_set, tok_of_pos, tgt_class, class_to_vmask):
    ptok = np.zeros(len(tok_of_pos)); pcls = np.zeros(len(tok_of_pos))
    pos0 = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x_in = x
            x, v1 = blk(x, v1, x0)
            if ablate_set is not None and li in ablate_set:
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
    class_pos = {c: (tgt_class == c) for c in CLASSES}

    base_t, base_c = measure(fresh, None, tok_of_pos, tgt_class, class_to_vmask)

    def per_class_sparing(ab_t, ab_c):
        res = {}
        for c in CLASSES:
            m_c = class_pos[c]
            if m_c.sum() < 30:
                continue
            rt = float(1 - ab_t[m_c].mean() / (base_t[m_c].mean() + 1e-9))
            rc = float(1 - ab_c[m_c].mean() / (base_c[m_c].mean() + 1e-9))
            res[c] = {'token_drop': round(rt, 3), 'class_drop': round(rc, 3),
                      'sparing': round(rt - rc, 3),
                      'base_Ptoken': round(float(base_t[m_c].mean()), 3),
                      'base_Pclass': round(float(base_c[m_c].mean()), 3)}
        return res

    out = {'bands': {}}
    for name, blocks in BANDS.items():
        ab_t, ab_c = measure(fresh, set(blocks), tok_of_pos, tgt_class,
                             class_to_vmask)
        pc = per_class_sparing(ab_t, ab_c)
        out['bands'][name] = pc
        print(f'\n{name} ablated -- per-class sparing (token-drop - class-drop):',
              flush=True)
        for c in sorted(pc, key=lambda k: -pc[k]['sparing']):
            print(f'  {c:12s} spare {pc[c]["sparing"]:+.3f} '
                  f'(tok {pc[c]["token_drop"]:.3f} cls {pc[c]["class_drop"]:.3f})',
                  flush=True)

    mid = out['bands']['middle']; fr = out['bands']['front']
    content = [c for c in ['subword', 'space_word', 'capitalized'] if c in mid]
    function = [c for c in ['determiner', 'punct', 'pronoun'] if c in mid]
    mid_content = np.mean([mid[c]['sparing'] for c in content]) if content else 0
    mid_func = np.mean([mid[c]['sparing'] for c in function]) if function else 0
    fr_content = np.mean([fr[c]['sparing'] for c in content]) if content else 0
    fr_func = np.mean([fr[c]['sparing'] for c in function]) if function else 0
    pa = mid_content > mid_func
    null_ok = (mid_content - mid_func) > (fr_content - fr_func)
    print(f'\n(a/b) middle content sparing {mid_content:.3f} > function '
          f'{mid_func:.3f}: {pa}', flush=True)
    print(f'NULL content>function gap larger for middle ({mid_content-mid_func:+.3f}) '
          f'than front ({fr_content-fr_func:+.3f}): {"ok" if null_ok else "CHECK"}',
          flush=True)

    out.update({'mid_content_sparing': round(float(mid_content), 3),
                'mid_function_sparing': round(float(mid_func), 3),
                'pred_a_content_refined': bool(pa), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
