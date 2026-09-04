"""MIDDLE REFINE ATTN VS MLP -- pivot to the least-characterized region
(focus D). The middle blocks (6-16) refine WITHIN-class token identity --
which specific content word to emit (631-632). Is that refinement
CONTEXT-driven (attention: which word fits the context) or TOKEN-LOCAL
(MLP)? Parallel to 634's front analysis (class identity was MLP-dominant).

Mean-ablate the middle [7-15] ATTENTION (c_proj = context) vs the middle
MLP (token-local transform), and measure P(correct next token) at
space_word targets (the open content-word slot the middle refines most,
632). Which ablation hurts the refinement more?

REGISTERED PREDICTIONS:
  (0) SANITY: baseline P(correct token) at space_word targets is
      substantial;
  (a) REFINEMENT IS CONTEXT-DRIVEN: ablating middle ATTENTION hurts
      P(correct content token) more than ablating the middle MLP --
      picking WHICH content word needs context, unlike class identity
      (634, MLP-dominant);
  (b) report P(correct token) under middle-attn vs middle-mlp ablation,
      at space_word targets and (contrast) at all targets;
  NULL: a LATE-block [16] attention ablation hurts the refinement less
      than the middle attention (the refinement is a middle function)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_refine_attn_vs_mlp_results.json'
NFRESH = 48

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


def meanfill(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


@torch.no_grad()
def ptok(fresh, blocks, kind, tok_of_pos):
    handles = []
    if kind is not None:
        for li in blocks:
            sub = (m.transformer.h[li].attn.c_proj if kind == 'attn'
                   else m.transformer.h[li].mlp)
            handles.append(sub.register_forward_hook(meanfill))
    out = np.zeros(NFRESH * T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        npos = p.shape[0]; base = i * T
        tk = tok_of_pos[base:base + npos]
        out[base:base + npos] = p[np.arange(npos), tk].cpu().numpy()
    for h in handles:
        h.remove()
    return out


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    tok_of_pos = nxt.astype(np.int64)
    cls = np.array([classify(cl.d1(int(t))) for t in nxt])
    sw = cls == 'space_word'
    allm = np.ones(len(nxt), bool)
    MID = list(range(7, 16))

    conds = {'baseline': (None, None), 'mid_attn': (MID, 'attn'),
             'mid_mlp': (MID, 'mlp'), 'late16_attn': ([16], 'attn')}
    P = {n: ptok(fresh, b, k, tok_of_pos) for n, (b, k) in conds.items()}

    def stat(p, mask):
        return float(p[mask].mean())
    out = {'space_word': {}, 'all': {}}
    for n in conds:
        out['space_word'][n] = round(stat(P[n], sw), 5)
        out['all'][n] = round(stat(P[n], allm), 5)
        print(f'{n:12s} P(tok) space_word {out["space_word"][n]:.4f}  '
              f'all {out["all"][n]:.4f}', flush=True)

    b = out['space_word']
    drop_attn = b['baseline'] - b['mid_attn']
    drop_mlp = b['baseline'] - b['mid_mlp']
    drop_late = b['baseline'] - b['late16_attn']
    p0 = b['baseline'] > 0.1
    pa = drop_attn > drop_mlp
    null_ok = drop_attn > drop_late
    print(f'\n(0) sane: {p0}', flush=True)
    print(f'(a) refinement context-driven (mid-attn drop {drop_attn:.4f} > '
          f'mid-mlp {drop_mlp:.4f}): {pa}', flush=True)
    print(f'NULL mid-attn > late16-attn ({drop_attn:.4f} > {drop_late:.4f}): {null_ok}',
          flush=True)

    out.update({'drop_mid_attn': round(drop_attn, 5), 'drop_mid_mlp': round(drop_mlp, 5),
                'drop_late16_attn': round(drop_late, 5),
                'pred_0': bool(p0), 'pred_a_context_driven': bool(pa),
                'null_ok': bool(null_ok), 'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
