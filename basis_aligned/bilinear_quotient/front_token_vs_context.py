"""FRONT TOKEN VS CONTEXT -- how does the front (blocks 0-2, where the
next-token CLASS is decided, 629-631) decide it: from the CURRENT token
or from CONTEXT? Pivot toward the input, the program's core goal.

Two paths carry information into the residual at each block: ATTENTION
(a.c_proj output = context, mixing other positions) and the MLP (a
per-position transform of the current residual = token-local
computation). Mean-ablating the attention output at the front blocks
removes context (each position keeps only the mean attention
contribution); mean-ablating the MLP output removes the front's
per-position transform. Per token class, which ablation hurts the
class prediction more says whether that class is decided from context
or from the current token.

REGISTERED PREDICTIONS:
  (0) SANITY: both ablations change per-class prediction measurably;
  (a) CONTEXT CLASSES: determiner (the article circuit, 614, is context/
      bigram-driven), preposition, and newline drop MORE under front-
      ATTENTION ablation than the same classes drop under a matched
      LATE-block [10-12] attention ablation -- context for these enters
      at the front;
  (b) report, per class, the P(class) drop under front-attention vs
      front-mlp ablation, and the attention-vs-mlp ratio -- which
      classes are context-driven (attn) vs token-driven (mlp);
  (c) which classes survive front-attention ablation (token-driven);
  NULL: late-block [10-12] attention ablation has a SMALLER aggregate
      per-class effect than front-attention ablation -- context that
      sets the class enters at the front, not the late-middle."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_token_vs_context_results.json'
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


def meanfill_hook(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


@torch.no_grad()
def run(fresh, blocks, kind, vmask, tmask):
    """kind in {None,'attn','mlp'}; ablate that submodule in `blocks`."""
    handles = []
    if kind is not None:
        for li in blocks:
            sub = (m.transformer.h[li].attn.c_proj if kind == 'attn'
                   else m.transformer.h[li].mlp)
            handles.append(sub.register_forward_hook(meanfill_hook))
    C = vmask.shape[1]
    sum_at = np.zeros(C); n_at = np.zeros(C)
    pos0 = 0
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        pc = (p @ vmask).cpu().numpy()
        npos = pc.shape[0]
        tm = tmask[pos0:pos0 + npos]
        sum_at += (pc * tm).sum(0); n_at += tm.sum(0)
        pos0 += npos
    for h in handles:
        h.remove()
    return sum_at / np.maximum(n_at, 1)


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
    tmask = np.zeros((len(nxt), C), dtype=bool)
    for j, t in enumerate(nxt):
        c = classify(cl.d1(int(t)))
        if c in cidx:
            tmask[j, cidx[c]] = True

    base = run(fresh, None, None, vmask, tmask)
    fa = run(fresh, [0, 1, 2], 'attn', vmask, tmask)
    fm = run(fresh, [0, 1, 2], 'mlp', vmask, tmask)
    la = run(fresh, [10, 11, 12], 'attn', vmask, tmask)

    out = {'baseline_P': {}, 'per_class': {}}
    agg_front_attn = 0.0; agg_late_attn = 0.0
    for c in CLASSES:
        k = cidx[c]
        b = base[k]
        d_fa = float(1 - fa[k] / (b + 1e-9))
        d_fm = float(1 - fm[k] / (b + 1e-9))
        d_la = float(1 - la[k] / (b + 1e-9))
        ratio = d_fa / (d_fm + 1e-9)
        driver = 'context(attn)' if d_fa > d_fm else 'token(mlp)'
        out['baseline_P'][c] = round(float(b), 4)
        out['per_class'][c] = {'front_attn_drop': round(d_fa, 4),
                               'front_mlp_drop': round(d_fm, 4),
                               'late_attn_drop': round(d_la, 4),
                               'attn_over_mlp': round(ratio, 3), 'driver': driver}
        agg_front_attn += abs(d_fa); agg_late_attn += abs(d_la)
        print(f'{c:12s} front-attn {d_fa:+.3f}  front-mlp {d_fm:+.3f}  '
              f'late-attn {d_la:+.3f}  -> {driver}', flush=True)

    context_classes = ['determiner', 'preposition', 'newline']
    pa = all(out['per_class'][c]['front_attn_drop'] >
             out['per_class'][c]['late_attn_drop'] for c in context_classes)
    null_ok = agg_front_attn > agg_late_attn
    print(f'\n(a) context classes drop more under front-attn than late-attn: {pa}',
          flush=True)
    print(f'NULL front-attn aggregate {agg_front_attn:.3f} > late-attn '
          f'{agg_late_attn:.3f}: {"ok" if null_ok else "CHECK"}', flush=True)
    ctx_driven = [c for c in CLASSES if out['per_class'][c]['driver'].startswith('context')]
    print(f'(c) context-driven classes: {ctx_driven}', flush=True)

    out.update({'pred_a_context_front': bool(pa),
                'agg_front_attn': round(agg_front_attn, 4),
                'agg_late_attn': round(agg_late_attn, 4), 'null_ok': bool(null_ok),
                'context_driven': ctx_driven, 'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
