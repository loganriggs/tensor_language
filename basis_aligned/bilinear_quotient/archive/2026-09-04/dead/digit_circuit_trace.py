"""DIGIT CIRCUIT TRACE -- a fresh thread on the MOST-computed class.
638 found digit prediction is barely in the embedding bigram (direct
0.069 at digit-target positions) yet the full model predicts it well
(0.571, full/direct 8.3x) -- the model COMPUTES digit prediction rather
than looking it up. What is the digit circuit, and what input feature
drives it?

Candidate trigger: the preceding token is itself a digit (number
continuation -- "3.14", "2023", "1,000"). This measures P(digit) grouped
by whether the current token is a digit, at direct (embedding bigram) vs
full model vs front-attention-ablated vs front-MLP-ablated, to trace
whether digit continuation is a bigram, and whether it is carried by
context (attention) or the token-local MLP.

REGISTERED PREDICTIONS:
  (0) SANITY: overall P(digit) is a few percent;
  (a) DIGIT-CONTINUATION TRIGGER: P(digit) is much higher when the
      current token is a digit than when it is not (number continuation);
  (b) COMPUTED, NOT BIGRAM: at prev-digit positions the FULL model
      predicts digit far better than the DIRECT embedding bigram (full
      >> direct) -- consistent with 638's 8.3x, the continuation is
      computed, not a memorized bigram;
  (c) report P(digit) for {direct, full, front-attn-abl, front-mlp-abl}
      x {prev-digit, prev-not};
  NULL: at prev-not-digit positions P(digit) stays low in every
      condition -- the digit signal is specific to numeric context."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'digit_circuit_trace_results.json'
NFRESH = 48


def is_digit_tok(t):
    s = cl.d1(int(t)).strip()
    return len(s) > 0 and s[0].isdigit()


def meanfill_hook(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


@torch.no_grad()
def pdigit(fresh, digit_mask, direct=False, blocks=None, kind=None):
    handles = []
    if kind is not None:
        for li in blocks:
            sub = (m.transformer.h[li].attn.c_proj if kind == 'attn'
                   else m.transformer.h[li].mlp)
            handles.append(sub.register_forward_hook(meanfill_hook))
    out = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        if not direct:
            x0 = x; v1 = None
            for blk in m.transformer.h:
                x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1).reshape(-1, lg.shape[-1])
        pv = (p @ digit_mask).cpu()
        out[i:i + B] = pv.view(B, T)
    for h in handles:
        h.remove()
    return out.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    digit_mask = torch.tensor([1.0 if is_digit_tok(t) else 0.0
                               for t in range(V)]).to(DEV)

    cur = fresh[:, :256].reshape(-1).numpy()
    prev_digit = np.array([is_digit_tok(t) for t in cur])
    print(f'{prev_digit.sum()} prev-digit positions, {(~prev_digit).sum()} not',
          flush=True)

    conds = {
        'direct': dict(direct=True),
        'full': dict(),
        'front_attn_abl': dict(blocks=[0, 1, 2], kind='attn'),
        'front_mlp_abl': dict(blocks=[0, 1, 2], kind='mlp')}
    P = {name: pdigit(fresh, digit_mask, **kw) for name, kw in conds.items()}

    out = {'by_cond': {}}
    for name, p in P.items():
        pd_ = float(p[prev_digit].mean()); pn = float(p[~prev_digit].mean())
        out['by_cond'][name] = {'prev_digit': round(pd_, 5), 'prev_not': round(pn, 5),
                                'overall': round(float(p.mean()), 5)}
        print(f'{name:15s} prev-digit {pd_:.4f}  prev-not {pn:.4f}  '
              f'overall {p.mean():.4f}', flush=True)

    b = out['by_cond']
    p0 = 0.002 < b['full']['overall'] < 0.2
    pa = b['full']['prev_digit'] > 3 * b['full']['prev_not']
    pb = b['full']['prev_digit'] > 2 * b['direct']['prev_digit']
    null_ok = all(b[c]['prev_not'] < 0.05 for c in conds)
    print(f'\n(0) overall plausible: {p0}', flush=True)
    print(f'(a) digit-continuation trigger (full prev-digit>3x prev-not): {pa}',
          flush=True)
    print(f'(b) computed not bigram (full>2x direct at prev-digit): {pb} '
          f'(full {b["full"]["prev_digit"]:.4f} vs direct '
          f'{b["direct"]["prev_digit"]:.4f})', flush=True)
    print(f'NULL prev-not stays low: {null_ok}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_continuation': bool(pa),
                'pred_b_computed': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
