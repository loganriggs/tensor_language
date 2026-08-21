"""NEWLINE DISCRIMINATION -- demonstrate 638's claim directly: the
embedding bigram is context-BLIND, and the blocks add context to
DISCRIMINATE which sentence-ending '.' are actually followed by a line
break.

638 argued the '.'-> newline bigram fires the same at every '.', and the
18 blocks discriminate the true line-ends from mid-paragraph periods.
Test: among end-punct positions (current token . ! ?), split by whether
a newline ACTUALLY follows, and compare P(newline) from the direct path
(embedding->unembedding, context-blind bigram) vs the full model in each
subset. If the blocks discriminate, the full model separates the two
subsets far more than the bigram does.

REGISTERED PREDICTIONS:
  (0) SANITY: enough end-punct positions in each subset (>=20);
  (a) FULL MODEL DISCRIMINATES: at end-punct positions actually followed
      by a newline, full-model P(newline) is much higher than at end-
      punct positions NOT followed by a newline;
  (b) BIGRAM IS BLIND: the direct-path P(newline) is similar in the two
      subsets (the context-blind bigram cannot tell them apart), so the
      full model's separation is much larger than the direct path's;
  (c) report P(newline) for {direct,full} x {newline-follows, not};
  NULL: the direct-path separation (follows vs not) is a small fraction
      of the full model's separation -- discrimination is added by the
      blocks, not present in the bigram."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_discrimination_results.json'
NFRESH = 48
NL1, NL2 = 198, 628


@torch.no_grad()
def pnl(fresh, direct):
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
        p = F.softmax(lg, dim=-1)
        out[i:i + B] = (p[..., NL1] + p[..., NL2]).cpu()
    return out.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    cur = fresh[:, :256].reshape(-1).numpy()
    nxt = fresh[:, 1:257].reshape(-1).numpy()

    def is_end_punct(t):
        s = cl.d1(int(t)).strip()
        return len(s) > 0 and s[-1] in '.!?'
    endp = np.array([is_end_punct(t) for t in cur])
    nl_follows = np.array([chr(10) in cl.d1(int(t)) for t in nxt])
    A = endp & nl_follows          # end-punct, newline follows
    Bm = endp & ~nl_follows        # end-punct, no newline
    print(f'end-punct: {A.sum()} followed by newline, {Bm.sum()} not', flush=True)

    pd = pnl(fresh, True)
    pf = pnl(fresh, False)
    dA, dB = float(pd[A].mean()), float(pd[Bm].mean())
    fA, fB = float(pf[A].mean()), float(pf[Bm].mean())
    sep_direct = dA - dB
    sep_full = fA - fB
    print(f'direct: newline-follows {dA:.4f}  not {dB:.4f}  (sep {sep_direct:+.4f})',
          flush=True)
    print(f'full:   newline-follows {fA:.4f}  not {fB:.4f}  (sep {sep_full:+.4f})',
          flush=True)

    p0 = A.sum() >= 20 and Bm.sum() >= 20
    pa = fA > 1.5 * fB
    pb = sep_full > 2 * sep_direct
    null_ok = sep_direct < 0.4 * sep_full
    print(f'\n(0) enough positions: {p0}', flush=True)
    print(f'(a) full model discriminates (fA>1.5x fB): {pa}', flush=True)
    print(f'(b) full separation >> direct: {pb} '
          f'(full {sep_full:.4f} vs direct {sep_direct:.4f})', flush=True)
    print(f'NULL direct sep < 0.4x full sep: {null_ok}', flush=True)

    out = {'n_follows': int(A.sum()), 'n_not': int(Bm.sum()),
           'direct_follows': round(dA, 5), 'direct_not': round(dB, 5),
           'full_follows': round(fA, 5), 'full_not': round(fB, 5),
           'sep_direct': round(sep_direct, 5), 'sep_full': round(sep_full, 5),
           'pred_0': bool(p0), 'pred_a_full_discriminates': bool(pa),
           'pred_b_full_gt_direct': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
