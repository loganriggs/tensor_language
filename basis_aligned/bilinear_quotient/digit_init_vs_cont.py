"""DIGIT INIT VS CONT -- confirm 641's split of the digit class into a
BIGRAM (number continuation) and a COMPUTED (number initiation) circuit.

641 showed digit-continuation (prev token is a digit) is carried by the
embedding bigram (full ~ direct), while 638's "digit is 8.3x computed"
must come from number initiation (first digit, preceded by a word). This
tests it directly: at DIGIT-TARGET positions (the next token really is a
digit), split by whether the current token is a digit (CONTINUATION) or
not (INITIATION), and compare direct (embedding bigram) vs full P(digit).

REGISTERED PREDICTIONS:
  (0) SANITY: both subsets have >= 20 positions;
  (a) CONTINUATION = BIGRAM: at continuation digit-targets, full P(digit)
      is close to direct (full/direct < 2) -- the bigram carries it;
  (b) INITIATION = COMPUTED: at initiation digit-targets, full P(digit)
      is much larger than direct (full/direct > 3) -- the model computes
      it from context, this is 638's 8.3x;
  (c) report direct and full P(digit) and the ratio for each subset;
  NULL: direct P(digit) at initiation targets is low (the context-blind
      bigram cannot predict a first digit from a non-digit token)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'digit_init_vs_cont_results.json'
NFRESH = 48


def is_digit_tok(t):
    s = cl.d1(int(t)).strip()
    return len(s) > 0 and s[0].isdigit()


@torch.no_grad()
def pdigit(fresh, digit_mask, direct):
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
        out[i:i + B] = (p @ digit_mask).cpu().view(B, T)
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
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    tgt_digit = np.array([is_digit_tok(t) for t in nxt])
    prev_digit = np.array([is_digit_tok(t) for t in cur])
    cont = tgt_digit & prev_digit          # continuation: digit after digit
    init = tgt_digit & ~prev_digit         # initiation: digit after non-digit
    print(f'digit-targets: {cont.sum()} continuation, {init.sum()} initiation',
          flush=True)

    pd = pdigit(fresh, digit_mask, True)
    pf = pdigit(fresh, digit_mask, False)

    out = {}
    for name, msk in [('continuation', cont), ('initiation', init)]:
        d = float(pd[msk].mean()); f = float(pf[msk].mean())
        out[name] = {'direct': round(d, 5), 'full': round(f, 5),
                     'full_over_direct': round(f / (d + 1e-9), 3),
                     'n': int(msk.sum())}
        print(f'{name:12s} direct {d:.4f}  full {f:.4f}  full/direct '
              f'{f/(d+1e-9):.2f}  (n={int(msk.sum())})', flush=True)

    p0 = out['continuation']['n'] >= 20 and out['initiation']['n'] >= 20
    pa = out['continuation']['full_over_direct'] < 2.0
    pb = out['initiation']['full_over_direct'] > 3.0
    null_ok = out['initiation']['direct'] < 0.5 * out['continuation']['direct']
    print(f'\n(0) enough: {p0}', flush=True)
    print(f'(a) continuation is bigram (full/direct<2): {pa}', flush=True)
    print(f'(b) initiation is computed (full/direct>3): {pb}', flush=True)
    print(f'NULL direct low at initiation: {null_ok}', flush=True)

    out.update({'pred_0': bool(p0), 'pred_a_cont_bigram': bool(pa),
                'pred_b_init_computed': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
