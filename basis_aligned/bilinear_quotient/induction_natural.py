"""INDUCTION NATURAL -- a fresh circuit thread: does the model do
INDUCTION/copying on natural text (predict the token that followed a
token's earlier occurrence), and where is it computed -- a contrast to
the sentence-boundary routing which lives in FRONT attention (644)?

For each position i with token A that also occurred earlier at j<i, the
induction target is B = the token that followed A at j (x[j+1]). If the
model does induction, P(B) at position i is elevated far above B's base
rate. Localize by ablating front [0-2] vs mid [6-9] vs late [12-15]
attention -- induction heads are classically mid/late, so (unlike
routing) this should NOT be front-attention-dominated.

REGISTERED PREDICTIONS:
  (0) SANITY: enough repeat positions (>=100);
  (a) INDUCTION EXISTS: mean P(B) at repeat positions is >= 3x B's base
      rate (the model copies the earlier continuation);
  (b) NOT FRONT-LOCALIZED: ablating mid or late attention reduces the
      induction signal more than ablating front attention -- a contrast
      to the routing circuit (644);
  (c) report induction P(B) and the per-band attention-ablation drops;
  NULL: a control target C (a random other vocab token) is NOT elevated
      at the same positions -- the elevation is specific to the
      induction continuation B."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'induction_natural_results.json'
NFRESH = 48


def meanfill_hook(mo, i_, o_):
    return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)


@torch.no_grad()
def run(fresh, blocks, kind):
    """Return per-(row,pos) full softmax rows are too big; instead return
    the logits-derived P for the induction target, computed per row using
    a precomputed target-token map. Here we return P over vocab is avoided;
    we return the softmax tensor row-gathered outside. To keep memory low
    we return P(B) and P(C) via target arrays passed as globals."""
    handles = []
    if kind is not None:
        for li in blocks:
            handles.append(m.transformer.h[li].attn.c_proj.register_forward_hook(
                meanfill_hook))
    PB = np.full(NFRESH * T, np.nan); PC = np.full(NFRESH * T, np.nan)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)                       # (B,T,V)
        for r in range(B):
            gi = (i + r) * T
            for t in range(T):
                bt = TGT_B[i + r, t]
                if bt >= 0:
                    PB[gi + t] = float(p[r, t, bt])
                    PC[gi + t] = float(p[r, t, TGT_C[i + r, t]])
    for h in handles:
        h.remove()
    return PB, PC


@torch.no_grad()
def main():
    global TGT_B, TGT_C
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    toks = fresh[:, :257].numpy()

    # induction target B[i]=token after the most recent earlier occurrence
    # of the current token; C = a control token (fixed offset in vocab).
    TGT_B = np.full((NFRESH, T), -1, dtype=np.int64)
    TGT_C = np.zeros((NFRESH, T), dtype=np.int64)
    freq = np.bincount(fresh[:, 1:257].reshape(-1).numpy(), minlength=V)
    for r in range(NFRESH):
        last = {}
        for t in range(T):
            a = int(toks[r, t])
            if a in last:
                j = last[a]
                TGT_B[r, t] = int(toks[r, j + 1])       # what followed A before
                TGT_C[r, t] = (TGT_B[r, t] + 101) % V   # control token
            last[a] = t
    valid = TGT_B.reshape(-1) >= 0
    print(f'{valid.sum()} repeat positions', flush=True)
    baseB = float(freq[TGT_B.reshape(-1)[valid]].mean() / freq.sum())

    conds = {'baseline': (None, None), 'front_attn': ([0, 1, 2], 'attn'),
             'mid_attn': ([6, 7, 8, 9], 'attn'), 'late_attn': ([12, 13, 14, 15], 'attn')}
    out = {'n_repeat': int(valid.sum()), 'base_rate_B': round(baseB, 6), 'conds': {}}
    for name, (b, k) in conds.items():
        PB, PC = run(fresh, b, k)
        pb = float(np.nanmean(PB)); pc = float(np.nanmean(PC))
        out['conds'][name] = {'P_B': round(pb, 5), 'P_C_control': round(pc, 5)}
        print(f'{name:12s} P(B) {pb:.5f}  P(control) {pc:.5f}', flush=True)

    base = out['conds']['baseline']
    p0 = valid.sum() >= 100
    pa = base['P_B'] >= 3 * baseB
    fa = base['P_B'] - out['conds']['front_attn']['P_B']
    md = base['P_B'] - out['conds']['mid_attn']['P_B']
    ld = base['P_B'] - out['conds']['late_attn']['P_B']
    pb = max(md, ld) > fa
    null_ok = base['P_B'] > 3 * base['P_C_control']
    print(f'\n(0) enough repeats: {p0}', flush=True)
    print(f'(a) induction exists (P(B) {base["P_B"]:.4f} >= 3x base {baseB:.5f}): {pa}',
          flush=True)
    print(f'(b) not front-localized (mid/late drop > front): {pb} '
          f'(front {fa:+.4f}, mid {md:+.4f}, late {ld:+.4f})', flush=True)
    print(f'NULL P(B) >> P(control): {null_ok}', flush=True)

    out.update({'drops': {'front': round(fa, 5), 'mid': round(md, 5), 'late': round(ld, 5)},
                'pred_0': bool(p0), 'pred_a_induction': bool(pa),
                'pred_b_not_front': bool(pb), 'null_ok': bool(null_ok),
                'runtime_s': time.time() - t0})
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
