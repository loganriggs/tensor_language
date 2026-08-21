"""MLP17 SUBWORD ISOLATION -- finish block-17's decomposition (657). We
found block 17 = [frequency bias w_freq: calibration = rare-content boost,
rank-1] + [subword-writing: separate, survives removing w_freq]. Is that
separate subword-writing itself an ADDITIVE rank-1 component (isolable),
or a CONDITIONAL computation (654 taxonomy: no low-rank carrier)?

Recipe: w_sub = cov(mlp17 output, subword-target indicator). Remove it
(rank-1, then top-8) from mlp17's output; measure P(subword) at subword-
target positions. Controls: random removal; mean-ablate reference.

REGISTERED PREDICTIONS:
  (0) SANITY: full P(subword) high (~0.82, 657); mean-ablate lowers it
      (~0.68);
  (a) REGIME (the finding): is subword-writing rank-1 additive (removing
      w_sub collapses P(subword) toward mean-ablate) or conditional
      (removal barely changes it)? Registered guess: subword-writing is
      a per-position CONDITIONAL magnitude ("finish this word"), so like
      article-magnitude (654) it will NOT be rank-1 isolable -- top-8
      removal < 25% lost;
  (b) SPECIFICITY: whatever w_sub does, random removal does less;
  (c) report P(subword) for full / remove top-r (1,8) / random / mean;
  NULL: random removal preserves P(subword)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_subword_isolation_results.json'
NFRESH = 48

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}
W = {'Q': None, 'mode': None}


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


def hook(mo, i_, o_):
    if W['mode'] is None:
        return o_
    if W['mode'] == 'mean':
        return o_.mean(dim=(0, 1), keepdim=True).expand_as(o_)
    Q = W['Q']
    return o_ - (o_ @ Q) @ Q.T


@torch.no_grad()
def run(fresh, sub_mask):
    part = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        p = F.softmax(lg, dim=-1)
        part[i:i + B] = (p @ sub_mask).cpu().view(B, T)
    return part.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    cls = np.array([classify(cl.d1(int(t))) for t in nxt])
    sub_t = cls == 'subword'
    sub_mask = torch.tensor([1.0 if classify(cl.d1(t)) == 'subword' else 0.0
                             for t in range(V)]).to(DEV)

    # capture mlp17 output for w_sub
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    W['mode'] = None
    run(fresh, sub_mask)
    hk.remove()
    O = torch.cat(cap, 0).numpy()
    ind = sub_t.astype(np.float64); Oc = O - O.mean(0); ic = ind - ind.mean()
    dirs = []; R = Oc.copy()
    for _ in range(8):
        w = R.T @ ic; w = w / (np.linalg.norm(w) + 1e-9)
        dirs.append(w); R = R - (R @ w)[:, None] * w[None, :]
    Dfull = np.stack(dirs, 1)

    hk = m.transformer.h[17].mlp.register_forward_hook(hook)

    def go(mode, Q):
        W['mode'] = mode; W['Q'] = Q
        return float(run(fresh, sub_mask)[sub_t].mean())

    def T_(a):
        q, _ = np.linalg.qr(a); return torch.tensor(q, dtype=torch.float32, device=DEV)

    full = go(None, None)
    mean = go('mean', None)
    r1 = go('rm', T_(Dfull[:, :1]))
    r8 = go('rm', T_(Dfull[:, :8]))
    g = np.random.default_rng(0); rr = g.standard_normal((D, 1)); rr /= np.linalg.norm(rr)
    rand = go('rm', T_(rr))
    hk.remove()

    gap = full - mean
    lost1 = (full - r1) / (gap + 1e-9); lost8 = (full - r8) / (gap + 1e-9)
    lost_rand = (full - rand) / (gap + 1e-9)
    print(f'full {full:.4f}  mean {mean:.4f}  remove1 {r1:.4f} ({100*lost1:.0f}%)  '
          f'remove8 {r8:.4f} ({100*lost8:.0f}%)  random {rand:.4f} '
          f'({100*lost_rand:.0f}%)', flush=True)

    p0 = full > 0.5 and mean < full
    regime = ('additive-bias (rank-1)' if lost1 >= 0.5
              else 'conditional (no low-rank carrier)' if lost8 < 0.25 else 'intermediate')
    pb = lost8 > lost_rand
    null_ok = abs(lost_rand) < 0.15
    print(f'\n(0) sane: {p0}', flush=True)
    print(f'(a) REGIME: {regime} (remove1 {100*lost1:.0f}%, remove8 {100*lost8:.0f}%)',
          flush=True)
    print(f'(b) w_sub > random: {pb}; NULL random preserves: {null_ok}', flush=True)

    out = {'full': round(full, 5), 'mean_ablate': round(mean, 5),
           'remove1': round(r1, 5), 'remove8': round(r8, 5), 'random': round(rand, 5),
           'lost1': round(float(lost1), 3), 'lost8': round(float(lost8), 3),
           'regime': regime, 'pred_0': bool(p0), 'pred_b_specific': bool(pb),
           'null_ok': bool(null_ok), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
