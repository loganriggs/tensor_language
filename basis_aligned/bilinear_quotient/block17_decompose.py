"""BLOCK17 DECOMPOSE -- finish the readout layer's characterization. We
know block 17's output splits into a rank-1 frequency CALIBRATION
direction w_freq (650-651, 656) and it also WRITES content classes (629,
subword/capitalized). Are these two SEPARABLE functions? Removing w_freq
should kill the calibration (626 signature) while leaving content-class
prediction intact.

Variants of mlp17's output:
  full           -- baseline
  remove_wfreq   -- rank-1 calibration direction removed
  mean_ablate    -- whole mlp17 output removed
Measure: calibration signature (freq-target vs rare-target CE) AND
content-class prediction P(subword), P(capitalized) at their targets.

REGISTERED PREDICTIONS:
  (0) SANITY: full vs mean_ablate shows both effects (mean-ablate helps
      freq CE, hurts rare CE, and lowers content-class P);
  (a) REMOVING w_freq KILLS CALIBRATION: remove_wfreq matches mean_ablate
      on the frequency signature (freq CE drops, calibration gone);
  (b) BUT PRESERVES CONTENT-WRITING: remove_wfreq keeps P(subword) and
      P(capitalized) near FULL (not collapsed like mean_ablate) -- the
      content-writing survives removing the calibration direction, so the
      two functions are separable / orthogonal components;
  (c) report freq/rare CE and content-class P for the three variants;
  NULL: removing a random rank-1 direction preserves BOTH calibration and
      content (nothing specific happens)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'block17_decompose_results.json'
NFRESH = 48
TOPK = 20

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'them', 'us',
        'me', 'who', 'which', 'what'}
W = {'dir': None, 'mode': None}


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
    d = W['dir']; proj = (o_ @ d)[..., None] * d
    return o_ - proj                              # remove


@torch.no_grad()
def run(fresh, sub_mask, cap_mask):
    V = m.lm_head.weight.shape[0]
    ce = torch.zeros(NFRESH, T); ps = torch.zeros(NFRESH, T); pc = torch.zeros(NFRESH, T)
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].reshape(-1); B = bb.shape[0]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        lg = (30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)).float()
        ce[i:i + B] = F.cross_entropy(lg.view(-1, V), tg, reduction='none').view(B, T).cpu()
        p = F.softmax(lg, dim=-1)
        ps[i:i + B] = (p @ sub_mask).cpu().view(B, T)
        pc[i:i + B] = (p @ cap_mask).cpu().view(B, T)
    return ce.reshape(-1).numpy(), ps.reshape(-1).numpy(), pc.reshape(-1).numpy()


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]
    nxt = fresh[:, 1:257].reshape(-1).numpy()
    freq = np.bincount(nxt, minlength=V)
    top = set(np.argsort(-freq)[:TOPK].tolist())
    is_freq = np.array([t in top for t in nxt])
    cls = np.array([classify(cl.d1(int(t))) for t in nxt])
    sub_t = cls == 'subword'; cap_t = cls == 'capitalized'
    sub_mask = torch.tensor([1.0 if classify(cl.d1(t)) == 'subword' else 0.0
                             for t in range(V)]).to(DEV)
    cap_mask = torch.tensor([1.0 if classify(cl.d1(t)) == 'capitalized' else 0.0
                             for t in range(V)]).to(DEV)

    # w_freq from mlp17 output
    cap = []
    hk = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D).cpu()))
    W['mode'] = None
    run(fresh, sub_mask, cap_mask)
    hk.remove()
    O = torch.cat(cap, 0).numpy()
    tgt_lf = np.log(freq[nxt] + 1.0); Oc = O - O.mean(0); tc = tgt_lf - tgt_lf.mean()
    w = Oc.T @ tc; w = w / (np.linalg.norm(w) + 1e-9)
    w_freq = torch.tensor(w, dtype=torch.float32, device=DEV)
    g = np.random.default_rng(0); rr = g.standard_normal(D); rr /= np.linalg.norm(rr)
    w_rand = torch.tensor(rr, dtype=torch.float32, device=DEV)

    hk = m.transformer.h[17].mlp.register_forward_hook(hook)
    res = {}
    for name, mode, d in [('full', None, None), ('remove_wfreq', 'rm', w_freq),
                          ('mean_ablate', 'mean', None), ('remove_random', 'rm', w_rand)]:
        W['mode'] = None if mode is None else ('mean' if mode == 'mean' else 'rm')
        W['dir'] = d
        ce, ps, pc = run(fresh, sub_mask, cap_mask)
        res[name] = {'CE_freq': round(float(ce[is_freq].mean()), 4),
                     'CE_rare': round(float(ce[~is_freq].mean()), 4),
                     'P_subword': round(float(ps[sub_t].mean()), 4),
                     'P_capitalized': round(float(pc[cap_t].mean()), 4)}
        print(f'{name:14s} CEfreq {res[name]["CE_freq"]:.3f} CErare '
              f'{res[name]["CE_rare"]:.3f}  P(sub) {res[name]["P_subword"]:.3f} '
              f'P(cap) {res[name]["P_capitalized"]:.3f}', flush=True)
    hk.remove()

    full = res['full']; mean = res['mean_ablate']; rm = res['remove_wfreq']
    # calibration killed? rm's rare CE near mean (calibration gone)
    cal_gap = (mean['CE_rare'] - full['CE_rare'])
    cal_killed = (rm['CE_rare'] - full['CE_rare']) / (cal_gap + 1e-9)
    # content preserved? rm's P(sub/cap) near full (not collapsed to mean)
    sub_gap = full['P_subword'] - mean['P_subword']
    cap_gap = full['P_capitalized'] - mean['P_capitalized']
    sub_kept = 1 - (full['P_subword'] - rm['P_subword']) / (sub_gap + 1e-9)
    cap_kept = 1 - (full['P_capitalized'] - rm['P_capitalized']) / (cap_gap + 1e-9)

    p0 = (mean['CE_freq'] < full['CE_freq']) and (mean['P_subword'] < full['P_subword'])
    pa = cal_killed >= 0.6
    pb = sub_kept >= 0.6 and cap_kept >= 0.6
    null_ok = abs(res['remove_random']['CE_rare'] - full['CE_rare']) < 0.3 * cal_gap
    print(f'\n(0) both effects present: {p0}', flush=True)
    print(f'(a) remove_wfreq kills calibration: {pa} ({100*cal_killed:.0f}%)', flush=True)
    print(f'(b) content-writing preserved: {pb} '
          f'(sub {100*sub_kept:.0f}%, cap {100*cap_kept:.0f}%)', flush=True)
    print(f'NULL remove_random preserves calibration: {null_ok}', flush=True)

    out = {'variants': res, 'cal_killed_frac': round(float(cal_killed), 3),
           'subword_kept_frac': round(float(sub_kept), 3),
           'capitalized_kept_frac': round(float(cap_kept), 3),
           'pred_0': bool(p0), 'pred_a_cal_killed': bool(pa),
           'pred_b_content_preserved': bool(pb), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
