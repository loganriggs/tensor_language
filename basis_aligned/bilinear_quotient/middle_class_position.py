"""WHAT DOES THE REDUNDANT MIDDLE COMPUTE? (§813 follow-up). §813 showed layers 6-11 do
1.93 nats of distributed/redundant work (per-component sum only 0.49). Per-component their
class+position keep was the lowest band (~0.63), but per-component keep is confounded by
redundancy (others compensate). Clean test: project ALL middle components (6-11) onto ONLY
their class+position subspace SIMULTANEOUSLY (mean-preserving) and measure how much of the
middle's collective 1.93-nat benefit that recovers. This asks whether the redundant middle
is maintaining/re-writing class+position (high recovery) or doing genuinely different
distributed content computation (low recovery).

REGISTERED PREDICTIONS:
  (0) SANITY: ablating all middle components reproduces ~1.93 nat benefit; full CE ~3.32;
  (a) IF middle collective keep-class+position is HIGH (>=0.6) and >> random, the redundant
      middle is largely class+position maintenance (the two variables are re-written /
      refreshed through the middle, redundantly);
  (b) IF LOW (<0.4, near random), the middle's distributed work is NOT class+position — it
      is the genuine content/context computation, and per-component keep under-reported it
      because redundancy masks single-component ablations;
  NULL: simultaneous random same-rank projection of the middle recovers far less."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'middle_class_position_results.json'
NEVAL = 160; MINCOUNT = 5; RTOK = 64; RPOS = 32
MID = list(range(6, 12))
SUBS = {}; MEANS = {}; MODE = {'op': None, 'rand': None}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    key = (w, L)
    def hook(mo, i_, o_):
        if MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        elif MODE['op'] == 'keeprand': U = MODE['rand']; v2 = (v @ U) @ U.T
        else: U = SUBS[key]; mu = MEANS[key]; v2 = mu + ((v - mu) @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture(rows, n, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r, gmean):
    rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - gmean[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    MODE['op'] = None
    for L in MID:
        for w in ('attn', 'mlp'):
            O, toks, pos = capture(rows, NEVAL, w, L)
            MEANS[(w, L)] = O.mean(0, keepdim=True)
            Utok = mean_subspace(O, toks, RTOK, MEANS[(w, L)]); Upos = mean_subspace(O, pos, RPOS, MEANS[(w, L)])
            SUBS[(w, L)] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for L in MID for w in ('attn', 'mlp')]
    MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
    MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL); ben = ce_abl - ce_full
    MODE['op'] = 'keep'; ce_keep = ce_on(rows, NEVAL)
    MODE['op'] = 'keeprand'; ce_rand = ce_on(rows, NEVAL); MODE['op'] = None
    for h in hooks: h.remove()
    rec = float((ce_abl - ce_keep)/max(ben, 1e-6)); recr = float((ce_abl - ce_rand)/max(ben, 1e-6))
    verdict = ('class+position maintenance' if rec >= 0.6 and rec > 2*recr else
               'genuine non-class+position content' if rec < 0.4 else 'mixed')
    out = {'band': 'middle_6_11', 'ce_full': round(ce_full, 4), 'collective_benefit': round(ben, 4),
           'keep_classpos': round(rec, 4), 'keep_random': round(recr, 4), 'verdict': verdict,
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'middle collective benefit {ben:.3f} | keep class+position {rec:.3f} | random {recr:.3f}', flush=True)
    print(f'VERDICT: redundant middle = {verdict}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
