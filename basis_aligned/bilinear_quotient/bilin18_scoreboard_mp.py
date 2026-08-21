"""BILIN18 per-component mean-preserving scoreboard — completes the corrected all-six
class+position table (§807). The five HF models were re-scored with mean-preserving keep
(cross_model_scoreboard_mp: 0.64-0.92); bilin18's own whole-model number (0.78) used
centered keep / the simultaneous metric and is a slight underestimate. This gives bilin18
the SAME per-component nat-weighted metric as the HF sweep: for EVERY component (attn0-17,
mlp0-17), single-component ablate / centered-keep / mean-preserving-keep / random-keep,
nat-weighted by benefit.

REGISTERED PREDICTIONS:
  (0) SANITY: ablate-all-components raises CE; random nat-weighted keep is low;
  (a) bilin18 nw mean-preserving >= nw centered (0.78-ish) and lands in the same band as
      the HF models (~0.8-0.9); the corrected all-six table has NO exception;
  (b) report centered vs mean-preserving vs random for bilin18."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'bilin18_scoreboard_mp_results.json'
NEVAL = 128; MINCOUNT = 5; RTOK = 64; RPOS = 32; LAYERS = list(range(18))
SUBS = {}; MEANS = {}; MODE = {'op': None, 'rand': None, 'single': None}


def comp(which, L):
    return m.transformer.h[L].mlp if which == 'mlp' else m.transformer.h[L].attn


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
def capture_out(rows, n, which, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(which, L).register_forward_hook(h)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def sh(which, L):
    key = (which, L)
    def hh(mo, i_, o_):
        if MODE['op'] is None or MODE['single'] != key: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh_ = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        elif MODE['op'] == 'keeprand': U = MODE['rand']; v2 = (v @ U) @ U.T
        elif MODE['op'] == 'keep': U = SUBS[key]; v2 = (v @ U) @ U.T
        else: U = SUBS[key]; mu = MEANS[key]; v2 = mu + ((v - mu) @ U) @ U.T
        yn = v2.reshape(sh_).to(y.dtype); return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hh


def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    comps = [(w, L) for L in LAYERS for w in ('attn', 'mlp')]
    MODE['op'] = None
    for w, L in comps:
        O, toks, pos = capture_out(rows, NEVAL, w, L)
        MEANS[(w, L)] = O.mean(0, keepdim=True)
        Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
        SUBS[(w, L)] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); MODE['rand'] = torch.linalg.qr(torch.randn(D, RTOK+RPOS, generator=g, device=DEV))[0]
    hooks = [comp(w, L).register_forward_hook(sh(w, L)) for w, L in comps]
    MODE['op'] = None; MODE['single'] = None; ce_full = ce_on(rows, NEVAL)
    per = {}; tb = 0.0; tk = 0.0; tkm = 0.0; tr = 0.0
    for w, L in comps:
        MODE['single'] = (w, L)
        MODE['op'] = 'ablate'; ca = ce_on(rows, NEVAL); ben = ca - ce_full
        MODE['op'] = 'keep'; ck = ce_on(rows, NEVAL)
        MODE['op'] = 'keepmean'; ckm = ce_on(rows, NEVAL)
        MODE['op'] = 'keeprand'; cr = ce_on(rows, NEVAL); MODE['op'] = None
        rec = float((ca-ck)/max(ben, 1e-6)); recm = float((ca-ckm)/max(ben, 1e-6)); recr = float((ca-cr)/max(ben, 1e-6))
        dc = float(MEANS[(w, L)].norm())
        per[f'{w}{L}'] = {'benefit': round(ben, 3), 'keep_cent': round(rec, 3), 'keep_mean': round(recm, 3), 'rand': round(recr, 3)}
        if ben > 0:
            tb += ben; tk += ben*max(min(rec, 1), 0); tkm += ben*max(min(recm, 1), 0); tr += ben*max(min(recr, 1), 0)
    MODE['single'] = None
    for h in hooks: h.remove()
    out = {'model': 'bilin18', 'd': D, 'n_components': len(comps), 'total_benefit': round(tb, 3),
           'nw_centered': round(tk/max(tb, 1e-9), 4), 'nw_meanpreserve': round(tkm/max(tb, 1e-9), 4),
           'nw_random': round(tr/max(tb, 1e-9), 4), 'per_component': per, 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"bilin18 ({out['n_components']} comp): NW class+pos CENTERED {out['nw_centered']} -> MEAN-PRESERVE {out['nw_meanpreserve']} | random {out['nw_random']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']:.0f}s)")


if __name__ == '__main__':
    main()
