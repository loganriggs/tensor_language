"""IS THE DISTRIBUTED REMAINDER FOR THE HARD PREDICTIONS? (new question that gives the
~22% a meaning). 795 showed class+position recovery is uniform across token-CLASS;
797 showed it is diffuse across DEPTH. Here: is it a function of DIFFICULTY? With all
36 components projected onto class+position, bin tokens by the full model's per-token
loss and measure the class+position CE-recovery per bin. If recovery is HIGH on easy
(low-loss) tokens and LOW on hard (high-loss, surprising) tokens, the distributed
computation is specifically for the HARD predictions -- the model is a class+position
machine for the predictable and a distributed machine for the surprising.

Runs on 128 rows.

REGISTERED PREDICTIONS:
  (0) SANITY: overall recovery ~ 0.78 (reproduce 794);
  (a) REMAINDER IS FOR HARD TOKENS: class+position CE-recovery DECREASES with token
      difficulty -- high on the easiest loss-decile (>= 0.85), lower on the hardest
      (gap >= 0.2). So the ~22% distributed computation is concentrated on the hard/
      surprising predictions;
  (b) report recovery per full-loss decile;
  ALT: if recovery is FLAT across difficulty (like it was across class 795), the
      remainder is a uniform overhead regardless of prediction difficulty too."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; NL = 18
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'remainder_by_difficulty_results.json'
NEVAL = 128; MINCOUNT = 5; RTOK = 64; RPOS = 32
SUBS = {}; MODE = {'op': None}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def hook_factory(w, L):
    key = (w, L)
    def h(mo, i_, o_):
        if MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: U = SUBS[key]; v2 = (v @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def per_token_nll(rows, n):
    nlls = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        nlls.append(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1), reduction='none').cpu())
    return torch.cat(nlls).numpy()


@torch.no_grad()
def capture_out(rows, n, w, L):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    MODE['op'] = None
    for L in range(NL):
        for w in ('attn', 'mlp'):
            O, toks, pos = capture_out(rows, NEVAL, w, L)
            Utok = mean_subspace(O, toks, RTOK); Upos = mean_subspace(O, pos, RPOS)
            SUBS[(w, L)] = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    hooks = [comp(w, L).register_forward_hook(hook_factory(w, L)) for L in range(NL) for w in ('attn', 'mlp')]
    MODE['op'] = None; nf = per_token_nll(rows, NEVAL)
    MODE['op'] = 'ablate'; na = per_token_nll(rows, NEVAL)
    MODE['op'] = 'keep'; nk = per_token_nll(rows, NEVAL); MODE['op'] = None
    for h in hooks: h.remove()

    overall = float((na.mean()-nk.mean())/max(na.mean()-nf.mean(), 1e-9))
    # bin by full-model per-token loss (deciles)
    order = np.argsort(nf); nb = 10; bins = np.array_split(order, nb)
    by_decile = []
    for b in range(nb):
        ix = bins[b]; num = na[ix].mean()-nk[ix].mean(); den = na[ix].mean()-nf[ix].mean()
        by_decile.append({'mean_full_loss': round(float(nf[ix].mean()), 3), 'recovery': round(float(num/max(den, 1e-9)), 4)})
    print(f'overall recovery {overall:.3f}', flush=True)
    for b, r in enumerate(by_decile):
        print(f'  loss-decile {b} (mean loss {r["mean_full_loss"]:.2f}): class+pos recovery {r["recovery"]:.3f}', flush=True)
    easy = by_decile[0]['recovery']; hard = by_decile[-1]['recovery']
    print(f'easiest-decile recovery {easy:.3f}  vs  hardest-decile {hard:.3f}  gap {easy-hard:.3f}', flush=True)

    p0 = abs(overall-0.78) < 0.15
    pa = easy >= 0.85 and (easy-hard) >= 0.2
    out = {'overall_recovery': round(overall, 4), 'by_loss_decile': by_decile, 'easiest_recovery': round(easy, 4),
           'hardest_recovery': round(hard, 4), 'easy_minus_hard': round(easy-hard, 4),
           'pred_0': bool(p0), 'pred_a_remainder_is_hard_predictions': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) the ~22% remainder is for HARD predictions (easy high, hard low): {pa}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
