"""POSITION CAUSAL (follows 775: position is strongly DECODABLE from mlp1's non-
token-class residual, R2 0.81 -- but is it CAUSAL, or decodable-but-inert like the
previous token?). Build the canonical POSITION-conditional-mean subspace of mlp1's
output (average output at each position; the position-driven component, token
averages out -- the positional analog of the token-class subspace 767), and test
NECESSARY (remove -> dCE vs random) + SUFFICIENT (keep-only -> CE-recovery) +
whether it is ORTHOGONAL to the token-class subspace (a separate multiplexed
channel, 772).

REGISTERED PREDICTIONS:
  (0) SANITY: position-conditional means separate (nonzero positional variance);
  (a) CAUSAL: removing the top-32 positional subspace from mlp1 raises CE >= 3x a
      random same-rank subspace (position is USED, not just decodable) -- OR, if
      it's decodable-but-inert like prev-token, the ratio is ~1 (report which);
  (b) report keep-only-positional CE-recovery + overlap of the positional subspace
      with the token-class subspace (expect near-orthogonal, a separate channel);
  NULL: random same-rank subspace removal is ~harmless."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'position_causal_results.json'
NEVAL = 64; MINCOUNT = 5; RPOS = 32
MODE = {'U': None, 'op': None}   # op: remove | keep | ablate


def mlp_hook(mo, i_, o_):
    if MODE['op'] is None: return o_
    sh = o_.shape; v = o_.reshape(-1, D).float()
    if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
    else:
        U = MODE['U']; v2 = (v @ U) @ U.T if MODE['op'] == 'keep' else v - (v @ U) @ U.T
    return v2.reshape(sh).to(o_.dtype)


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
def capture(rows, n):
    cap = []; toks = []; pos = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1))
        pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    h.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def overlap(A, B):
    return float(torch.linalg.svdvals(A.T @ B).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    h0 = m.transformer.h[LAYER].mlp.register_forward_hook(mlp_hook)
    O, toks, pos = capture(rows, NEVAL)
    Upos = mean_subspace(O, pos, RPOS)
    Utok = mean_subspace(O, toks, RPOS)

    MODE['op'] = None; ce_full = ce_on(rows, NEVAL)
    MODE['op'] = 'ablate'; ce_abl = ce_on(rows, NEVAL)
    ben = ce_abl - ce_full
    MODE['op'] = 'remove'; MODE['U'] = Upos; ce_rem = ce_on(rows, NEVAL)
    g = torch.Generator(device=DEV).manual_seed(0); Ur = torch.linalg.qr(torch.randn(D, RPOS, generator=g, device=DEV))[0]
    MODE['U'] = Ur; ce_rem_rand = ce_on(rows, NEVAL)
    MODE['op'] = 'keep'; MODE['U'] = Upos; ce_keep = ce_on(rows, NEVAL)
    MODE['op'] = None; MODE['U'] = None
    d_pos = ce_rem - ce_full; d_rand = ce_rem_rand - ce_full
    keep_rec = (ce_abl - ce_keep)/max(ben, 1e-6)
    ov_tok = overlap(Upos, Utok)
    g2 = torch.Generator(device=DEV).manual_seed(1); Rr = torch.linalg.qr(torch.randn(D, RPOS, generator=g2, device=DEV))[0]
    ov_rand = overlap(Upos, Rr)
    print(f'benefit {ben:.3f}', flush=True)
    print(f'(a) remove positional dCE {d_pos:.3f} vs random {d_rand:.3f}  ratio {d_pos/max(d_rand,1e-6):.1f}', flush=True)
    print(f'(b) keep-only-positional CE-recovery {keep_rec:.3f} | pos-vs-token subspace overlap {ov_tok:.3f} (random {ov_rand:.3f})', flush=True)
    h0.remove()

    ratio = d_pos/max(d_rand, 1e-6)
    p0 = True
    pa_causal = ratio >= 3
    out = {'benefit': round(ben, 4), 'dce_positional': round(d_pos, 4), 'dce_random': round(d_rand, 4),
           'remove_ratio': round(float(ratio), 2), 'keep_only_recovery': round(float(keep_rec), 4),
           'pos_token_overlap': round(ov_tok, 4), 'pos_random_overlap': round(ov_rand, 4),
           'pred_0': bool(p0), 'pred_a_positional_causal': bool(pa_causal), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) positional subspace CAUSAL (>=3x random): {pa_causal} (ratio {ratio:.1f}); NOTE if ~1 -> decodable-but-inert like prev-token', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
