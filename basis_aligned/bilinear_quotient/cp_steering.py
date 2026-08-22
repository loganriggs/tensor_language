"""CAUSAL SUFFICIENCY: can we STEER the prediction by injecting class+position content?
(capstone — all prior evidence is necessity via ablation; this tests sufficiency). If the
class+position subspace at mlp0 is the causal carrier of 'which token/class am I', then
replacing a token's class+position content with a SOURCE token B's content should make the
model predict as if the current token were B — i.e. move the next-token distribution toward
B's typical continuation p_B. Steer ALL positions' mlp0 output: keep each token's mean +
orthogonal complement, but swap its class+position projection for source B's:
  v_steer = (v - proj_cp(v)) + proj_cp(mu_B).
Measure KL(steered_avg_prediction || p_B) vs KL(normal || p_B); the class+position steer should
move predictions toward p_B far more than a random-direction steer of the same norm (NULL).

p_B = the model's average next-token distribution at positions where the current token IS B
(its empirical typical continuation).

REGISTERED PREDICTIONS:
  (0) SANITY: unsteered KL(normal||p_B) > 0; steering toward B is applied to all positions;
  (a) SUFFICIENT: KL(steered||p_B) << KL(normal||p_B) — injecting B's class+position content
      pulls the prediction toward B's continuation — and the drop is >> the random-direction
      null (matched norm); averaged over several source tokens B;
  (b) if steering barely moves prediction toward p_B (or no better than random), class+position
      at mlp0 is NOT causally sufficient to set the prediction (necessity without sufficiency).
  NULL: random-direction steer (same per-token norm as the cp swap) moves toward p_B far less."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'cp_steering_results.json'
NEVAL = 200; MINCOUNT = 5; RTOK = 64; RPOS = 32
SRC_TOKENS = [262, 257, 13, 198, 290]   # frequent, distinct: " the"," a",".","\n"," and"
STEER = {'on': False, 'mode': None, 'U': None, 'inject': None, 'rand': None}
LAYER = 0


def comp(): return m.transformer.h[LAYER].mlp


def hook(mo, i_, o_):
    if not STEER['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float(); U = STEER['U']
    cp = (v @ U) @ U.T                      # token's class+position content
    if STEER['mode'] == 'cp':
        v2 = v - cp + STEER['inject']      # swap in source B's cp content (broadcast)
    else:                                  # random: add a vector matched to the swap's norm, random direction
        delta = STEER['inject'] - cp       # the cp swap delta (per row)
        n = delta.norm(dim=1, keepdim=True)
        r = STEER['rand'].expand_as(v); r = r / r.norm(dim=1, keepdim=True)
        v2 = v + n * r
    yn = v2.reshape(sh).to(y.dtype)
    return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_mlp0(rows):
    cap = []; toks = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp().register_forward_hook(h)
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        c = idx.cpu().numpy(); toks.append(c.reshape(-1)); pos.append(np.broadcast_to(np.arange(c.shape[1]), c.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(toks), np.concatenate(pos)


def mean_subspace(O, labels, r, g):
    rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


@torch.no_grad()
def avg_pred_given_token(rows, tok_id):
    # model's average next-token distribution at positions where current token == tok_id
    STEER['on'] = False
    ps = []
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        lg = forward_logits(idx).float(); p = F.softmax(lg, -1).reshape(-1, lg.shape[-1])
        mk = (idx.reshape(-1) == tok_id)
        if mk.any(): ps.append(p[mk].cpu())
    return torch.cat(ps, 0).mean(0) if ps else None


@torch.no_grad()
def steered_avg_pred(rows):
    ps = []
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        lg = forward_logits(idx).float(); ps.append(F.softmax(lg, -1).reshape(-1, lg.shape[-1]).mean(0).cpu())
    return torch.stack(ps, 0).mean(0)


def kl(p, q):
    p = p + 1e-9; q = q + 1e-9; p = p/p.sum(); q = q/q.sum()
    return float((p * (p/q).log()).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    O, toks, pos = capture_mlp0(rows); gmean = O.mean(0, keepdim=True)
    Ut = mean_subspace(O, toks, RTOK, gmean); Up = mean_subspace(O, pos, RPOS, gmean)
    U = torch.linalg.svd(torch.cat([Ut, Up], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    STEER['U'] = U
    g = torch.Generator(device=DEV).manual_seed(0); STEER['rand'] = torch.randn(1, D, generator=g, device=DEV)
    steer_hook = comp().register_forward_hook(hook)   # BUGFIX: actually attach the steering hook
    # per-token mean mlp0 output for source tokens
    tmean = {}
    for b in SRC_TOKENS:
        mk = toks == b
        if mk.sum() >= MINCOUNT: tmean[b] = O[mk].mean(0, keepdim=True).to(DEV)
    STEER['on'] = False; p_normal = steered_avg_pred(rows)
    res = {}
    for b in SRC_TOKENS:
        if b not in tmean: continue
        p_B = avg_pred_given_token(rows, b)
        if p_B is None: continue
        inject = (tmean[b] @ U) @ U.T                # source B's class+position content
        STEER['inject'] = inject
        STEER['on'] = True; STEER['mode'] = 'cp'; p_cp = steered_avg_pred(rows)
        STEER['mode'] = 'rand'; p_rnd = steered_avg_pred(rows)
        STEER['on'] = False
        res[str(b)] = {'kl_normal_to_B': round(kl(p_normal, p_B), 4),
                       'kl_cpsteer_to_B': round(kl(p_cp, p_B), 4),
                       'kl_randsteer_to_B': round(kl(p_rnd, p_B), 4)}
        print(f"src {b}: KL(normal||B) {res[str(b)]['kl_normal_to_B']} -> KL(cp-steer||B) {res[str(b)]['kl_cpsteer_to_B']} | rand-steer {res[str(b)]['kl_randsteer_to_B']}", flush=True)
    steer_hook.remove()
    norm = np.mean([res[k]['kl_normal_to_B'] for k in res]); cp = np.mean([res[k]['kl_cpsteer_to_B'] for k in res]); rnd = np.mean([res[k]['kl_randsteer_to_B'] for k in res])
    out = {'per_source': res, 'mean_kl_normal': round(norm, 4), 'mean_kl_cpsteer': round(cp, 4), 'mean_kl_randsteer': round(rnd, 4),
           'pred_a_sufficient': bool(cp < 0.6*norm and cp < 0.8*rnd), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nMEAN KL to source-B continuation: normal {norm:.3f} | cp-steer {cp:.3f} | rand-steer {rnd:.3f}', flush=True)
    print(f'(a) class+position steering is causally SUFFICIENT (cp-steer moves toward B, < normal and < rand): {out["pred_a_sufficient"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
