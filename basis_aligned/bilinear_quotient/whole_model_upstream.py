"""CLEAN whole-model understanding benchmark (§913 fix): replace all 36 components at once with a smooth map,
using each component's OWN clean INPUT residual (rank-128 PCA) as the content feature — upstream by
construction, so no downstream leakage (unlike the r512 L15-content version, §913). Features are computed from
the UNMODIFIED forward (fixed), so they are not contaminated by the replacements. This is the honest rank-128
generalizing whole-model number.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-feature null ~0;
  (a) HONEST rank-128 benchmark is ABOVE the 12-topic 0.41 (front/readout are smooth functions of their input
      captured by rank-128) but WELL BELOW the leaky 0.85 (§913) — the clean number reflects genuine
      generalizing structure, bounded by the middle's bilinear nonlinearity;
  (b) report the aggregate + null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_upstream_results.json'
NEVAL = 200; SEQ = 256; RPCA = 128; RIDGE = 1e2; NLAYER = 18
TAGS = [f"{k}{L}" for L in range(NLAYER) for k in ('attn', 'mlp')]
REPL = {'mode': 'off', 'W': None, 'inpca': None, 'wc': None, 'wp': None, 'ones': None, 'row': 0, 'gmeans': None}


def submod(tag):
    k = 'attn' if tag.startswith('attn') else 'mlp'; L = int(tag[len(k):]); return getattr(m.transformer.h[L], k)


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off': return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape; r = REPL['row']; nrow = B*T
        if REPL['mode'] == 'mean':
            yn = REPL['gmeans'][tag].expand(B, T, D).clone()
        else:
            feat = torch.cat([REPL['wc'][r:r+nrow], REPL['wp'][r:r+nrow], REPL['inpca'][tag][r:r+nrow], REPL['ones'][r:r+nrow]], 1)
            yn = (feat @ REPL['W'][tag]).reshape(B, T, D)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def ce_pass(test_blocks, row0):
    tot = 0.0; n = 0; REPL['row'] = row0
    for i in range(0, test_blocks.shape[0], 8):
        bb = test_blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel(); REPL['row'] += idx.shape[0]*(SEQ-1)
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True; trpos = np.repeat(TRAIN, SEQ-1)
    wte = m.transformer.wte.weight.detach().float()
    cur = S[:, :-1].reshape(-1); prev = np.full((nb, SEQ-1), -1, dtype=np.int64); prev[:, 1:] = S[:, :-2]; prev = prev.reshape(-1)
    # capture each component's clean OUTPUT (post) and INPUT (pre)
    hooks = [submod(t).register_forward_hook(repl_hook_factory(t)) for t in TAGS]
    outc = {t: [] for t in TAGS}; inc = {t: [] for t in TAGS}; caph = []
    for t in TAGS:
        def mkpost(t):
            def h(mo, i_, o_): outc[t].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        def mkpre(t):
            def h(mo, a): inc[t].append(a[0].detach().float().reshape(-1, D))
            return h
        caph.append(submod(t).register_forward_hook(mkpost(t)))
        caph.append(submod(t).register_forward_pre_hook(mkpre(t)))
    REPL['mode'] = 'off'
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in caph: h.remove()
    outs = {t: torch.cat(outc[t], 0) for t in TAGS}; ins = {t: torch.cat(inc[t], 0) for t in TAGS}
    wc = wte[torch.tensor(cur, device=DEV)]; wp = wte[torch.tensor(np.where(prev >= 0, prev, 0), device=DEV)]
    ones = torch.ones(len(cur), 1, device=DEV)
    tr = torch.tensor(trpos, device=DEV)
    rng = np.random.RandomState(0); perm = torch.tensor(rng.permutation(len(cur)), device=DEV)
    W = {}; W_sh = {}; inpca = {}; inpca_sh = {}; gmeans = {}
    for t in TAGS:
        Xin = ins[t]; gci = Xin.mean(0, keepdim=True)
        Vp = torch.linalg.svd(Xin[tr] - gci, full_matrices=False)[2][:RPCA].T.contiguous()
        ip = (Xin - gci) @ Vp; inpca[t] = ip; inpca_sh[t] = ip[perm]
        feat = torch.cat([wc, wp, ip, ones], 1); Ftr = feat[tr]
        A = Ftr.T @ Ftr + RIDGE*torch.eye(feat.shape[1], device=DEV); W[t] = torch.linalg.solve(A, Ftr.T @ outs[t][tr])
        featp = torch.cat([wc[perm], wp[perm], ip[perm], ones], 1); Fp = featp[tr]
        Ash = Fp.T @ Fp + RIDGE*torch.eye(feat.shape[1], device=DEV); W_sh[t] = torch.linalg.solve(Ash, Fp.T @ outs[t][tr])
        gmeans[t] = outs[t].mean(0); outc[t] = None; inc[t] = None
    REPL['wc'] = wc; REPL['wp'] = wp; REPL['ones'] = ones; REPL['gmeans'] = gmeans
    te_blocks = blocks[~TRAIN]; row0 = ntr*(SEQ-1)
    REPL['mode'] = 'off'; ce_full = ce_pass(te_blocks, row0)
    REPL['mode'] = 'mean'; ce_mean = ce_pass(te_blocks, row0)
    denom = max(ce_mean - ce_full, 1e-6)
    REPL['mode'] = 'set'; REPL['W'] = W; REPL['inpca'] = inpca; ce_map = ce_pass(te_blocks, row0)
    REPL['W'] = W_sh; REPL['inpca'] = inpca_sh; REPL['wc'] = wc[perm]; REPL['wp'] = wp[perm]; ce_sh = ce_pass(te_blocks, row0)
    REPL['mode'] = 'off'
    for h in hooks: h.remove()
    out = {'ce_full': round(ce_full, 3), 'ce_all_mean': round(ce_mean, 3),
           'upstream_smoothmap_understanding': round(float((ce_mean-ce_map)/denom), 3),
           'shuffled_feature_null': round(float((ce_mean-ce_sh)/denom), 3),
           'ref_12topic_0.41': 0.406, 'ref_leaky_r512_0.85': 0.849, 'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"CLEAN whole-model UPSTREAM smooth-map (all 36, held-out): full {ce_full:.3f} all-mean {ce_mean:.3f}", flush=True)
    print(f"understanding {out['upstream_smoothmap_understanding']} (null {out['shuffled_feature_null']}) vs 12-topic 0.41 / leaky-r512 0.85", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
