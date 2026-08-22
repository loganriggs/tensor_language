"""DEFINITIVE whole-model understanding with a SMOOTH MAP (not a memorizing table). §905: a low-rank linear
map of named variables recovers the FRONT to 0.90 fresh (vs table 0.59) but the MIDDLE only ~0.10. Aggregate
it: replace ALL 36 components at once with a linear map from [current-token embedding, prev-token embedding,
continuous-topic-subspace projection] to their output, fit on 70% train rows, CE evaluated on held-out 30%.
This is the definitive 'how much of the whole model is a smooth generalizing function of named variables'.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-feature null ~0 or negative;
  (a) SMOOTH MAP BEATS THE TABLE: whole-model smooth-map understanding is HIGHER than the table's 0.30 (§901/
      904) because the front/readout are smooth token functions the map captures (mlp0 0.90 §905) — but still
      well below 1, bounded by the middle content which is not a function of these variables;
  (b) report the aggregate + shuffled-feature null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_smoothmap_results.json'
NEVAL = 200; SEQ = 256; CONTENT_L = 15; K = 12; RTOK = 64; RPOS = 32; RTOPIC = 24; RIDGE = 1e2; NLAYER = 18
REPL = {'mode': 'off', 'W': None, 'feat_batch': None, 'gmeans': None}
TAGS = [f"{k}{L}" for L in range(NLAYER) for k in ('attn', 'mlp')]


def submod(tag):
    k = 'attn' if tag.startswith('attn') else 'mlp'; L = int(tag[len(k):]); return getattr(m.transformer.h[L], k)


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off': return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if REPL['mode'] == 'mean':
            yn = REPL['gmeans'][tag].expand(B, T, D).clone()
        else:
            yn = (REPL['feat_batch'] @ REPL['W'][tag]).reshape(B, T, D)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


@torch.no_grad()
def ce_pass(test_blocks, feat, row0):
    tot = 0.0; n = 0; row = row0
    for i in range(0, test_blocks.shape[0], 8):
        bb = test_blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous(); bsz = idx.shape[0]
        REPL['feat_batch'] = feat[row:row+bsz*(SEQ-1)]
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel(); row += bsz*(SEQ-1)
    return tot/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True; trpos = np.repeat(TRAIN, SEQ-1)
    wte = m.transformer.wte.weight.detach().float()
    cur = S[:, :-1].reshape(-1); prev = np.full((nb, SEQ-1), -1, dtype=np.int64); prev[:, 1:] = S[:, :-2]; prev = prev.reshape(-1)
    # capture all 36 outputs + L15 content (one clean pass)
    hooks = [submod(t).register_forward_hook(repl_hook_factory(t)) for t in TAGS]
    cap = {}
    caph = []
    for t in TAGS:
        def mk(t):
            def h(mo, i_, o_): cap.setdefault(t, []).append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        caph.append(submod(t).register_forward_hook(mk(t)))
    c15 = []
    def hc(mo, i_, o_): c15.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hch = m.transformer.h[CONTENT_L].register_forward_hook(hc)
    REPL['mode'] = 'off'
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in caph: h.remove(); hch.remove()
    outs = {t: torch.cat(cap[t], 0) for t in TAGS}; R15 = torch.cat(c15, 0)
    posarr = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    Utok, g = mean_subspace(R15, cur, RTOK); Upos, _ = mean_subspace(R15, posarr.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g) - ((R15-g)@Ucp)@Ucp.T
    Utopic, _ = mean_subspace(content, kmeans(content/(content.norm(dim=1,keepdim=True)+1e-9), K).cpu().numpy(), RTOPIC)
    feat = torch.cat([wte[torch.tensor(cur, device=DEV)], wte[torch.tensor(np.where(prev>=0,prev,0), device=DEV)],
                      content @ Utopic, torch.ones(len(cur), 1, device=DEV)], 1)
    rng = np.random.RandomState(0); featperm = feat[torch.tensor(rng.permutation(len(cur)), device=DEV)]
    tr = torch.tensor(trpos, device=DEV)
    Ftr = feat[tr]; A = Ftr.T @ Ftr + RIDGE*torch.eye(feat.shape[1], device=DEV)
    Ftr_sh = featperm[tr]; A_sh = Ftr_sh.T @ Ftr_sh + RIDGE*torch.eye(feat.shape[1], device=DEV)
    W = {}; W_sh = {}; gmeans = {}
    for t in TAGS:
        W[t] = torch.linalg.solve(A, Ftr.T @ outs[t][tr]); W_sh[t] = torch.linalg.solve(A_sh, Ftr_sh.T @ outs[t][tr]); gmeans[t] = outs[t].mean(0)
        cap[t] = None
    REPL['gmeans'] = gmeans
    test_blocks = blocks[~TRAIN]; row0 = ntr*(SEQ-1)
    REPL['mode'] = 'off'; ce_full = ce_pass(test_blocks, feat, row0)
    REPL['mode'] = 'mean'; ce_mean = ce_pass(test_blocks, feat, row0)
    denom = max(ce_mean - ce_full, 1e-6)
    REPL['mode'] = 'set'; REPL['W'] = W; ce_map = ce_pass(test_blocks, feat, row0)
    REPL['W'] = W_sh; ce_sh = ce_pass(test_blocks, featperm, row0)
    REPL['mode'] = 'off'
    for h in hooks: h.remove()
    out = {'ce_full': round(ce_full, 3), 'ce_all_mean': round(ce_mean, 3),
           'smoothmap_understanding': round(float((ce_mean-ce_map)/denom), 3),
           'shuffled_feature_null': round(float((ce_mean-ce_sh)/denom), 3),
           'table_ref_fresh': 0.29, 'table_ref_backoff': 0.315, 'runtime_s': round(time.time()-t0, 1)}
    out['pred_a_smooth_beats_table'] = bool(out['smoothmap_understanding'] > 0.35 and out['smoothmap_understanding'] > out['shuffled_feature_null'] + 0.2)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"WHOLE-MODEL SMOOTH-MAP (all 36, held-out): full {ce_full:.3f}, all-mean {ce_mean:.3f}", flush=True)
    print(f"smooth-map understanding {out['smoothmap_understanding']} (shuffled-feature null {out['shuffled_feature_null']}) vs table fresh 0.29/backoff 0.315", flush=True)
    print(f"(a) smooth map beats table: {out['pred_a_smooth_beats_table']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
