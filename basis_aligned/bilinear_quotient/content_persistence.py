"""SYMMETRIC companion to §962: is CONTENT carried/accumulated (does NOT recover after early ablation) or also
re-aggregated? §962 showed class is FULLY re-derived (ablate at L2 -> 98% recovered at L15). Do the matched test
for content: ablate the content/topic subspace at L8, decode the L15 topic; measure recovery vs clean. Content is
built by long-range attention pooling (§929/§932), so later layers CAN re-aggregate from context — the question is
how MUCH recovers, vs class's 98%.

REGISTERED PREDICTIONS:
  (0) SANITY: clean L15 topic decode >> base; ablating the content subspace AT L15 drops it toward base.
  (a) CONTENT LESS RE-DERIVED THAN CLASS: ablating content at L8 recovers a SMALLER fraction at L15 than class did
      (§962: 0.98) -> content is more carried/accumulated (partial re-aggregation by later attention, but not the
      near-complete rebuild class shows); report the recovered fraction and contrast with §962;
  (b) report L15 topic decode: clean, after-L8-ablation, after-L15-ablation, base."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_persistence_results.json'
NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 12; RCONTENT = 24; RIDGE = 1e2; L_EARLY = 8; L_LATE = 15
ABL = {'L': -1, 'U': None}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def abl_hook(L):
    def h(mo, i_, o_):
        if ABL['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; U = ABL['U']; v = y.reshape(-1, D)
        v2 = v - (v @ U) @ U.T
        return (v2.reshape(sh),) + tuple(o_[1:]) if isinstance(o_, tuple) else v2.reshape(sh)
    return h


def forward_capL(idx, capL):
    cap = {}
    def ch(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)
    hh = m.transformer.h[capL].register_forward_hook(ch)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return cap['r']


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


def content_and_topic(R, toks, pos):
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T
    return content


def acc(F_, y, ncls, tr, te):
    Y = torch.zeros(len(tr), ncls, device=DEV); Y[torch.arange(len(tr)), torch.tensor(y[tr], device=DEV)] = 1.0
    A = F_[tr].T @ F_[tr] + RIDGE*torch.eye(F_.shape[1], device=DEV); W = torch.linalg.solve(A, F_[tr].T @ Y)
    return float((F_[te] @ W).argmax(1).cpu().numpy().__eq__(y[te]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    hooks = [m.transformer.h[L_EARLY].register_forward_hook(abl_hook(L_EARLY)),
             m.transformer.h[L_LATE].register_forward_hook(abl_hook(L_LATE))]
    ABL['L'] = -1
    # clean L8 (for early content subspace) and L15 (for topic labels + late content subspace)
    R8 = []; R15 = []
    for i in range(0, nb, 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous()
        R8.append(forward_capL(idx, L_EARLY)); R15.append(forward_capL(idx, L_LATE))
    R8 = torch.cat(R8, 0); R15 = torch.cat(R15, 0)
    c8 = content_and_topic(R8, toks, pos); c15 = content_and_topic(R15, toks, pos)
    U8, _ = mean_subspace(c8, toks, 1)  # placeholder not used; build content subspaces below
    # content subspaces via topic clustering
    cn15 = c15/(c15.norm(dim=1, keepdim=True)+1e-9); topic = kmeans(cn15, K).cpu().numpy()
    U15c, _ = mean_subspace(c15, topic, RCONTENT)
    cn8 = c8/(c8.norm(dim=1, keepdim=True)+1e-9); topic8 = kmeans(cn8, K).cpu().numpy()
    U8c, _ = mean_subspace(c8, topic8, RCONTENT)
    n = R15.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n); tr, te = perm[:ntr], perm[ntr:]
    base = float(np.bincount(topic, minlength=K).max()/len(topic))
    clean = acc(c15, topic, K, tr, te)
    # ablate content subspace at L8, decode topic at L15
    ABL['L'] = L_EARLY; ABL['U'] = U8c; R15_a8 = []
    for i in range(0, nb, 4): R15_a8.append(forward_capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), L_LATE))
    R15_a8 = torch.cat(R15_a8, 0); c15_a8 = content_and_topic(R15_a8, toks, pos); after_L8 = acc(c15_a8, topic, K, tr, te)
    # ablate content subspace at L15, decode at L15
    ABL['L'] = L_LATE; ABL['U'] = U15c; R15_a15 = []
    for i in range(0, nb, 4): R15_a15.append(forward_capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), L_LATE))
    R15_a15 = torch.cat(R15_a15, 0); c15_a15 = content_and_topic(R15_a15, toks, pos); after_L15 = acc(c15_a15, topic, K, tr, te)
    ABL['L'] = -1
    for h in hooks: h.remove()
    out = {'base_rate': round(base, 4), 'clean_L15_topic': round(clean, 4),
           'after_L8_ablation': round(after_L8, 4), 'after_L15_ablation': round(after_L15, 4)}
    out['recovered_frac_after_L8'] = round((after_L8 - base)/(clean - base + 1e-9), 3)
    out['class_recovered_frac_ref_962'] = 0.982
    out['pred_a_content_less_rederived'] = bool(out['recovered_frac_after_L8'] < 0.982)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"L15 topic decode: clean {clean:.3f} | after-L8-ablation {after_L8:.3f} | after-L15-ablation {after_L15:.3f} | base {base:.3f}", flush=True)
    print(f"content recovered frac after L8 ablation {out['recovered_frac_after_L8']} (vs class 0.982, §962)", flush=True)
    print(f"(a) content less re-derived than class: {out['pred_a_content_less_rederived']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
