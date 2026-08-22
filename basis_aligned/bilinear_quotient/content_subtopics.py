"""Does the high-dim content have TOPIC->SUBTOPIC hierarchy? (content-frontier, on-box). §866: content clusters
into ~12 topics; §908/§922: content is high-rank (needs >100 dims). Test whether re-clustering WITHIN a topic
yields finer, coherent SUB-topics — which would name the high-dim content as a hierarchy (topics of subtopics)
and explain why it needs many dimensions.

REGISTERED PREDICTIONS:
  (0) SANITY: the 12 top-level topics are coherent (§866 reproduced);
  (a) HIERARCHY: sub-clustering a topic yields sub-clusters with DISTINCT coherent distinctive-token sets
      (sub-cluster distinctive tokens differ from each other and refine the parent topic) -> content is
      hierarchical topic->subtopic, naming the high-dim structure as a coarse-to-fine hierarchy;
  (b) if sub-clusters are incoherent/identical, no clean hierarchy (report)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter
D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_subtopics_results.json'
CONTENT_L = 15; NEVAL = 320; RTOK = 64; RPOS = 32; K = 8; SUBK = 4; NDIST = 8
def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])
def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)
@torch.no_grad()
def capL(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h); forward_logits(idx); hh.remove(); return cap['r']
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
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :257].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    Rs = []
    for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous()))
    R = torch.cat(Rs, 0); toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(256), (nb, 256)).reshape(-1)
    tgt = S[:, 1:].reshape(-1); base = Counter(tgt[tgt>=0]); Nn = int((tgt>=0).sum())  # (nb,256) aligned with content
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1,keepdim=True)+1e-9)
    top = kmeans(cn, K).cpu().numpy()
    def distinct(mask):
        nc = Counter(); nj = int((tgt[mask]>=0).sum())
        for t in np.unique(tgt[mask]):
            if t<0: continue
            c=int((tgt[mask]==t).sum())
            if c<3: continue
            nc[t]=(c/max(nj,1))/((base.get(t,0)+1)/Nn)
        return [repr(d(int(t))) for t,_ in nc.most_common(NDIST)]
    out={'topics':{}}
    for j in range(K):
        mk = top==j
        if mk.sum()<80: continue
        parent = distinct(mk)
        sub = kmeans(cn[torch.tensor(mk,device=DEV)], SUBK).cpu().numpy()
        idxj = np.where(mk)[0]; subs=[]
        for sidx in range(SUBK):
            sm = np.zeros(len(tgt), bool); sm[idxj[sub==sidx]] = True
            if sm.sum()<20: continue
            subs.append(distinct(sm))
        out['topics'][j]={'parent': parent[:6], 'subtopics': [s[:6] for s in subs]}
        print(f"topic{j} {parent[:4]}", flush=True)
        for si,s in enumerate(subs): print(f"    sub{si}: {s[:5]}", flush=True)
    # coherence: mean pairwise Jaccard between sibling subtopics (low = distinct subtopics)
    jac=[]
    for j,tj in out['topics'].items():
        ss=[set(x) for x in tj['subtopics']]
        for a in range(len(ss)):
            for b in range(a+1,len(ss)): jac.append(len(ss[a]&ss[b])/max(len(ss[a]|ss[b]),1))
    out['mean_sibling_subtopic_jaccard']=round(float(np.mean(jac)),3) if jac else None
    out['pred_a_hierarchy']=bool(out['mean_sibling_subtopic_jaccard'] is not None and out['mean_sibling_subtopic_jaccard']<0.3)
    out['runtime_s']=round(time.time()-t0,1)
    json.dump(out, open(OUT,'w'), indent=1)
    print(f"\nmean sibling-subtopic Jaccard {out['mean_sibling_subtopic_jaccard']} (low=distinct subtopics=hierarchy)", flush=True)
    print(f"(a) content is hierarchical topic->subtopic: {out['pred_a_hierarchy']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")
if __name__=='__main__': main()
