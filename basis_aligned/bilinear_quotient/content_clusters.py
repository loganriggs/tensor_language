"""DATA-DRIVEN: what does the CONTENT residual encode? (naming attempt via unsupervised discovery, §864).
Prior probes IMPOSED categories (grammar, embedding-semantics) and failed. Instead let the data speak:
take the content residual (L15 residual with the class+position projection REMOVED), k-means it, and read
what each cluster's positions share — their most common current token, most common NEXT token, and example
decoded contexts. If clusters are coherent (a nameable content facet — numeric context, dialogue, a topic),
that names content facets; if incoherent, the content is genuinely structureless at this granularity.

REGISTERED PREDICTIONS:
  (0) SANITY: clusters are non-trivial (varied sizes);
  (a) if clusters have coherent shared next-tokens/contexts, name those facets;
  (b) if clusters are incoherent (no shared context/next-token), the content residual has no clean
      cluster structure — the wall is confirmed data-driven, not just probe-limited."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter
D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_clusters_results.json'; NEVAL = 260; RTOK = 64; RPOS = 32; K = 16; READ_L = 15
def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)
def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows=[]; wt=[]
    for t in np.unique(labels):
        mk = labels==t
        if mk.sum()<5: continue
        rows.append(O[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows,0)*torch.tensor(wt,device=O.device,dtype=O.dtype)[:,None]
    return torch.linalg.svd(M,full_matrices=False)[2][:min(r,M.shape[0])].T.contiguous(), g
def kmeans(X,k,iters=25,seed=0):
    g=torch.Generator(device=X.device).manual_seed(seed); c=X[torch.randperm(X.shape[0],generator=g,device=X.device)[:k]].clone()
    for _ in range(iters):
        a=torch.cdist(X,c).argmin(1)
        for j in range(k):
            mk=a==j
            if mk.any(): c[j]=X[mk].mean(0)
    return a
@torch.no_grad()
def main():
    t0=time.time(); cl.use_state(PT+'census_state_diverse.pt'); rows=cl.fineweb_rows(NEVAL)
    import tiktoken; enc=tiktoken.get_encoding('gpt2'); d=lambda i: enc.decode([int(i)])
    outs=[]; seqs=[]
    def h(mo,i_,o_): outs.append((o_[0] if isinstance(o_,tuple) else o_).detach().float())
    hh=m.transformer.h[READ_L].register_forward_hook(h)
    for i in range(0,NEVAL,4):
        idx=rows[i:i+4,:257].to(DEV)[:,:-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    hh.remove()
    R=torch.cat([o.reshape(-1,D) for o in outs],0); S=np.concatenate(seqs,0); T=S.shape[1]
    cur=S.reshape(-1); tgt=np.full_like(S,-1); tgt[:,:-1]=S[:,1:]; tgt=tgt.reshape(-1); pos=np.broadcast_to(np.arange(T),S.shape).reshape(-1)
    Utok,g=mean_subspace(R,cur,RTOK); Upos,_=mean_subspace(R,pos.astype(np.int64),RPOS)
    Ucp=torch.linalg.svd(torch.cat([Utok,Upos],1),full_matrices=False)[0][:,:RTOK+RPOS].contiguous()
    content=(R-g)-((R-g)@Ucp)@Ucp.T                       # content residual (grammar removed)
    content=content/(content.norm(dim=1,keepdim=True)+1e-9)
    a=kmeans(content,K).cpu().numpy()
    clusters=[]
    for j in range(K):
        mk=a==j; n=int(mk.sum())
        if n<10: continue
        curc=Counter(cur[mk]); nxtc=Counter(tgt[mk][tgt[mk]>=0])
        clusters.append({'cluster':j,'n':n,
                         'top_current':[repr(d(t)) for t,_ in curc.most_common(6)],
                         'top_next':[repr(d(t)) for t,_ in nxtc.most_common(6)]})
    out={'k':K,'read_layer':READ_L,'clusters':clusters,'runtime_s':round(time.time()-t0,1)}
    json.dump(out,open(OUT,'w'),indent=1)
    for c in clusters: print(f"cluster {c['cluster']} (n={c['n']}): cur {c['top_current'][:5]} -> next {c['top_next'][:5]}",flush=True)
    print(f"wrote {OUT}")
if __name__=='__main__': main()
