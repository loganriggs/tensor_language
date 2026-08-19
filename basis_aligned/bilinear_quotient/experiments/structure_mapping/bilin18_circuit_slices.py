"""CIRCUIT LOOP, iteration 1, step SLICE: do well-predicted tokens cluster
by which components own them? Data: the fingerprint atlas (36 components x
16384 tokens of per-token damage) -- no new forward passes needed for
discovery. Take tokens with base CE below the median (the model's
confident half), represent each token by its 36-dim damage profile
(unit-normalized), k-means k=8.

REGISTERED PREDICTIONS: (a) structure is real: mean within-cluster cosine
to centroid exceeds the same statistic under a token-shuffle null (each
fingerprint independently permuted, destroying co-structure) by >= 0.10;
(b) component-sparsity: >= 3 of 8 clusters have their top-3 components
carrying >= 50% of mean absolute damage (circuit-like ownership, not
diffuse); (c) recognition: at least one sparse cluster's top components
include a known circuit pair (attn6+mlp5 cargo, or mlp1+attn1 front, or
mlp16/attn17+mlp16 tail) -- the discovery instrument re-finds what manual
work found. Outputs cluster assignments + per-cluster component profiles
to bilin18_circuit_slices.pt for the LOCALIZE step."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
DEV='cuda'
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bilin18_circuit_slices_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    d=torch.load(PT+'bilin18_fingerprint_atlas.pt',weights_only=False)
    base=d['base'].float().to(DEV)
    keys=sorted(d['fingerprints'])
    M=torch.stack([d['fingerprints'][k].float().to(DEV) for k in keys])
    med=base.median()
    sl=base<=med
    X=M[:,sl].T                                   # (Ntok, 36)
    Xn=X/X.norm(dim=1,keepdim=True).clamp_min(1e-8)
    def kmeans(Z,k=8,iters=60,seed=0):
        g=torch.Generator(device=DEV).manual_seed(seed)
        C=Z[torch.randperm(len(Z),generator=g,device=DEV)[:k]].clone()
        for _ in range(iters):
            a=(Z@C.T).argmax(1)
            for j in range(k):
                m_=a==j
                if m_.any(): C[j]=Z[m_].mean(0)
                C[j]=C[j]/C[j].norm().clamp_min(1e-8)
        return a,C
    a,C=kmeans(Xn)
    within=float((Xn*C[a]).sum(1).mean())
    g=torch.Generator(device=DEV).manual_seed(1)
    Mn=torch.stack([M[i][torch.randperm(M.shape[1],generator=g,device=DEV)]
                    for i in range(M.shape[0])])
    Xs=Mn[:,sl].T
    Xs=Xs/Xs.norm(dim=1,keepdim=True).clamp_min(1e-8)
    a0,C0=kmeans(Xs)
    within0=float((Xs*C0[a0]).sum(1).mean())
    print(f'within-cluster cosine: real {within:.3f} vs shuffle-null '
          f'{within0:.3f}',flush=True)
    sparse=0; recog=False; clusters=[]
    KNOWN=[{'attn6','mlp5'},{'mlp1','attn1'},{'attn17','mlp16'}]
    for j in range(8):
        m_=a==j
        prof=X[m_].abs().mean(0)
        share=prof/prof.sum()
        top=share.argsort(descending=True)[:3]
        tshare=float(share[top].sum())
        tnames=[keys[i] for i in top.tolist()]
        is_sp=tshare>=0.5
        sparse+=is_sp
        if is_sp and any(k<=set(tnames) for k in KNOWN): recog=True
        clusters.append({'size':int(m_.sum()),'top3':tnames,
                         'top3_share':tshare})
        print(f'cluster {j}: n={int(m_.sum()):5d} top3 {tnames} '
              f'share {tshare:.2f}{"  SPARSE" if is_sp else ""}',flush=True)
    pa=(within-within0)>=0.10; pb=sparse>=3; pc=recog
    torch.save({'assign':a.cpu(),'slice_mask':sl.cpu(),'keys':keys,
                'centroids':C.cpu()},PT+'bilin18_circuit_slices.pt')
    out={'within':within,'null':within0,'clusters':clusters,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) structure real (gap >=0.10): {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=3 sparse clusters: {'HELD' if pb else 'FAILED'} ({sparse})")
    print(f"(c) re-finds a known circuit: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
