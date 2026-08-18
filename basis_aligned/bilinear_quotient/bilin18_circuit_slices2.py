"""SLICE step v2. v1 found real structure (0.72 vs 0.54 null) but zero
sparse clusters: raw damage profiles cluster by global loudness (attn1/2/4,
mlp17 own everything). Fix: DISTINCTIVE ownership -- z-score each
component's damage across the well-predicted slice (so a component scores
high on a token only where it is unusually responsible relative to its own
norm), then cluster (k=10 for finer grain).

REGISTERED PREDICTIONS on the distinctive representation: (a) structure
real: within-cosine gap over shuffle-null >= 0.10; (b) >= 3 of 10 clusters
component-sparse (top-3 |z| share >= 0.40 -- bar adjusted from 0.50: z-space
spreads mass, floor measured by the null's top-3 share, reported); (c) at
least one sparse cluster's top-3 include a known pair (attn6+mlp5,
mlp1+attn1, attn17+mlp16, or mlp16+mlp17); (d) cluster-null top-3 share
reported per the floor rule."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
DEV='cuda'
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'bilin18_circuit_slices2_results.json'

@torch.no_grad()
def main():
    t0=time.time()
    d=torch.load(PT+'bilin18_fingerprint_atlas.pt',weights_only=False)
    base=d['base'].float().to(DEV)
    keys=sorted(d['fingerprints'])
    M=torch.stack([d['fingerprints'][k].float().to(DEV) for k in keys])
    sl=base<=base.median()
    Z=M[:,sl]
    Z=(Z-Z.mean(1,keepdim=True))/Z.std(1,keepdim=True).clamp_min(1e-8)
    X=Z.T
    Xn=X/X.norm(dim=1,keepdim=True).clamp_min(1e-8)
    def kmeans(A,k=10,iters=80,seed=0):
        g=torch.Generator(device=DEV).manual_seed(seed)
        C=A[torch.randperm(len(A),generator=g,device=DEV)[:k]].clone()
        for _ in range(iters):
            a=(A@C.T).argmax(1)
            for j in range(k):
                m_=a==j
                if m_.any():
                    C[j]=A[m_].mean(0)
                    C[j]=C[j]/C[j].norm().clamp_min(1e-8)
        return a,C
    a,C=kmeans(Xn)
    within=float((Xn*C[a]).sum(1).mean())
    g=torch.Generator(device=DEV).manual_seed(1)
    Zs=torch.stack([Z[i][torch.randperm(Z.shape[1],generator=g,device=DEV)]
                    for i in range(Z.shape[0])])
    Xs=Zs.T; Xs=Xs/Xs.norm(dim=1,keepdim=True).clamp_min(1e-8)
    a0,C0=kmeans(Xs)
    within0=float((Xs*C0[a0]).sum(1).mean())
    nulltop=[]
    for j in range(10):
        m_=a0==j
        if not m_.any(): continue
        pr=Xs[m_].abs().mean(0); sh=pr/pr.sum()
        nulltop.append(float(sh.sort(descending=True).values[:3].sum()))
    nullfloor=sorted(nulltop)[len(nulltop)//2]
    print(f'within: real {within:.3f} null {within0:.3f} | '
          f'null top-3 floor {nullfloor:.2f}',flush=True)
    sparse=0; recog=False; clusters=[]
    KNOWN=[{'attn6','mlp5'},{'mlp1','attn1'},{'attn17','mlp16'},
           {'mlp16','mlp17'}]
    for j in range(10):
        m_=a==j
        prof=X[m_].abs().mean(0)
        share=prof/prof.sum()
        top=share.argsort(descending=True)[:3]
        tshare=float(share[top].sum())
        tnames=[keys[i] for i in top.tolist()]
        is_sp=tshare>=0.40
        sparse+=is_sp
        if is_sp and any(k<=set(tnames) for k in KNOWN): recog=True
        clusters.append({'size':int(m_.sum()),'top3':tnames,
                         'top3_share':tshare})
        print(f'cluster {j}: n={int(m_.sum()):5d} top3 {tnames} '
              f'share {tshare:.2f}{"  SPARSE" if is_sp else ""}',flush=True)
    pa=(within-within0)>=0.10; pb=sparse>=3; pc=recog
    torch.save({'assign':a.cpu(),'slice_mask':sl.cpu(),'keys':keys,
                'centroids':C.cpu()},PT+'bilin18_circuit_slices2.pt')
    out={'within':within,'null':within0,'null_top3_floor':nullfloor,
         'clusters':clusters,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f"\n(a) structure (gap>=0.10): {'HELD' if pa else 'FAILED'}")
    print(f"(b) >=3 sparse (top3>=0.40): {'HELD' if pb else 'FAILED'} ({sparse})")
    print(f"(c) re-finds known pair: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
