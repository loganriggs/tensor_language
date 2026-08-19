"""Completes the vocabulary-scope arc. The picture from weak_writers +
causal_split + random_v: sharing is writer-general but not universal (L6
private at 0.16, below the random-V floor 0.23), causality irrelevant
(upstream folds 0.51), global geometry insufficient. Remaining hypothesis:
readers share a vocabulary over a COMMUNAL subspace -- the thing L0 and L1
write and the lambda1*x0 re-injection keeps live everywhere -- and a writer's
shareability is just how much of its output lands in that subspace. Measure:
for each writer W in (0,1,6,9,12), the fraction of W's top-48 output variance
lying in span(V_L0 union V_L1) (96-dim communal span; for W=0 and W=1 use the
OTHER one's 48-dim span alone).

REGISTERED PREDICTIONS with shareability s=(0.70,0.64,0.16,0.54,0.51):
(a) L6 has the MINIMUM communal overlap of the five (rank prediction);
(b) the overlap ordering matches shareability ordering on the three
mid-writers: L6 < L12 <= L9 or L6 < L9 <= L12 -- i.e. L6 strictly last;
(c) null: a random 96-dim subspace absorbs <= half of what the communal span
absorbs for every writer."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_communal_overlap_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    Vs={}
    spec={}
    for Wl in (0,1,6,9,12):
        Y=grab(Wl,0,120); Yc=(Y-Y.mean(0)).float()
        U_,S,Vh=torch.linalg.svd(Yc, full_matrices=False)
        Vs[Wl]=orth(Vh[:K].T); spec[Wl]=(S[:K]**2)
    g=torch.Generator(device=DEV).manual_seed(0)
    res={}
    for Wl in (0,1,6,9,12):
        if Wl==0: C=Vs[1]
        elif Wl==1: C=Vs[0]
        else: C=orth(torch.cat([Vs[0],Vs[1]],dim=1))
        w=spec[Wl]/spec[Wl].sum()
        ov=float((w*((Vs[Wl].T@C)**2).sum(1)).sum())
        Rn=orth(torch.randn(D,C.shape[1],device=DEV,generator=g))
        ovr=float((w*((Vs[Wl].T@Rn)**2).sum(1)).sum())
        res[Wl]=(ov,ovr)
        print(f'writer L{Wl:2d}: communal overlap {ov:.3f} (random-span {ovr:.3f})',
              flush=True)
    mids={k:res[k][0] for k in (6,9,12)}
    pa=res[6][0]==min(v[0] for v in res.values())
    pb=mids[6]<mids[9] and mids[6]<mids[12]
    pc=all(v[1]<=0.5*v[0] for v in res.values())
    out={'overlap':{str(k):{'communal':v[0],'random':v[1]}
                    for k,v in res.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) L6 minimum of five: {'HELD' if pa else 'FAILED'}")
    print(f"(b) L6 strictly last among mids: {'HELD' if pb else 'FAILED'}")
    print(f"(c) random-span <= half everywhere: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
