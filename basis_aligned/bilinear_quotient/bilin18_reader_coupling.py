"""Is layer 1's 'dense' interaction actually path-separable? Per-reader coupling
matrices, computed from weights.

User insight (2026-08-17): the high-order interaction of L1's output bands is measured
through the WHOLE downstream stack, but every individual reader is architecturally
order-2 -- so the apparent density may be an aggregation artifact: each downstream
component couples only a few L1-direction pairs, different components couple DIFFERENT
pairs, and the union looks dense while being separable by computation path.

The object: for each downstream MLP j (j=2..17), the coupling matrix in L1's output
basis, B_j[a,b] = sum_dirs |V_a^T M_j^(d) V_b| aggregated over j's own top output
directions d -- i.e. how strongly j's quadratic couples L1-output directions a and b.
Computable from weights + L1's output basis; symmetric by construction (forms are
symmetrized). Top-48 L1 output directions.

REGISTERED PREDICTIONS:
  (a) per-reader concentration: for the median reader, the top 5% of |B_j| entries
      carry >= 40% of its mass (each reader reads FEW pairs);
  (b) reader disjointness: the mean pairwise cosine between different readers'
      normalised |B_j| matrices is <= 0.5 (different readers read substantially
      different pair-sets -- the union is dense, the parts are not);
  (c) the union of top-5% supports across readers covers >= 3x the pairs of any
      single reader's top-5% (the aggregation mechanism).
If (a)+(b) hold, the user's mutual-exclusivity picture is right in the structural
sense: direction pairs interact under SOME readers and not others, and the dense
whole-model interaction is path-separable."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_reader_coupling_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    V=orth(Vh[:K].T)                    # (D, K) L1 output basis
    Bs={}
    for j in range(2,18):
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V     # (4608, K)
        R=mlp.Right.weight.detach().float()@V
        Dw=mlp.Down.weight.detach().float()      # (D, 4608)
        # aggregate |coupling| over output dirs via the Gram trick:
        # B[a,b] = sum_i |sum_k D_ik L_ka R_kb| ~ use energy: sum_i (.)^2
        # C[a,b]^2 = sum_i (sum_k D_ik (L_ka R_kb + L_kb R_ka)/2)^2
        T=torch.einsum('ik,ka,kb->abi',Dw*0+1,L*0,R*0) if False else None
        # memory-light: loop over output dims in chunks
        B=torch.zeros(K,K,device=DEV)
        for lo in range(0,D,128):
            Dch=Dw[lo:lo+128]                     # (c, 4608)
            M=torch.einsum('ck,ka,kb->cab',Dch,L,R)
            M=0.5*(M+M.transpose(1,2))
            B+= (M**2).sum(0)
        B=B.sqrt()
        Bs[j]=B
    out={'readers':{}}
    # (a) concentration
    concs=[]
    iu=torch.triu_indices(K,K,offset=0)
    for j,B in Bs.items():
        v=B[iu[0],iu[1]]
        k5=max(1,int(0.05*v.numel()))
        conc=float(v.topk(k5).values.sum()/v.sum())
        concs.append(conc)
        out['readers'][j]={'top5_mass':conc}
    med=sorted(concs)[len(concs)//2]
    # (b) disjointness
    mats=[ (B/B.norm()).flatten() for B in Bs.values() ]
    cos=[]
    for a,b in itertools.combinations(range(len(mats)),2):
        cos.append(float(mats[a]@mats[b]))
    meancos=sum(cos)/len(cos)
    # (c) union coverage
    sups=[]
    for B in Bs.values():
        v=B[iu[0],iu[1]]
        k5=max(1,int(0.05*v.numel()))
        s=set(v.topk(k5).indices.tolist()); sups.append(s)
    union=set().union(*sups)
    cover=len(union)/max(len(sups[0]),1)
    out['median_top5_mass']=med
    out['mean_reader_cosine']=meancos
    out['union_over_single']=cover
    pa=med>=0.40; pb=meancos<=0.5; pc=cover>=3.0
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f'per-reader top-5% mass: median {med:.2f} '
          f'(range {min(concs):.2f}-{max(concs):.2f})')
    print(f'mean cross-reader coupling cosine: {meancos:.2f}')
    print(f'union of top-5% supports / single reader: {cover:.1f}x')
    print(f"\n(a) readers concentrated (>=0.40): {'HELD' if pa else 'FAILED'}")
    print(f"(b) readers disjoint (cos<=0.5): {'HELD' if pb else 'FAILED'}")
    print(f"(c) union >=3x single: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
