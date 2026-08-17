"""Path-separation at the right resolution: PER-FORM coupling of L1's directions.

§54: the whole-reader coupling test was confounded -- aggregating over 1152 output
directions per reader flattens any per-direction sparsity (cos 0.99 between readers is
what central-limit flattening produces regardless of underlying structure). The
resolution the user's hypothesis lives at is the individual FORM: reader j's output
direction d couples L1-directions (a,b) through B_{j,d}[a,b] = V_a^T M_j^(d) V_b, a
K x K matrix of rank <= rank(M_d) -- necessarily structured. Sample 32 output
directions (top output-PCs) per reader for readers j in {2,3,5,9,13,17}, K = 48.

REGISTERED PREDICTIONS:
  (a) per-form concentration: median top-5%-entry mass of |B_{j,d}| >= 0.35
      (individual forms couple few pairs);
  (b) within-reader disjointness: mean |cosine| between different forms' coupling
      matrices of the SAME reader <= 0.5 (different outputs of one reader read
      different pairs);
  (c) cross-reader disjointness: mean |cosine| across readers' forms <= within-reader
      mean (readers differ at least as much as their own outputs do).
If (a)+(b) hold, the dense whole-model interaction IS an aggregation of sparse,
largely disjoint per-form couplings -- the user's structural-exclusivity picture at
the resolution where the architecture actually reads."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=32
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_perform_coupling_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    V=orth(Vh[:K].T)
    out={'readers':{}}
    allmats={}
    for j in READERS:
        # reader j's own top output directions
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=(mlp.Down.weight.detach().float().T@P)      # (4608, NF)
        mats=[]
        for f in range(NF):
            c=DwP[:,f]
            M=torch.einsum('k,ka,kb->ab',c,L,R)
            M=0.5*(M+M.T)
            mats.append(M.abs())
        allmats[j]=mats
        iu=torch.triu_indices(K,K)
        concs=[]
        for M in mats:
            v=M[iu[0],iu[1]]
            k5=max(1,int(0.05*v.numel()))
            concs.append(float(v.topk(k5).values.sum()/v.sum().clamp_min(1e-12)))
        flat=[(M/M.norm().clamp_min(1e-12)).flatten() for M in mats]
        wc=[float(flat[a]@flat[b]) for a,b in
            itertools.combinations(range(NF),2)]
        out['readers'][j]={'median_top5':sorted(concs)[NF//2],
                           'within_cos':sum(wc)/len(wc)}
        print(f'reader L{j}: per-form top-5% mass median '
              f'{sorted(concs)[NF//2]:.2f} | within-reader cos '
              f'{sum(wc)/len(wc):.2f}',flush=True)
    xc=[]
    keys=list(allmats)
    for a,b in itertools.combinations(keys,2):
        fa=[(M/M.norm().clamp_min(1e-12)).flatten() for M in allmats[a][:8]]
        fb=[(M/M.norm().clamp_min(1e-12)).flatten() for M in allmats[b][:8]]
        for u in fa:
            for w in fb: xc.append(float(u@w))
    meds=[out['readers'][j]['median_top5'] for j in READERS]
    wins=[out['readers'][j]['within_cos'] for j in READERS]
    med_all=sorted(meds)[len(meds)//2]
    win_all=sum(wins)/len(wins)
    x_all=sum(xc)/len(xc)
    out['median_top5_overall']=med_all
    out['within_cos_overall']=win_all
    out['cross_cos_overall']=x_all
    pa=med_all>=0.35; pb=win_all<=0.5; pc=x_all<=win_all
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f'\noverall: per-form top-5% {med_all:.2f} | within-reader cos '
          f'{win_all:.2f} | cross-reader cos {x_all:.2f}')
    print(f"(a) forms concentrated: {'HELD' if pa else 'FAILED'} | "
          f"(b) within-reader disjoint: {'HELD' if pb else 'FAILED'} | "
          f"(c) cross <= within: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
