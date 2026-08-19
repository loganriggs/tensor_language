"""Realized pattern rank -- the RoPE-honest complement of the score-rank census.
The score-rank numbers (K = C^{1/2} Wq^T Wk C^{1/2}) ignore RoPE, which sits
between the projections and the dot product. Here: the effective rank of each
head's REALIZED pattern matrix on data (per sequence, the T x T masked score
matrix; eff-rank averaged over 6 sequences), layers 0,2,5,9,13,16.

REGISTERED PREDICTIONS: (a) realized patterns are low-rank: median head <= 12
(of 257); (b) the pre-RoPE weight rank predicts realized rank across heads
(Spearman >= 0.5) -- the weights-level census survives RoPE; (c) null: patterns
of a weight-shuffled attention (Wq rows permuted) have median realized rank
>= 2x the real median."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_pattern_census import get_patterns
NH,HD,D=9,128,1152
LAYERS=(0,2,5,9,13,16)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_pattern_rank_results.json')

def effrank(M):
    sv=torch.linalg.svdvals(M.float()); e=sv**2
    return float(e.sum()**2/(e**2).sum())

@torch.no_grad()
def main():
    t0=time.time()
    pats=get_patterns(FW[300:306,:257].to(DEV))
    kr=json.load(open('/workspace/tensor_language/basis_aligned/'
                      'bilinear_quotient/bilin18_score_rank_results.json'))
    real={};wrank={}
    for li in LAYERS:
        rs=[]
        for h in range(NH):
            r=sum(effrank(pats[li][b,h]) for b in range(6))/6
            rs.append(r)
        real[li]=rs
        row=kr[str(li)]
        per={}
        for x in row:
            per.setdefault(x['head'],[]).append(x['rank'])
        wrank[li]=[sum(per[h])/2 for h in range(NH)]
        print(f'L{li:2d}: realized ranks '+' '.join(f'{r:5.1f}' for r in rs),
              flush=True)
    # shuffled-weights null at L2 and L9
    g=torch.Generator(device=DEV).manual_seed(0)
    nulls=[]
    for li in (2,9):
        a=m.transformer.h[li].attn
        Wq=a.c_q.weight.data.clone()
        perm=torch.randperm(Wq.shape[0],generator=g,device=DEV)
        a.c_q.weight.data=Wq[perm]
        p2=get_patterns(FW[300:306,:257].to(DEV))
        a.c_q.weight.data=Wq
        for h in range(NH):
            nulls.append(sum(effrank(p2[li][b,h]) for b in range(6))/6)
    allr=[r for li in LAYERS for r in real[li]]
    allw=[r for li in LAYERS for r in wrank[li]]
    mr=sorted(allr)[len(allr)//2]
    mn=sorted(nulls)[len(nulls)//2]
    a_=torch.tensor(allr); b_=torch.tensor(allw)
    ra=a_.argsort().argsort().float(); rb=b_.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    sp=float((ra*rb).mean())
    pa=mr<=12; pb=sp>=0.5; pc=mn>=2*mr
    out={'realized':{str(k):v for k,v in real.items()},
         'weight_rank':{str(k):v for k,v in wrank.items()},
         'median_realized':mr,'median_null':mn,'spearman':sp,
         'pred_a':bool(pa),'pred_b':bool(pb),'null_c':bool(pc)}
    print(f'\nmedian realized {mr:.1f} | shuffled-null {mn:.1f} | '
          f'Spearman(weights, realized) {sp:+.2f}')
    print(f"(a) realized low-rank (<=12): {'HELD' if pa else 'FAILED'}")
    print(f"(b) weights predict realized (>=0.5): {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffled null >=2x: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
