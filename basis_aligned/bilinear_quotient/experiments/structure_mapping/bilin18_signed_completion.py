"""Completing §56: signed cosines across readers and for QK heads.

REGISTERED PREDICTIONS: (a) signed CROSS-reader form cosine <= 0.15 (functional
orthogonality extends across readers, not just within L5); (b) signed cross-head QK
cosine <= 0.25 (heads' signed score functionals are also near-orthogonal despite the
0.79 unsigned envelope)."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
NH,HD,D=9,128,1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_signed_completion_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    V=orth(Vh[:K].T)
    perreader={}
    for j in (2,5,13,17):
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:8].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        mats=[]
        for f in range(8):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            M=0.5*(M+M.T)
            mats.append((M/M.norm().clamp_min(1e-12)).flatten())
        perreader[j]=mats
    xc=[]
    for a,b in itertools.combinations(perreader,2):
        for u in perreader[a]:
            for w in perreader[b]:
                xc.append(abs(float(u@w)))
    cross=sum(xc)/len(xc)
    heads=[]
    for jj in (2,3,4):
        at=m.transformer.h[jj].attn
        def hm(W): return (W.weight.detach().float()@V).view(NH,HD,K)
        Q1,K1,Q2,K2=hm(at.c_q),hm(at.c_k),hm(at.c_q2),hm(at.c_k2)
        for h in range(NH):
            B=(torch.einsum('ea,eb->ab',Q1[h],K1[h])
               +torch.einsum('ea,eb->ab',Q2[h],K2[h]))
            B=0.5*(B+B.T)
            heads.append((B/B.norm().clamp_min(1e-12)).flatten())
    qc=[abs(float(heads[a]@heads[b]))
        for a,b in itertools.combinations(range(len(heads)),2)]
    qcos=sum(qc)/len(qc)
    out={'signed_cross_reader_cos':cross,'signed_qk_cross_cos':qcos}
    pa=cross<=0.15; pb=qcos<=0.25
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f'signed cross-reader form cos: {cross:.3f} -> (a) '
          f"{'HELD' if pa else 'FAILED'}")
    print(f'signed cross-head QK cos:     {qcos:.3f} -> (b) '
          f"{'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
