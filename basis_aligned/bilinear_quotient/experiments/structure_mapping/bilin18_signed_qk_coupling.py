"""Two caveats of §55 closed: signed coupling overlap, and QK circuits as readers.

REGISTERED PREDICTIONS:
  (a) signed within-reader coupling cosine drops below 0.40 (much of the 0.63 was
      absolute-value inflation) -- if it stays high, the shared template is a real
      signed object, stronger conclusion;
  (b) attention QK readers ARE more specialised than MLP forms: per-head QK coupling
      matrices over L1's directions have top-5% mass >= 0.25 and cross-head cosine
      <= 0.45 (heads specialise where MLP forms do not).
QK coupling of head h at layer j: B[a,b] = |(Wq_h V_a).(Wk_h V_b)| + q2/k2 twin,
symmetrised -- the score-relevant pair coupling, weights-only. Layers 2-4, all heads."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
NH,HD,D=9,128,1152; K=48
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_signed_qk_coupling_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    V=orth(Vh[:K].T)
    out={}
    # (a) signed cosines for L5's forms (representative reader)
    j=5
    accs=[]
    for i in range(0,60,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
    Yj=torch.cat(accs)
    _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
    P=orth(Vhj[:32].T)
    mlp=m.transformer.h[j].mlp
    L=mlp.Left.weight.detach().float()@V
    R=mlp.Right.weight.detach().float()@V
    DwP=mlp.Down.weight.detach().float().T@P
    mats=[]
    for f in range(32):
        M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
        mats.append(0.5*(M+M.T))
    flat=[(M/M.norm().clamp_min(1e-12)).flatten() for M in mats]
    sc=[abs(float(flat[a]@flat[b])) for a,b in itertools.combinations(range(32),2)]
    signed_cos=sum(sc)/len(sc)
    out['signed_within_cos_L5']=signed_cos
    print(f'(a) signed within-reader cos (L5 forms): {signed_cos:.2f} '
          f'(abs version was 0.64)')
    pa=signed_cos<0.40
    # (b) QK readers, layers 2-4
    concs=[]; heads=[]
    for jj in (2,3,4):
        a=m.transformer.h[jj].attn
        def hm(W): return (W.weight.detach().float()@V).view(NH,HD,K)
        Q1,K1,Q2,K2=hm(a.c_q),hm(a.c_k),hm(a.c_q2),hm(a.c_k2)
        for h in range(NH):
            B=(torch.einsum('ea,eb->ab',Q1[h],K1[h]).abs()
               +torch.einsum('ea,eb->ab',Q2[h],K2[h]).abs())
            B=0.5*(B+B.T)
            heads.append(B/B.norm().clamp_min(1e-12))
            iu=torch.triu_indices(K,K)
            v=B[iu[0],iu[1]]
            k5=max(1,int(0.05*v.numel()))
            concs.append(float(v.topk(k5).values.sum()/v.sum().clamp_min(1e-12)))
    xc=[float((heads[a].flatten()@heads[b].flatten()))
        for a,b in itertools.combinations(range(len(heads)),2)]
    med_conc=sorted(concs)[len(concs)//2]
    mean_cos=sum(xc)/len(xc)
    out['qk_median_top5']=med_conc; out['qk_cross_cos']=mean_cos
    pb=med_conc>=0.25 and mean_cos<=0.45
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f'(b) QK heads (L2-4, 27 heads): median top-5% mass {med_conc:.2f} | '
          f'cross-head cos {mean_cos:.2f}')
    print(f"\n(a) signed cos < 0.40: {'HELD' if pa else 'FAILED'} | "
          f"(b) QK specialised: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
