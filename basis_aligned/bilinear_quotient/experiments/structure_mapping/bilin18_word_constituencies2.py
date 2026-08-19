"""Confirmatory rerun of word individuation, envelope-normalised and pre-registered.

§65's raw test was confounded by the movability envelope; its post-hoc normalisation
showed anti-correlated residual constituencies. This registers that statistic IN
ADVANCE on a larger design: 5 vocabulary words + 2 random controls, 12 reader
coefficients, profiles normalised per-coefficient by the mean response across all
seven steers. REGISTERED PREDICTIONS:
  (a) mean pairwise residual correlation among the 5 words <= 0.1;
  (b) no word pair exceeds +0.5;
  (c) every word remains causally live (max raw move >= 0.3 sigma)."""
import json, sys, time, torch, itertools, math
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_gradient_steering import run_forward, collect_basis, orth
from bilin18_joint_removal import fwd, m, FW, DEV
from bilin18_identifiable import form_for_direction
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_word_constituencies2_results.json')

def main():
    t0=time.time()
    Y1c=collect_basis()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    s_out=float(Y1c.norm(dim=1).mean())/K**0.5
    mag=2*s_out*K**0.5*0.2
    rows=[]; targets=[]
    for j in READERS:
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            rows.append((0.5*(M+M.T)).flatten())
        for f in (0,3):
            targets.append((j,form_for_direction(mlp,P[:,f].float()).float()))
    X=torch.stack(rows)
    _,sv,W=torch.linalg.svd(X, full_matrices=False)
    steers=[]
    for p_ in range(5):
        Pm=0.5*(W[p_].view(K,K)+W[p_].view(K,K).T)
        evp,Up=torch.linalg.eigh(Pm.double())
        w=(V@Up[:,evp.abs().argmax()].float()); steers.append(w/w.norm())
    g=torch.Generator(device=DEV).manual_seed(7)
    for _ in range(2):
        r=torch.randn(D,device=DEV,generator=g); steers.append(r/r.norm())
    rows_in=FW[300:312,:257].to(DEV)
    with torch.no_grad():
        base,_=run_forward(rows_in,targets)
        sig={i: float(base[i].std()) for i in base}
        profiles=[]
        for w in steers:
            p,_=run_forward(rows_in,targets,steer=(w,mag))
            profiles.append([abs(float((p[i]-base[i]).mean()))/sig[i]
                             for i in range(len(targets))])
    n=len(profiles[0])
    norm=[sum(profiles[k][i] for k in range(7))/7 for i in range(n)]
    Rn=[[profiles[k][i]/max(norm[i],1e-6) for i in range(n)] for k in range(7)]
    def corr(a,b):
        ma=sum(a)/len(a); mb=sum(b)/len(b)
        num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
        da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
        return num/max(da*db,1e-9)
    wc=[corr(Rn[a],Rn[b]) for a,b in itertools.combinations(range(5),2)]
    maxmoves=[max(profiles[k]) for k in range(5)]
    meanc=sum(wc)/len(wc)
    out={'profiles':[[round(v,3) for v in pr] for pr in profiles],
         'residual_pairwise':wc,'mean_residual_corr':meanc,
         'max_moves':maxmoves}
    for k in range(5):
        print(f'word #{k+1}: max move {maxmoves[k]:.2f}s')
    print(f'\nmean residual pairwise correlation (5 words): {meanc:+.2f} | '
          f'max pair {max(wc):+.2f}')
    pa=meanc<=0.1; pb=max(wc)<0.5; pc=all(v>=0.3 for v in maxmoves)
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"(a) mean <= 0.1: {'HELD' if pa else 'FAILED'} | "
          f"(b) max < 0.5: {'HELD' if pb else 'FAILED'} | "
          f"(c) all live: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
