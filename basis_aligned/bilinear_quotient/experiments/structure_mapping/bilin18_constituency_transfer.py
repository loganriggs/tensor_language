"""Do the word constituencies transfer in-distribution?

§66 established causal individuation on pile (shifted corpus). The robustness check:
same 5 words, same 12 coefficients, constituency profiles measured on the saved
fineweb sample. REGISTERED PREDICTIONS: (a) mean residual pairwise correlation stays
<= 0.1 on fineweb; (b) per-word profiles correlate across corpora at mean >= 0.5
(the constituencies are the same constituencies, not corpus artifacts)."""
import json, sys, time, torch, itertools, math
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_gradient_steering import run_forward, collect_basis, orth
from bilin18_joint_removal import fwd, m, FW, DEV
from bilin18_identifiable import form_for_direction
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_constituency_transfer_results.json')

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
    fine=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'fineweb_eval_tokens.pt')[:12,:257].to(DEV)
    pile=FW[300:312,:257].to(DEV)
    def profiles_on(rows_in):
        with torch.no_grad():
            base,_=run_forward(rows_in,targets)
            sig={i: float(base[i].std()) for i in base}
            prof=[]
            for w in steers:
                p,_=run_forward(rows_in,targets,steer=(w,mag))
                prof.append([abs(float((p[i]-base[i]).mean()))/sig[i]
                             for i in range(len(targets))])
        return prof
    pf=profiles_on(fine); pp=profiles_on(pile)
    n=len(pf[0])
    def resid(prof):
        norm=[sum(prof[k][i] for k in range(5))/5 for i in range(n)]
        return [[prof[k][i]/max(norm[i],1e-6) for i in range(n)] for k in range(5)]
    Rf=resid(pf); Rp=resid(pp)
    def corr(a,b):
        ma=sum(a)/len(a); mb=sum(b)/len(b)
        num=sum((x-ma)*(y-mb) for x,y in zip(a,b))
        da=math.sqrt(sum((x-ma)**2 for x in a)); db=math.sqrt(sum((y-mb)**2 for y in b))
        return num/max(da*db,1e-9)
    wc=[corr(Rf[a],Rf[b]) for a,b in itertools.combinations(range(5),2)]
    meanc=sum(wc)/len(wc)
    xcorr=[corr(Rf[k],Rp[k]) for k in range(5)]
    mx=sum(xcorr)/len(xcorr)
    out={'fineweb_mean_residual_corr':meanc,'cross_corpus_per_word':xcorr,
         'mean_cross_corpus':mx}
    print(f'fineweb: mean residual pairwise corr {meanc:+.2f}')
    print(f'per-word cross-corpus profile corr: '
          f'{[round(c,2) for c in xcorr]} (mean {mx:+.2f})')
    pa=meanc<=0.1; pb=mx>=0.5
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"(a) individuated in-distribution: {'HELD' if pa else 'FAILED'} | "
          f"(b) same constituencies across corpora: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
