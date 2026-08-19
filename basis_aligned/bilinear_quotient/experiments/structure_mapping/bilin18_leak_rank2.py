"""Second leak hypothesis: the interchange leak is the form's non-rank-1 content.

§40's same-document test refuted the document-mixture hypothesis (60.8% faithful,
no better than the 68.0% cross-document). The remaining candidate is intrinsic: c0's
rank-2 whitened truncation repaired ~100% of deletion damage where rank-1 repaired 92%
(§19), so the second eigendirection carries real function. REGISTERED PREDICTION:
transplanting BOTH coefficients (z1, z2 -> c0_hat = a1 z1^2 + a2 z2^2 + b, refit)
lifts interchange faithfulness to >= 85% on the §21 cross-document pairing."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import bilin18_interchange as IC
from bilin18_joint_removal import fwd, orth, m, FW, DEV, LAYER
from bilin18_identifiable import form_for_direction
from bilin18_source_folding import forward_tracked

def main():
    t0=time.time()
    Y=IC.collect_out(FW[0:300,:513])
    _,_,Vh=torch.linalg.svd((Y-Y.mean(0)).float(), full_matrices=False)
    Q=orth(Vh[:32].T); d0=Q[:,0].float()
    mlp1=m.transformer.h[LAYER].mlp
    M=form_for_direction(mlp1,d0).float()
    Xl=[]
    for i in range(0,96,6):
        _,xh,_=forward_tracked(FW[i:i+6,:513].to(DEV)); Xl.append(xh)
    Xh=torch.cat(Xl)
    S=(Xh.T@Xh/Xh.shape[0]).double(); ev,U=torch.linalg.eigh(S)
    kd=ev>1e-8*ev.max()
    Sih=(U[:,kd]*ev[kd].rsqrt())@U[:,kd].T; Shh=(U[:,kd]*ev[kd].sqrt())@U[:,kd].T
    Mw=Shh@M.double()@Shh; ew,Uw=torch.linalg.eigh(Mw)
    idx=ew.abs().argsort(descending=True)[:2]
    U2=(Sih@Uw[:,idx]).float()
    u1=U2[:,0]/U2[:,0].norm(); u2=U2[:,1]/U2[:,1].norm()
    c=torch.einsum('ni,ij,nj->n',Xh,M,Xh)
    f1=(Xh@u1)**2; f2=(Xh@u2)**2
    A=torch.stack([f1,f2,torch.ones_like(f1)],1)
    co=torch.linalg.lstsq(A,c[:,None]).solution.squeeze()
    a1,a2,b_=float(co[0]),float(co[1]),float(co[2])
    r2=1-float(((a1*f1+a2*f2+b_-c)**2).mean()/c.var())
    print(f'rank-2 coefficient fit R^2 on fit rows: {r2:.3f}')
    out={'fit_r2':r2,'pairings':{}}
    for tag,(b0,b1_,s0,s1_) in (('cross-doc',(300,324,400,424)),
                                ('same-doc',(300,324,None,None))):
        if tag=='cross-doc':
            base_rows=FW[b0:b1_,:257].to(DEV); src_rows=FW[s0:s1_,:257].to(DEV)
        else:
            base_rows=FW[b0:b1_,0:257].to(DEV); src_rows=FW[b0:b1_,256:513].to(DEV)
        ctx_s,xh_s=IC.ctx1_of(src_rows)
        c_src=torch.einsum('bti,ij,btj->bt',xh_s.float(),M,xh_s.float())
        chat_src=a1*(xh_s.float()@u1)**2+a2*(xh_s.float()@u2)**2+b_
        def mk(cv):
            def hook(xhat,mo):
                cb=mo.float()@d0
                return mo+((cv-cb)[...,None]*d0).to(mo.dtype)
            return hook
        IC.COEFF_FN=None; lp_b,_=IC.fwd_logits(base_rows)
        IC.COEFF_FN=mk(c_src); lp_c,_=IC.fwd_logits(base_rows)
        IC.COEFF_FN=mk(chat_src); lp_z,_=IC.fwd_logits(base_rows)
        IC.COEFF_FN=None
        def kl(a,b): return (a.exp()*(a-b)).sum(-1)
        eff=kl(lp_c,lp_b); mz=kl(lp_c,lp_z)
        sel=eff>eff.flatten().kthvalue(int(0.5*eff.numel())).values
        f=1-float(mz[sel].mean()/eff[sel].mean())
        out['pairings'][tag]=f
        print(f'{tag:>10}: rank-2 transplant faithfulness {100*f:.1f}%  '
              f'(rank-1 was {"68.0" if tag=="cross-doc" else "60.8"}%)',flush=True)
    held=out['pairings']['cross-doc']>=0.85
    out['prediction_held']=bool(held)
    print(f"\nregistered prediction (cross-doc >= 85%): {'HELD' if held else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_leak_rank2_results.json','w'),indent=1)
    print(f'wrote bilin18_leak_rank2_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
