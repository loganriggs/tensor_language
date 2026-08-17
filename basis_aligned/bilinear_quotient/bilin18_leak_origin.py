"""Where does the 32% interchange leak live? Registered: in the document component.

§21: transplanting z cross-document reproduces 68% of the true c0-transplant's
downstream effect; on-distribution the surrogate carries 92%. §17: the leader
coefficient's variance is 56% document identity. Hypothesis: the leak IS the document
component -- z transplanted within the same document should carry it correctly.
REGISTERED PREDICTION: same-document interchange faithfulness >= 85% (vs 68%
cross-document, 13% shuffle control)."""
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
    u=(Sih@Uw[:,ew.abs().argmax()]).float(); u=u/u.norm()
    c=torch.einsum('ni,ij,nj->n',Xh,M,Xh)
    p2=(Xh@u)**2
    co=torch.linalg.lstsq(torch.stack([p2,torch.ones_like(p2)],1),c[:,None]).solution.squeeze()
    a_s,b_s=float(co[0]),float(co[1])
    # SAME-document pairing: base = first half of long rows, source = second half
    base_rows=FW[300:324,0:257].to(DEV)
    src_rows =FW[300:324,256:513].to(DEV)
    ctx_s,xh_s=IC.ctx1_of(src_rows)
    c_src=torch.einsum('bti,ij,btj->bt',xh_s.float(),M,xh_s.float())
    z_src=(xh_s.float()@u)
    def mk_hook(cv):
        def hook(xhat,mo):
            cb=mo.float()@d0
            return mo+((cv-cb)[...,None]*d0).to(mo.dtype)
        return hook
    IC.COEFF_FN=None
    lp_base,_=IC.fwd_logits(base_rows)
    IC.COEFF_FN=mk_hook(c_src);           lp_c,_=IC.fwd_logits(base_rows)
    IC.COEFF_FN=mk_hook(a_s*z_src**2+b_s); lp_z,_=IC.fwd_logits(base_rows)
    perm=torch.randperm(c_src.shape[1])
    IC.COEFF_FN=mk_hook(c_src[:,perm]);   lp_sh,_=IC.fwd_logits(base_rows)
    IC.COEFF_FN=None
    def kl(a,b): return (a.exp()*(a-b)).sum(-1)
    eff=kl(lp_c,lp_base); mz=kl(lp_c,lp_z); ms=kl(lp_c,lp_sh)
    sel=eff>eff.flatten().kthvalue(int(0.5*eff.numel())).values
    f=1-float(mz[sel].mean()/eff[sel].mean())
    fs=1-float(ms[sel].mean()/eff[sel].mean())
    out={'faithfulness_same_doc':f,'shuffle_control':fs,
         'cross_doc_reference':0.680,'prediction_held':bool(f>=0.85)}
    print(f'same-document interchange faithfulness: {100*f:.1f}% '
          f'(cross-doc was 68.0%; shuffle control {100*fs:.1f}%)')
    print(f"registered prediction (>=85%): {'HELD' if f>=0.85 else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_leak_origin_results.json','w'),indent=1)
    print(f'wrote bilin18_leak_origin_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
