"""Operating-point scaling of error amplification, at four times the sample.

§41(b) landed at rho +0.29 against a 0.3 bar with n=12 positions -- unresolved. Same
single-position design at n=48. REGISTERED PREDICTION: rho(mismatch per unit
coefficient error, base coefficient magnitude) > 0.3 at n=48; the quadratic readout
says sensitivity grows with operating point."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import bilin18_interchange as IC
from bilin18_joint_removal import fwd, orth, m, FW, DEV, LAYER
from bilin18_identifiable import form_for_direction
from bilin18_source_folding import forward_tracked

def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return float((ra@rb)/(ra.norm()*rb.norm()).clamp_min(1e-30))

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
    base_rows=FW[300:324,:257].to(DEV); src_rows=FW[400:424,:257].to(DEV)
    _,xh_s=IC.ctx1_of(src_rows); _,xh_b=IC.ctx1_of(base_rows)
    c_src=torch.einsum('bti,ij,btj->bt',xh_s.float(),M,xh_s.float())
    c_base=torch.einsum('bti,ij,btj->bt',xh_b.float(),M,xh_b.float())
    chat_src=a_s*(xh_s.float()@u)**2+b_s
    def mk(q,cv):
        def hook(xhat,mo):
            cb=mo.float()@d0
            delta=torch.zeros_like(cb); delta[:,q]=cv[:,q]-cb[:,q]
            return mo+(delta[...,None]*d0).to(mo.dtype)
        return hook
    def kl(a,b): return (a.exp()*(a-b)).sum(-1)
    IC.COEFF_FN=None; lp_b,_=IC.fwd_logits(base_rows)
    rows=[]
    for q in range(16,256,5):
        IC.COEFF_FN=mk(q,c_src);   lp_c,_=IC.fwd_logits(base_rows)
        IC.COEFF_FN=mk(q,chat_src);lp_z,_=IC.fwd_logits(base_rows)
        IC.COEFF_FN=None
        rows.append({'q':q,
                     'mismatch':float(kl(lp_c,lp_z)[:,q:].sum(1).mean()),
                     'coeff_err':float(((chat_src[:,q]-c_src[:,q])**2).mean()),
                     'base_coeff_sq':float((c_base[:,q]**2).mean())})
    t=torch.tensor
    mi=t([r['mismatch'] for r in rows]); ce=t([r['coeff_err'] for r in rows])
    cm=t([r['base_coeff_sq'] for r in rows])
    r_a=spearman(mi,ce)
    r_b=spearman(mi/ce.clamp_min(1e-12),cm)
    out={'n_positions':len(rows),'spearman_mismatch_coefferr':r_a,
         'spearman_amplification_vs_magnitude':r_b,
         'pred_held':bool(r_b>0.3),'rows':rows}
    print(f'n={len(rows)}: rho(mismatch, coeff err) = {r_a:+.2f} | '
          f'rho(amplification, operating point) = {r_b:+.2f}')
    print(f"registered prediction (rho > 0.3): {'HELD' if r_b>0.3 else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_opscaling_results.json','w'),indent=1)
    print(f'wrote bilin18_opscaling_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
