"""The leak's accounting: is the interchange mismatch position-heavy?

§40 eliminated document identity and the form's second direction; the surviving
hypothesis is that the downstream quadratic stack amplifies the few positions where
the surrogate's transplant error is large. REGISTERED PREDICTIONS:
  (a) the top 5% of positions by mismatch KL carry > 60% of the total mismatch;
  (b) per-position mismatch KL correlates with the squared coefficient error
      (chat_src - c_src)^2 at Spearman rho > 0.4 -- i.e. the KL heaviness IS the
      coefficient-error heaviness amplified, not some third variable.
Control: the same top-5% share computed for the c-patch's own effect KL (if effect KL
is equally heavy-tailed, heaviness is generic to the model, not to the leak)."""
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
    ctx_s,xh_s=IC.ctx1_of(src_rows)
    c_src=torch.einsum('bti,ij,btj->bt',xh_s.float(),M,xh_s.float())
    chat_src=a_s*(xh_s.float()@u)**2+b_s
    coeff_err=(chat_src-c_src)**2
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
    mism=kl(lp_c,lp_z).flatten(); eff=kl(lp_c,lp_b).flatten()
    def top_share(v,frac=0.05):
        k=max(1,int(frac*v.numel()))
        return float(v.topk(k).values.sum()/v.sum().clamp_min(1e-30))
    sh_m=top_share(mism); sh_e=top_share(eff)
    rho=spearman(mism.cpu(), coeff_err.flatten().cpu())
    out={'top5_share_mismatch':sh_m,'top5_share_effect_control':sh_e,
         'spearman_mismatch_vs_coeff_err':rho,
         'pred_a_held':bool(sh_m>0.60),'pred_b_held':bool(rho>0.4)}
    print(f'top-5% of positions carry {100*sh_m:.0f}% of the mismatch KL '
          f'(control: {100*sh_e:.0f}% of the effect KL)')
    print(f'Spearman(mismatch KL, squared coefficient error) = {rho:+.3f}')
    print(f"(a) top-5% > 60%: {'HELD' if out['pred_a_held'] else 'FAILED'}  |  "
          f"(b) rho > 0.4: {'HELD' if out['pred_b_held'] else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_leak_accounting_results.json','w'),indent=1)
    print(f'wrote bilin18_leak_accounting_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
