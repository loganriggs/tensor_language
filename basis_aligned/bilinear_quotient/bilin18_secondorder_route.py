"""Second-order operator signature: is pattern-dominance predictable after all?

§50: first-order norms failed to predict head 1's pattern-route (v-enrichment 8.1 >
qk 4.4, yet the measured route is 54% pattern / 30% value). The right object is the
RESPONSE ENERGY: injecting delta = 2 sigma * d at attn1's input changes
  values:  Delta ctx_v = pat_base @ (W_v delta) -- first order, pattern fixed;
  pattern: Delta pat = (Delta s1) s2 + s1 (Delta s2) with Delta s from W_q/W_k delta
           against the EXISTING k/q activations -- also first order in delta but
           weighted by the standing scores.
Both are computable from weights + cached activations, no interventions. REGISTERED
PREDICTION: the pattern-response energy of head 1's context change exceeds its
value-response energy (matching the measured 54/30), and the pattern:value ratio for
head 1 exceeds the median head's ratio."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs)
    _,_,Vh0=torch.linalg.svd((Y0-Y0.mean(0)).float(), full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    d=orth(Vh0[:32].T)[:,int(phi0.argmax())].float()
    s0=float((((Y0-Y0.mean(0)).float())@d).std())
    # collect attn1's actual q/k/v/pattern on base rows (reuse mediation forward)
    import bilin18_edge_mediation as EM
    rows=FW[300:312,:257].to(DEV)
    B,T=rows.shape
    x=F.rms_norm(m.transformer.wte(rows),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    st={}
    for li in range(2):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        a=blk.attn
        hcur=F.rms_norm(x,(D,))
        def qk(l):
            z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,))
            return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k1_,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k1_)/HD
        s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0)
        if li==1:
            st=dict(q=q,k=k1_,q2=q2,k2=k2,v=v,s1=s1,s2=s2,pat=pat,attn=a)
            break
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
    a=st['attn']; delta=(2*s0)*d
    def headmats(W): return W.weight.detach().float().view(NH,HD,D)
    dq=torch.einsum('hed,d->he',headmats(a.c_q),delta)/HD**0
    dk=torch.einsum('hed,d->he',headmats(a.c_k),delta)
    dq2=torch.einsum('hed,d->he',headmats(a.c_q2),delta)
    dk2=torch.einsum('hed,d->he',headmats(a.c_k2),delta)
    dq1=torch.einsum('hed,d->he',headmats(a.c_q),delta)
    dv=torch.einsum('hed,d->he',headmats(a.c_v),delta)
    out={'heads':{}}
    print(f"  {'head':>5} {'pattern-resp':>13} {'value-resp':>11} {'ratio':>7}")
    ratios=[]
    for h in range(NH):
        # value response: existing pattern moves injected value content
        Ev=float((st['pat'][:,h].abs().sum(-1)**2).mean())*float((dv[h]**2).sum())
        # pattern response: ds1 = (dq.k + q.dk)/HD etc, cross-multiplied with standing s2/s1, times existing v energy
        ds1=(torch.einsum('e,bkhe->bk',dq1[h],st['k'][:,:,h:h+1,:].transpose(1,2))
             +torch.einsum('bqhe,e->bq',st['q'][:,:,h:h+1,:].transpose(1,2)
                           .transpose(2,3).squeeze(-2)*0+st['q'][:,:,h,:],dk[h])
             .unsqueeze(-1)*0).sum()*0
        # simpler numeric estimate: ds1[q,k] = (dq.k[k] + q[q].dk)/HD
        dqk1=(torch.einsum('e,bke->bk',dq1[h],st['k'][:,:,h,:])[:,None,:]
              +torch.einsum('bqe,e->bq',st['q'][:,:,h,:],dk[h])[:,:,None])/HD
        dqk2=(torch.einsum('e,bke->bk',dq2[h],st['k2'][:,:,h,:])[:,None,:]
              +torch.einsum('bqe,e->bq',st['q2'][:,:,h,:],dk2[h])[:,:,None])/HD
        dpat=(dqk1*st['s2'][:,h]+st['s1'][:,h]*dqk2)*mask
        vE=float((st['v'][:,:,h,:]**2).sum(-1).mean())
        Ep=float((dpat**2).sum(-1).mean())*vE
        r=Ep/max(Ev,1e-12)
        ratios.append(r)
        out['heads'][h]={'pattern_resp':Ep,'value_resp':Ev,'ratio':r}
        print(f"  {h:>5} {Ep:>13.3e} {Ev:>11.3e} {r:>7.2f}",flush=True)
    med=sorted(ratios)[NH//2]
    pa=ratios[1]>1.0; pb=ratios[1]>med
    out['pred_h1_pattern_dominant']=bool(pa)
    out['pred_h1_above_median']=bool(pb)
    print(f"\n(a) head-1 pattern-resp > value-resp: {'HELD' if pa else 'FAILED'} "
          f"(ratio {ratios[1]:.2f})")
    print(f"(b) head-1 ratio above median ({med:.2f}): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_secondorder_route_results.json','w'),indent=1)
    print(f'wrote bilin18_secondorder_route_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
