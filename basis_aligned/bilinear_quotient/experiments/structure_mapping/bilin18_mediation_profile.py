"""Can second-order per-head response energies predict the mediation PROFILE?

§53: first-order enrichment fails at carrier prediction (wrong head on the #3 edge,
wrong presumption a carrier exists). The §51 second-order signature recovered route
character; the hypothesis is that its per-head energies also recover the mediation
DISTRIBUTION -- concentrated for punct (head 1, 96%), spread for #3 (max 32%).
Compute per-head total response energies E_h(d) for both steered directions and
compare with the measured kill profiles (bilin18_mediation_heads / blind_routing2
results). REGISTERED PREDICTIONS:
  (a) Spearman(E_h, kills_h) >= 0.6 on the punct edge;
  (b) Spearman(E_h, kills_h) >= 0.6 on the #3 edge;
  (c) concentration transfers: the Gini-style top-1 share of E_h is higher for punct
      than for #3 (predicting concentrated vs spread mediation)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152

def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return float((ra@rb)/(ra.norm()*rb.norm()).clamp_min(1e-30))

@torch.no_grad()
def per_head_E(d, s0):
    rows=FW[300:312,:257].to(DEV)
    B,T=rows.shape
    x=F.rms_norm(m.transformer.wte(rows),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
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
            delta=(2*s0)*d
            def hm(W): return W.weight.detach().float().view(NH,HD,D)
            dq=torch.einsum('hed,d->he',hm(a.c_q),delta)
            dk=torch.einsum('hed,d->he',hm(a.c_k),delta)
            dq2=torch.einsum('hed,d->he',hm(a.c_q2),delta)
            dk2=torch.einsum('hed,d->he',hm(a.c_k2),delta)
            dv=torch.einsum('hed,d->he',hm(a.c_v),delta)
            Es=[]
            for h in range(NH):
                dqk1=(torch.einsum('e,bke->bk',dq[h],k1_[:,:,h,:])[:,None,:]
                      +torch.einsum('bqe,e->bq',q[:,:,h,:],dk[h])[:,:,None])/HD
                dqk2=(torch.einsum('e,bke->bk',dq2[h],k2[:,:,h,:])[:,None,:]
                      +torch.einsum('bqe,e->bq',q2[:,:,h,:],dk2[h])[:,:,None])/HD
                dpat=(dqk1*s2[:,h]+s1[:,h]*dqk2)*mask
                vE=float((v[:,:,h,:]**2).sum(-1).mean())
                Ep=float((dpat**2).sum(-1).mean())*vE
                Ev=float((pat[:,h].abs().sum(-1)**2).mean())*float((dv[h]**2).sum())
                Es.append(Ep+Ev)
            return torch.tensor(Es)
        x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
        xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
        x=x+mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias

def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=0, acc=acc); accs.append(acc[0])
    Y0=torch.cat(accs); Y0c=(Y0-Y0.mean(0)).float()
    _,_,Vh0=torch.linalg.svd(Y0c, full_matrices=False)
    phi0=torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                    'bilin18_layer0_battery_results_phi.pt').mean(1)
    order=phi0.argsort(descending=True)
    Q0=orth(Vh0[:32].T)
    kp=json.load(open('bilin18_mediation_heads_results.json'))['kills']
    k3=json.load(open('bilin18_blind_routing2_results.json'))['kills']
    out={'edges':{}}
    for tag,idx,kills in (('punct',int(order[0]),kp),('#3',int(order[2]),k3)):
        d=Q0[:,idx].float(); s0=float((Y0c@d).std())
        E=per_head_E(d,s0)
        r=spearman(E,torch.tensor(kills))
        top1=float(E.max()/E.sum())
        out['edges'][tag]={'E':[float(v) for v in E],'kills':kills,
                           'spearman':r,'top1_share':top1}
        print(f'{tag}: per-head E {[f"{v:.1e}" for v in E]}')
        print(f'      kills      {[round(k,2) for k in kills]}')
        print(f'      Spearman {r:+.2f} | top-1 energy share {top1:.2f}',flush=True)
    pa=out['edges']['punct']['spearman']>=0.6
    pb=out['edges']['#3']['spearman']>=0.6
    pc=out['edges']['punct']['top1_share']>out['edges']['#3']['top1_share']
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) punct rho>=0.6: {'HELD' if pa else 'FAILED'} | "
          f"(b) #3 rho>=0.6: {'HELD' if pb else 'FAILED'} | "
          f"(c) concentration transfers: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_mediation_profile_results.json','w'),indent=1)
    print(f'wrote bilin18_mediation_profile_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
