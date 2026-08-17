"""Does the second-order response energy predict edge STRENGTH from weights?

§51: the punctuation axis routes into c0 at +1.04 sigma; the number axis at +0.23.
The second-order signature gave head 1 a towering pattern-response for the punctuation
axis. If the calculus is right, the response energy computed for each steered signal
should ORDER the measured edge strengths. Test five L0 signals: causal directions
#1 (punct), #2 (number), #3, plus two random directions. For each: (i) weights+cache
max-over-heads total response energy E(d); (ii) measured Delta c1 under +2 sigma
steering. REGISTERED PREDICTIONS: (a) Spearman(E, |Delta c1|) >= 0.8 over the five
signals; (b) the punctuation axis tops both lists."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import bilin18_edge_mediation as EM
import bilin18_secondorder_route as SR
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152

def spearman(a,b):
    ra=a.argsort().argsort().double(); rb=b.argsort().argsort().double()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    return float((ra@rb)/(ra.norm()*rb.norm()).clamp_min(1e-30))

@torch.no_grad()
def response_energy(d, s0):
    """max-over-heads total (pattern+value) response energy for injecting 2*s0*d."""
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
            dqv=torch.einsum('hed,d->he',hm(a.c_q),delta)
            dkv=torch.einsum('hed,d->he',hm(a.c_k),delta)
            dq2v=torch.einsum('hed,d->he',hm(a.c_q2),delta)
            dk2v=torch.einsum('hed,d->he',hm(a.c_k2),delta)
            dvv=torch.einsum('hed,d->he',hm(a.c_v),delta)
            best=0.0
            for h in range(NH):
                dqk1=(torch.einsum('e,bke->bk',dqv[h],k1_[:,:,h,:])[:,None,:]
                      +torch.einsum('bqe,e->bq',q[:,:,h,:],dkv[h])[:,:,None])/HD
                dqk2=(torch.einsum('e,bke->bk',dq2v[h],k2[:,:,h,:])[:,None,:]
                      +torch.einsum('bqe,e->bq',q2[:,:,h,:],dk2v[h])[:,:,None])/HD
                dpat=(dqk1*s2[:,h]+s1[:,h]*dqk2)*mask
                vE=float((v[:,:,h,:]**2).sum(-1).mean())
                Ep=float((dpat**2).sum(-1).mean())*vE
                Ev=float((pat[:,h].abs().sum(-1)**2).mean())*float((dvv[h]**2).sum())
                best=max(best,Ep+Ev)
            return best
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
    g=torch.Generator(device=DEV).manual_seed(0)
    sigs={}
    for tag,dv in (('punct(#1)',Q0[:,int(order[0])]),('number(#2)',Q0[:,int(order[1])]),
                   ('#3',Q0[:,int(order[2])]),
                   ('rand-a',torch.randn(D,device=DEV,generator=g)),
                   ('rand-b',torch.randn(D,device=DEV,generator=g))):
        dv=dv/dv.norm(); sigs[tag]=(dv.float(),float((Y0c@dv).std()))
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs)
    _,_,Vh1=torch.linalg.svd((Y1-Y1.mean(0)).float(), full_matrices=False)
    EM.RUN['M1']=form_for_direction(m.transformer.h[1].mlp,
                                    orth(Vh1[:32].T)[:,0].float()).float()
    rows=FW[300:324,:257].to(DEV)
    EM.CFG.update({'steer':None,'freeze1':None,'freeze0':None})
    c1b,caps=EM.run(rows); s1=float(c1b.std())
    EM.CFG['cap1']=caps[1]; EM.CFG['cap0']=caps[0]
    out={'signals':{}}
    Es=[]; Ds=[]
    print(f"  {'signal':>10} {'resp energy':>12} {'|Delta c1|':>11}")
    for tag,(dv,s0) in sigs.items():
        E=response_energy(dv,s0)
        EM.CFG.update({'steer':(dv,2*s0),'freeze1':None,'freeze0':None})
        c1p,_=EM.run(rows)
        EM.CFG.update({'steer':None,'freeze1':None,'freeze0':None})
        dc=abs(float((c1p-c1b).mean())/s1)
        out['signals'][tag]={'resp_energy':E,'abs_dc1':dc}
        Es.append(E); Ds.append(dc)
        print(f"  {tag:>10} {E:>12.3e} {dc:>11.3f}",flush=True)
    rho=spearman(torch.tensor(Es),torch.tensor(Ds))
    top=max(out['signals'],key=lambda k:out['signals'][k]['resp_energy'])
    topd=max(out['signals'],key=lambda k:out['signals'][k]['abs_dc1'])
    pa=rho>=0.8; pb=top=='punct(#1)' and topd=='punct(#1)'
    out['spearman']=rho; out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    print(f"\nSpearman(E, |dc1|) = {rho:+.2f} -> (a) {'HELD' if pa else 'FAILED'}")
    print(f"punct tops both: {'HELD' if pb else 'FAILED'} "
          f"(E-top {top}, dc1-top {topd})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                       'bilin18_edge_strength_results.json','w'),indent=1)
    print(f'wrote bilin18_edge_strength_results.json ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
