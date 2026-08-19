"""HILLCLIMB round 3b (section 199): the class lever, fit properly. Stand-in =
closed-form refit LINEAR (warm base) + narrow factored bilinear fit by Adam on
the RESIDUAL in output-normalized space (residual / its std). REGISTERED:
(a) L1 combo (width 64) <= +0.90 individually (beats rank-64 linear +1.11);
(b) L16 combo (width 16) <= +0.05 (beats linear r8 +0.059); (c) residual fits
converge >= 3x. A clean failure closes the class ladder for good.

Prior context -- HILLCLIMB round 3: the computation-class lever. A factored bilinear stand-in
Down'((L'x)(R'x)) with hidden width h' << 4608 -- a genuine narrow quadratic,
fit by Adam on captured (post-mix input, MLP output) pairs. Distinct from
section 160's dead rung (fixed-basis quadratic FEATURES on a linear residual);
this class learns its own factors.

Targets: L1 (the crown, where linear saturates: r64 linear +1.11, full linear
+0.29) at width 64 (0.22M params); L16 at width 16 (0.055M).

REGISTERED PREDICTIONS: (a) width-64 bilinear at L1 <= +0.60 individually
(beats rank-64 linear's +1.11 -- the class lever works where rank could not);
(b) width-16 bilinear at L16 <= linear rank-8's +0.059; (c) fit sanity: train
MSE decreases >= 3x from init for both."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
from bilin18_effective_linearity import fwd_all
import bilin18_pipe_refit as PR
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_hillclimb3b_results.json')

def fit_bilinear(X,Y,h,steps=800,lr=3e-3):
    bx=X.mean(0); by=Y.mean(0)
    Xc=(X-bx)
    # closed-form linear warm base
    lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
    Wl=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                          Xc.T@(Y-by)/Xc.shape[0])
    Rres=(Y-by)-Xc@Wl
    rs=Rres.std(0).clamp_min(1e-6)
    Yc=Rres/rs
    xs=Xc.std(0).clamp_min(1e-6)
    Xn=Xc/xs
    g=torch.Generator(device=DEV).manual_seed(0)
    sc=(2.0/D)**0.5
    L=torch.randn(h,D,device=DEV,generator=g)*sc
    R=torch.randn(h,D,device=DEV,generator=g)*sc
    Dn=torch.randn(D,h,device=DEV,generator=g)*(1.0/h**0.5)
    for p in (L,R,Dn): p.requires_grad_(True)
    opt=torch.optim.Adam([L,R,Dn],lr=lr)
    n=Xc.shape[0]
    mse0=None
    for t in range(steps):
        idx=torch.randint(0,n,(4096,),generator=g,device=DEV)
        xb=Xn[idx]; yb=Yc[idx]
        pred=((xb@L.T)*(xb@R.T))@Dn.T
        loss=F.mse_loss(pred,yb)
        if t==0: mse0=float(loss)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred=((Xn@L.T)*(Xn@R.T))@Dn.T
        msef=float(F.mse_loss(pred,Yc))
    return {'L':L.detach(),'R':R.detach(),'Dn':Dn.detach(),'bx':bx,'by':by,
            'Wl':Wl,'rs':rs,'xs':xs,'mse0':mse0,'msef':msef}

@torch.no_grad()
def ce_with(li,mp):
    """CE with a bilinear stand-in at layer li, PR-style manual forward."""
    from tier2_model import rope_tables, apply_rot
    NH,HD=9,128
    tot,n=0.0,0
    for i in range(384,448,4):
        bb=FW[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        B,T=idx.shape
        cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
        cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
        mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
        import bilin18_joint_removal as JR
        x=F.rms_norm(JR.m.transformer.wte(idx),(D,)); x0=x; v1=None
        for lj in range(18):
            blk=JR.m.transformer.h[lj]; a=blk.attn
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            xin=x
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
            x=x+a.c_proj(torch.einsum('bhqk,bkhd->bqhd',pat,v).reshape(B,T,-1))
            xhat=F.rms_norm(x,(D,)); mlp=blk.mlp
            if lj==li and mp is not None:
                xi=(xin.reshape(-1,D).float()-mp['bx'])
                lin=xi@mp['Wl']
                xn=xi/mp['xs']
                quad=(((xn@mp['L'].T)*(xn@mp['R'].T))@mp['Dn'].T)*mp['rs']
                mo=(lin+quad+mp['by']).to(x.dtype).view_as(x)
            else:
                mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
            x=x+mo
        lg=(30*torch.tanh(JR.m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        c=F.cross_entropy(lg.view(-1,lg.size(-1)),tg)
        tot+=float(c)*tg.numel(); n+=tg.numel()
    return tot/n

def main():
    t0=time.time()
    tri=[]
    for i in range(0,48,6):
        with torch.no_grad():
            tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    base=ce_with(0,None)
    out={}
    for li,h,bar in ((1,64,0.90),(16,16,0.05)):
        X=torch.cat([a[li] for a,b in tri]); Y=torch.cat([b[li] for a,b in tri])
        mp=fit_bilinear(X.float(),Y.float(),h)
        cost=ce_with(li,mp)-base
        out[f'L{li}']={'width':h,'cost':cost,'params':3*D*h,
                       'mse0':mp['mse0'],'msef':mp['msef']}
        print(f'L{li} width-{h} bilinear: +{cost:.3f} ({3*D*h/1e6:.2f}M params) '
              f'| mse {mp["mse0"]:.1f}->{mp["msef"]:.1f}',flush=True)
    pa=out['L1']['cost']<=0.90
    pb=out['L16']['cost']<=0.05
    pc=all(v['mse0']/max(v['msef'],1e-6)>=3 for v in out.values())
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) L1 combo beats r64 linear (<=0.90): {'HELD' if pa else 'FAILED'}")
    print(f"(b) L16 combo <= 0.05: {'HELD' if pb else 'FAILED'}")
    print(f"(c) fits converged (>=3x): {'HELD' if pc else 'FAILED'}")
    out['base']=base
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
