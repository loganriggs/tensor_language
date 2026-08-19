"""The ladder's upper rung: does a compact QUADRATIC stand-in beat linear where
linear saturates? Section 113 says the nonlinear residues are diffuse in every
natural basis, predicting no. Stand-in for L16: linear rank-32 refit PLUS a
quadratic correction on the top-32 refit input directions (features z_i z_j,
528 dims, ridge to the residual), vs plain linear rank-32 refit. Individual
install (L16 only), fresh eval rows.

REGISTERED PREDICTIONS (skeptical): (a) the quadratic correction buys < 0.05
nats over rank-32 linear at L16; (b) same at L9 (< 0.02); alternative: >= 0.1
would reopen compact quadratics at stand-in level and matter for the benchmark
ladder."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_quadratic_rung_results.json')
STAND={}

@torch.no_grad()
def fwd_stand(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    for li in range(18):
        blk=m.transformer.h[li]; a=blk.attn
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
        if li in STAND:
            sd=STAND[li]
            xi=xin.reshape(-1,D).float()
            mo=(xi-sd['bx'])@sd['W']+sd['by']
            if 'A' in sd:
                Z=(xi-sd['bx'])@sd['U']
                idxs=sd['tri']
                Q2=Z[:,idxs[0]]*Z[:,idxs[1]]
                mo=mo+(Q2-sd['qm'])@sd['A']
            mo=mo.to(x.dtype).view_as(x)
        else:
            mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        x=x+mo
    lg=m.lm_head(F.rms_norm(x,(D,)))
    return (30*torch.tanh(lg/30)).float()

@torch.no_grad()
def ce():
    tot,n=0.0,0
    for i in range(384,448,4):
        b=FW[i:i+4,:257].to(DEV)
        lg=fwd_stand(b[:,:-1].contiguous())
        c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
        tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

@torch.no_grad()
def main():
    global STAND
    t0=time.time()
    tri_=[]
    for i in range(0,48,6): tri_.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    out={}
    STAND={}
    base=ce()
    for li in (16,9):
        X=torch.cat([a[li] for a,b in tri_]); Y=torch.cat([b[li] for a,b in tri_])
        bx=X.mean(0); by=Y.mean(0)
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        U_,S_,Vh_=torch.linalg.svd(W)
        W32=U_[:,:32]@torch.diag(S_[:32])@Vh_[:32]
        STAND={li:{'W':W32,'bx':bx,'by':by}}
        lin=ce()-base
        # quadratic correction on top-32 input dirs of W
        Uin=orth(U_[:,:32])
        Z=Xc@Uin
        idxs=torch.triu_indices(32,32)
        Q2=Z[:,idxs[0]]*Z[:,idxs[1]]
        qm=Q2.mean(0)
        R=Yc-Xc@W32
        Q2c=Q2-qm
        lam2=1e-2*float((Q2c**2).mean())*Q2c.shape[1]/Q2c.shape[0]
        A=torch.linalg.solve(Q2c.T@Q2c/Q2c.shape[0]+
                             lam2*torch.eye(Q2c.shape[1],device=DEV),
                             Q2c.T@R/Q2c.shape[0])
        STAND={li:{'W':W32,'bx':bx,'by':by,'U':Uin,'A':A,'qm':qm,'tri':idxs}}
        quad=ce()-base
        STAND={}
        out[str(li)]={'linear32':lin,'quad32':quad,'gain':lin-quad}
        print(f'L{li}: linear-32 +{lin:.4f} | +quad-32 +{quad:.4f} | '
              f'gain {lin-quad:+.4f}',flush=True)
    pa=out['16']['gain']<0.05; pb=out['9']['gain']<0.02
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['base']=base
    print(f"\n(a) L16 quad gain < 0.05: {'HELD' if pa else 'FAILED'}")
    print(f"(b) L9 quad gain < 0.02: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
