"""Is the tail effectively LINEAR? Section 100: L16's bus coordinates are linear
functions of its input stream at R^2 0.979, though the MLP is quadratic. If tail
MLPs generally operate in a near-linear regime, the whole tail phenomenology --
unaimed writes (92), dilution routing (93), shallow compressibility, small
deletion effects (96) -- reduces to one statement: the tail is a big linear
filter, and genuine quadratic computation lives at the front.

Per layer 1-17: linear ridge from the layer's input stream (pre-attention residual)
to its FULL MLP write, held-out R^2 averaged over the top-16 output-PCA coords.
REGISTERED PREDICTIONS: (a) tail layers 9-16 all have R^2 >= 0.8; (b) front layers
1-3 are the least linear, all <= tail minimum minus 0.1 (real quadratic work is at
the front); (c) monotone-ish trend: Spearman(layer index, R^2) >= 0.6."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_effective_linearity_results.json')

@torch.no_grad()
def fwd_all(idx):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    ins={}; mos={}
    for li in range(18):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        ins[li]=x.detach().reshape(-1,D).float()
        a=blk.attn; hcur=F.rms_norm(x,(D,))
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
        mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        mos[li]=mo.detach().reshape(-1,D).float()
        x=x+mo
    return ins,mos

@torch.no_grad()
def main():
    t0=time.time()
    tri,trm=[],[]
    tei,tem=[],[]
    for i in range(0,48,6):
        a_,b_=fwd_all(FW[i:i+6,:257].to(DEV)); tri.append(a_); trm.append(b_)
    for i in range(300,324,6):
        a_,b_=fwd_all(FW[i:i+6,:257].to(DEV)); tei.append(a_); tem.append(b_)
    r2s={}
    for li in range(1,18):
        Xt=torch.cat([d[li] for d in tri]); Yt=torch.cat([d[li] for d in trm])
        Xe=torch.cat([d[li] for d in tei]); Ye=torch.cat([d[li] for d in tem])
        Ytc=Yt-Yt.mean(0)
        _,_,Vh=torch.linalg.svd(Ytc[:20000],full_matrices=False)
        P=orth(Vh[:16].T)
        yt=Yt@P; ye=Ye@P
        Xt=Xt-Xt.mean(0); Xe=Xe-Xe.mean(0)
        ytc=yt-yt.mean(0); yec=ye-ye.mean(0)
        lam=1e-2*float((Xt**2).mean())*Xt.shape[1]/Xt.shape[0]
        W=torch.linalg.solve(Xt.T@Xt/Xt.shape[0]+lam*torch.eye(D,device=DEV),
                             Xt.T@ytc/Xt.shape[0])
        r2=1-float(((yec-Xe@W)**2).mean()/(yec**2).mean())
        r2s[li]=r2
        print(f'L{li:2d}: linear R^2 {r2:+.3f}',flush=True)
    tail=[r2s[li] for li in range(9,17)]
    front=[r2s[li] for li in range(1,4)]
    idxs=list(r2s.keys()); vals=[r2s[k] for k in idxs]
    ri=torch.tensor(idxs,dtype=torch.float); rv=torch.tensor(vals)
    ra=ri.argsort().argsort().float(); rb=rv.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    sp=float((ra*rb).mean())
    pa=all(r>=0.8 for r in tail)
    pb=all(f<=min(tail)-0.1 for f in front)
    pc=sp>=0.6
    out={'r2':{str(k):v for k,v in r2s.items()},'spearman_depth':sp,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) tail 9-16 all >=0.8: {'HELD' if pa else 'FAILED'}")
    print(f"(b) front 1-3 least linear: {'HELD' if pb else 'FAILED'}")
    print(f"(c) linearity grows with depth (Spearman>=0.6): {'HELD' if pc else 'FAILED'} ({sp:+.2f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
