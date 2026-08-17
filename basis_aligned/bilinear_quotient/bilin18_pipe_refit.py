"""Sequential-refit rescue of the linear pipe. Section 104: naive joint
linearization of layers 5-17 costs +5.49 nats, 2.68x superadditive -- hypothesized
mechanism is approximation-validity drift (each stand-in fit on the intact model's
distribution, invalidated by upstream stand-ins). Fix: linearize front-to-back,
REFITTING each layer's ridge map on activations of the model with all upstream
stand-ins already installed.

REGISTERED PREDICTIONS: (a) drift is the mechanism: sequential-refit joint(5-17)
<= 1.5 nats (vs naive 5.49); (b) the L5 outlier is also drift-free pathology or
not: refit L5-alone cost <= 0.5 (its +1.51 was fit quality, recoverable with the
same data by construction -- if it stays high, L5's function is irreducibly
nonlinear); (c) the per-step CE curve is convex-ish: the last five steps add more
than the first five (drift still compounds even with refitting). Alternative to
(a): refit joint > 3 nats means tail function is irreducibly nonlinear in
composition."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_pipe_refit_results.json')
LINS={}

@torch.no_grad()
def fwd_lin(idx, want=None):
    """Forward with LINS stand-ins installed; optionally return (input, output)
    of layer `want` under the current hybrid model."""
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    cap=None
    for li in range(18):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0
        xin=x
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
        if li in LINS:
            mp=LINS[li]
            xi=xin.reshape(-1,D).float()
            mo=((xi-mp['bx'])@mp['W']+mp['by']).to(x.dtype).view_as(x)
        else:
            mo=mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
        if want is not None and li==want:
            cap=(xin.detach().reshape(-1,D).float(),
                 (mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias
                  ).detach().reshape(-1,D).float())
        x=x+mo
    logits=m.lm_head(F.rms_norm(x,(D,)))
    logits=30*torch.tanh(logits/30)
    return logits.float(), cap

@torch.no_grad()
def ce_eval():
    tot,n=0.0,0
    for i in range(300,380,4):
        b=FW[i:i+4,:257].to(DEV)
        lg,_=fwd_lin(b[:,:-1].contiguous())
        ce=F.cross_entropy(lg.view(-1,lg.size(-1)),
                           b[:,1:].reshape(-1))
        tot+=float(ce)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

@torch.no_grad()
def fit_layer(li):
    xs=[]; ys=[]
    for i in range(0,48,6):
        _,cap=fwd_lin(FW[i:i+6,:256].to(DEV), want=li)
        xs.append(cap[0]); ys.append(cap[1])
    X=torch.cat(xs); Y=torch.cat(ys)
    bx=X.mean(0); by=Y.mean(0)
    Xc=X-bx; Yc=Y-by
    lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
    W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                         Xc.T@Yc/Xc.shape[0])
    return {'W':W,'bx':bx,'by':by}

@torch.no_grad()
def main():
    global LINS
    t0=time.time()
    LINS={}
    base=ce_eval()
    print(f'base {base:.4f}',flush=True)
    # (b) refit L5-alone == naive L5 by construction (same distribution) but
    # measure to compare fit-quality story
    LINS={5:fit_layer(5)}
    l5=ce_eval()-base
    print(f'L5 alone (refit protocol): +{l5:.4f}',flush=True)
    LINS={}
    curve=[]
    for li in range(5,18):
        LINS[li]=fit_layer(li)     # fit on CURRENT hybrid (upstream already linear)
        c=ce_eval()-base
        curve.append(c)
        print(f'after L{li:2d}: cumulative +{c:.4f}',flush=True)
    joint=curve[-1]
    first5=curve[4]; last5=joint-curve[7]
    pa=joint<=1.5; pb=l5<=0.5; pc=last5>first5
    out={'base':base,'l5_alone':l5,'curve':curve,'joint_refit':joint,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) drift rescued (<=1.5): {'HELD' if pa else 'FAILED'} (+{joint:.3f})")
    print(f"(b) L5 recoverable (<=0.5): {'HELD' if pb else 'FAILED'}")
    print(f"(c) still compounding (last5 > first5): {'HELD' if pc else 'FAILED'}")
    if joint>3: print('alternative: tail function irreducibly nonlinear in composition')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
