"""CHAMPION assembly: the full hillclimb recipe on the standard scope (layers
5-16 replaced; front and L17 real). Class selection from the program's maps:
constants at L8/L9/L14/L15 (section 155/157 licensed), warm-started width-16
factored bilinear at L5 and L16 (the two costly tail layers), rank-4 refit
linear elsewhere. Built sequentially front-to-back; scored jointly.
Total stand-in params ~0.18M.

Reference points to beat: every linear-class point measured -- best linear is
uniform-r64 refit at +1.541 (1.18M); shared basis +1.734 (0.20M); uniform-r16
+1.660 (0.29M).

REGISTERED PREDICTIONS: (a) champion joint <= +1.45 at <= 0.2M params (beats
the entire measured linear curve at ~1/6 the best linear point's params);
(b) beats the matched-param shared-basis point by >= 0.25; (c) ablation: the
two bilinear rungs matter -- replacing them with rank-4 linear raises the
joint by >= 0.05."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
import bilin18_pipe_refit as PR
from bilin18_hillclimb3b import fit_bilinear
from tier2_model import rope_tables, apply_rot
import bilin18_joint_removal as JR
NH,HD,D=9,128,1152
LAYERS=list(range(5,17))
CONST={8,9,14,15}
BILIN={5:16,16:16}
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_champion_results.json')

@torch.no_grad()
def fwd_champ(idx, assign, want=None):
    B,T=idx.shape
    cos,sin=rope_tables(T,HD,DEV,torch.float32,'bf16')
    cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    x=F.rms_norm(JR.m.transformer.wte(idx),(D,)); x0=x; v1=None
    cap=None
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
        real_mo=None
        if want is not None and lj==want:
            real_mo=(mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias)
            cap=(xin.detach().reshape(-1,D).float(),
                 real_mo.detach().reshape(-1,D).float())
        sp=assign.get(lj)
        if sp is None:
            mo=real_mo if real_mo is not None else \
               (mlp.Down(mlp.Left(xhat)*mlp.Right(xhat))+mlp.Down_bias)
        elif sp['kind']=='const':
            mo=sp['by'][None,None,:].to(x.dtype).expand_as(x)
        elif sp['kind']=='lin':
            xi=xin.reshape(-1,D).float()
            mo=((xi-sp['bx'])@sp['W']+sp['by']).to(x.dtype).view_as(x)
        else:  # bilinear combo
            xi=(xin.reshape(-1,D).float()-sp['bx'])
            lin=xi@sp['Wl']
            xn=xi/sp['xs']
            quad=(((xn@sp['L'].T)*(xn@sp['R'].T))@sp['Dn'].T)*sp['rs']
            mo=(lin+quad+sp['by']).to(x.dtype).view_as(x)
        x=x+mo
    lg=JR.m.lm_head(F.rms_norm(x,(D,)))
    return (30*torch.tanh(lg/30)).float(), cap

@torch.no_grad()
def ce(assign):
    tot,n=0.0,0
    for i in range(384,448,4):
        b=FW[i:i+4,:257].to(DEV)
        lg,_=fwd_champ(b[:,:-1].contiguous(),assign)
        c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
        tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

def capture(assign, li):
    xs=[];ys=[]
    with torch.no_grad():
        for i in range(0,48,6):
            _,cap=fwd_champ(FW[i:i+6,:256].to(DEV),assign,want=li)
            xs.append(cap[0]); ys.append(cap[1])
    return torch.cat(xs),torch.cat(ys)

def fit_lin(X,Y,r):
    bx=X.mean(0); by=Y.mean(0)
    Xc=X-bx; Yc=Y-by
    lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
    W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                         Xc.T@Yc/Xc.shape[0])
    U,S,Vh=torch.linalg.svd(W)
    return {'kind':'lin','W':U[:,:r]@torch.diag(S[:r])@Vh[:r],'bx':bx,'by':by}

def main():
    t0=time.time()
    base=ce({})
    def build(use_bilin):
        assign={}
        for li in LAYERS:
            X,Y=capture(assign,li)
            if li in CONST:
                assign[li]={'kind':'const','by':Y.mean(0)}
            elif use_bilin and li in BILIN:
                mp=fit_bilinear(X.float(),Y.float(),BILIN[li])
                mp['kind']='bl'
                assign[li]=mp
            else:
                assign[li]=fit_lin(X.float(),Y.float(),4)
        return assign
    champ=build(True)
    c_champ=ce(champ)-base
    params=sum({'const':0,'lin':2*D*4}.get(sp['kind'],
               2*D*4+3*D*BILIN.get(li,16)) if sp['kind']!='bl'
               else D*D*0+3*D*BILIN[li]+D*D  # Wl is full-rank! count it
               for li,sp in champ.items())
    # honest params: bl rung = full linear Wl (D*D) + 3*D*h -- recount
    params=0
    for li,sp in champ.items():
        if sp['kind']=='const': params+=0
        elif sp['kind']=='lin': params+=2*D*4
        else: params+=D*D+3*D*BILIN[li]
    lin_only=build(False)
    c_lin=ce(lin_only)-base
    print(f'champion: +{c_champ:.3f} at {params/1e6:.2f}M | '
          f'bilinear-rungs->r4-linear: +{c_lin:.3f}',flush=True)
    pa=c_champ<=1.45 and params<=0.2e6
    pb=(1.734-c_champ)>=0.25
    pc=(c_lin-c_champ)>=0.05
    out={'base':base,'champion':c_champ,'params':params,'lin_only':c_lin,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    if params>0.2e6:
        print(f'NOTE: honest param count {params/1e6:.2f}M exceeds 0.2M -- the '
              f'bilinear combo carries a full-rank warm linear (D^2); '
              f'(a) scored against the honest count.')
    print(f"(a) <= +1.45 at <= 0.2M: {'HELD' if pa else 'FAILED'}")
    print(f"(b) beats shared-basis by >= 0.25: {'HELD' if pb else 'FAILED'}")
    print(f"(c) bilinear rungs matter (>= 0.05): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
