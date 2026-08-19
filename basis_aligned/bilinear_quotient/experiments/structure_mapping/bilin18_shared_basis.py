"""HILLCLIMB round 2: the cross-layer sharing lever. One shared basis pair
(U: 1152x64 output, V: 1152x64 input) for ALL 12 tail stand-ins, with per-layer
64x64 cores: W'_li = U (U^T W_li V) V^T. Params = 2*1152*64 + 12*64^2 = 0.20M
for effective rank-64 everywhere (unshared r64 costs 1.18M).

Bases from SVD of the stacked sequential-refit maps; cores are the projected
refit maps; the whole assembly re-scored jointly (and sequentially refit once
in the shared basis).

REGISTERED PREDICTIONS: (a) shared-basis r64 at ~0.20M beats the unshared
uniform-r16 point (+1.66 at 0.29M) by >= 0.05 -- sharing is a real lever;
(b) it beats the unshared r64 cost only mildly worse (within 0.10 of +1.54 at
1.18M) -- 83% param saving nearly free; (c) held-out-layer test: bases fit on
the six even tail layers only, applied to all twelve, within 0.08 of the
all-layer bases (the tail's stand-ins genuinely share structure)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth
import bilin18_pipe_refit as PR
D=1152; R=64
LAYERS=list(range(5,17))
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_shared_basis_results.json')

@torch.no_grad()
def ce():
    tot,n=0.0,0
    for i in range(384,448,4):
        b=FW[i:i+4,:257].to(DEV)
        lg,_=PR.fwd_lin(b[:,:-1].contiguous())
        c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
        tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

@torch.no_grad()
def seq_refit_maps(project=None):
    maps={}
    for li in LAYERS:
        PR.LINS=dict(maps)
        xs=[];ys=[]
        for i in range(0,48,6):
            _,cap=PR.fwd_lin(FW[i:i+6,:256].to(DEV), want=li)
            xs.append(cap[0]); ys.append(cap[1])
        X=torch.cat(xs); Y=torch.cat(ys)
        bx=X.mean(0); by=Y.mean(0)
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        if project is not None:
            U,V=project
            W=U@(U.T@W@V)@V.T
        maps[li]={'W':W,'bx':bx,'by':by}
    PR.LINS={}
    return maps

def bases_from(maps, layers):
    Wo=torch.cat([maps[li]['W'] for li in layers],dim=1)   # output side stack
    Uo,_,_=torch.linalg.svd(Wo@Wo.T)
    U=orth(Uo[:,:R])
    Wi=torch.cat([maps[li]['W'].T for li in layers],dim=1)
    Ui,_,_=torch.linalg.svd(Wi@Wi.T)
    V=orth(Ui[:,:R])
    return U,V

@torch.no_grad()
def main():
    t0=time.time()
    PR.LINS={}
    base=ce()
    maps=seq_refit_maps()
    U,V=bases_from(maps,LAYERS)
    shared=seq_refit_maps(project=(U,V))
    PR.LINS=shared
    c_shared=ce()-base
    PR.LINS={}
    params=2*D*R+len(LAYERS)*R*R
    print(f'shared-basis r{R}: +{c_shared:.3f} at {params/1e6:.2f}M',flush=True)
    Ue,Ve=bases_from(maps,LAYERS[::2])   # even-index tail layers only
    shared_h=seq_refit_maps(project=(Ue,Ve))
    PR.LINS=shared_h
    c_held=ce()-base
    PR.LINS={}
    print(f'held-out bases (6 layers): +{c_held:.3f}',flush=True)
    pa=c_shared<=1.66-0.05
    pb=c_shared<=1.54+0.10
    pc=(c_held-c_shared)<=0.08
    out={'base':base,'shared':c_shared,'params':params,'heldout':c_held,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) beats uniform-r16 by >=0.05 at 0.2M: {'HELD' if pa else 'FAILED'}")
    print(f"(b) within 0.10 of unshared r64: {'HELD' if pb else 'FAILED'}")
    print(f"(c) bases generalize across layers (<=0.08): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
