"""Locating section 204's architecture modulation: swiglu18 licenses only one
constant where bilin18 licensed four. Is its tail merely not
CONSTANT-replaceable (mean output insufficient, but rank-4 linear fine), or
less replaceable at every rung? Individual rank-4 refit-on-clean stand-in
costs for its tail (layers 5-15).

REGISTERED PREDICTIONS: (a) >= 6 of 11 layers drop to <= 0.05 at rank-4 (the
difference from bilin18 is only the zeroth rung -- swiglu's tail needs a
little input-dependence but is equally replaceable); (b) alternative: <= 3
qualify -> genuinely less replaceable at all rungs, and the architecture
modulation is deep."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'swiglu18_rank4_results.json')
LAYERS=list(range(5,16))

@torch.no_grad()
def main():
    t0=time.time()
    m2,_=load_elriggs('swiglu18', device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    caps={li:{'x':[],'y':[]} for li in LAYERS}
    def fwd(idx, assign, collect=False):
        x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
        for lj,blk in enumerate(m2.transformer.h):
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            xin=x
            x1,v1=blk.attn(F.rms_norm(x,(D,)),v1)
            x=x+x1
            sp=assign.get(lj)
            if sp is None:
                mo=blk.mlp(F.rms_norm(x,(D,)))
                if collect and lj in caps:
                    caps[lj]['x'].append(xin.detach().reshape(-1,D).float())
                    caps[lj]['y'].append(mo.detach().reshape(-1,D).float())
            else:
                xi=xin.reshape(-1,D).float()
                mo=((xi-sp['bx'])@sp['W']+sp['by']).to(x.dtype).view_as(x)
            x=x+mo
        return (30*torch.tanh(m2.lm_head(F.rms_norm(x,(D,)))/30)).float()
    def ce(assign):
        tot,n=0.0,0
        for i in range(384,448,4):
            b=FW[i:i+4,:257].to(DEV)
            lg=fwd(b[:,:-1].contiguous(),assign)
            c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
            tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    for i in range(0,48,6):
        fwd(FW[i:i+6,:256].to(DEV),{},collect=True)
    base=ce({})
    ok=0; costs={}
    for li in LAYERS:
        X=torch.cat(caps[li]['x']); Y=torch.cat(caps[li]['y'])
        bx=X.mean(0); by=Y.mean(0)
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        U,S,Vh=torch.linalg.svd(W)
        W4=U[:,:4]@torch.diag(S[:4])@Vh[:4]
        c=ce({li:{'W':W4,'bx':bx,'by':by}})-base
        costs[li]=c
        if c<=0.05: ok+=1
        print(f'L{li:2d}: rank-4 +{c:.4f}',flush=True)
    pa=ok>=6; pb=ok<=3
    out={'costs':{str(k):v for k,v in costs.items()},'n_ok':ok,
         'pred_a':bool(pa),'pred_b_alt':bool(pb),'base':base}
    print(f"\n(a) rank-4 rescues >=6/11: {'HELD' if pa else 'FAILED'} ({ok}/11)")
    if pb: print('alternative: genuinely less replaceable at all rungs')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
