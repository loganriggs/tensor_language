"""PORTABILITY DEMO: the closed hillclimb recipe applied end-to-end to bilin12
(the 'throw GPUs at a new model' workflow): (1) rank-0 scan licenses constants;
(2) rank-4 sequential-refit linear elsewhere (tail layers 3-10; front 0-2 and
final layer 11 stay real); (3) joint scoring; (4) the refit-vs-naive lever
check.

REGISTERED PREDICTIONS: (a) >= 2 tail layers license constants (individual
rank-0 cost <= 0.05); (b) the recipe's joint cost <= +1.2 at <= 0.1M stand-in
params; (c) the refit lever transfers: sequential refit beats naive same-
architecture assignment by >= 15%."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
from tier2_model import load_elriggs
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_recipe_results.json')
LAYERS=list(range(3,11))

@torch.no_grad()
def main():
    t0=time.time()
    m2,_=load_elriggs('bilin12', device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def fwd(idx, assign, want=None):
        x=F.rms_norm(m2.transformer.wte(idx),(D,)); x0=x; v1=None
        cap=None
        for lj,blk in enumerate(m2.transformer.h):
            x=blk.lambdas[0]*x+blk.lambdas[1]*x0
            xin=x
            x1,v1=blk.attn(F.rms_norm(x,(D,)),v1)
            x=x+x1
            real=None
            if want is not None and lj==want:
                real=blk.mlp(F.rms_norm(x,(D,)))
                cap=(xin.detach().reshape(-1,D).float(),
                     real.detach().reshape(-1,D).float())
            sp=assign.get(lj)
            if sp is None:
                mo=real if real is not None else blk.mlp(F.rms_norm(x,(D,)))
            elif sp['kind']=='const':
                mo=sp['by'][None,None,:].to(x.dtype).expand_as(x)
            else:
                xi=xin.reshape(-1,D).float()
                mo=((xi-sp['bx'])@sp['W']+sp['by']).to(x.dtype).view_as(x)
            x=x+mo
        lg=m2.lm_head(F.rms_norm(x,(D,)))
        return (30*torch.tanh(lg/30)).float(), cap
    def ce(assign):
        tot,n=0.0,0
        for i in range(384,448,4):
            b=FW[i:i+4,:257].to(DEV)
            lg,_=fwd(b[:,:-1].contiguous(),assign)
            c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
            tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    def capture(assign,li):
        xs=[];ys=[]
        for i in range(0,48,6):
            _,cap=fwd(FW[i:i+6,:256].to(DEV),assign,want=li)
            xs.append(cap[0]); ys.append(cap[1])
        return torch.cat(xs),torch.cat(ys)
    def fit(X,Y,r):
        bx=X.mean(0); by=Y.mean(0)
        if r==0: return {'kind':'const','by':by}
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        U,S,Vh=torch.linalg.svd(W)
        return {'kind':'lin','W':U[:,:r]@torch.diag(S[:r])@Vh[:r],'bx':bx,'by':by}
    base=ce({})
    # step 1: rank-0 scan
    const_ok=[]
    for li in LAYERS:
        X,Y=capture({},li)
        c0=ce({li:{'kind':'const','by':Y.mean(0)}})-base
        print(f'L{li} rank-0 alone: +{c0:.4f}',flush=True)
        if c0<=0.05: const_ok.append(li)
    print(f'constants licensed: {const_ok}',flush=True)
    # step 2: sequential refit build
    assign={}
    for li in LAYERS:
        X,Y=capture(assign,li)
        assign[li]=fit(X,Y,0 if li in const_ok else 4)
    c_refit=ce(assign)-base
    # naive: fit all on clean model
    naive={}
    for li in LAYERS:
        X,Y=capture({},li)
        naive[li]=fit(X,Y,0 if li in const_ok else 4)
    c_naive=ce(naive)-base
    params=sum(2*D*4 for li in LAYERS if li not in const_ok)
    print(f'recipe joint: refit +{c_refit:.3f} | naive +{c_naive:.3f} | '
          f'{params/1e6:.2f}M',flush=True)
    pa=len(const_ok)>=2
    pb=c_refit<=1.2 and params<=0.1e6
    pc=(c_naive-c_refit)/max(c_naive,1e-6)>=0.15
    out={'base':base,'const_ok':const_ok,'refit':c_refit,'naive':c_naive,
         'params':params,'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) >=2 constants licensed: {'HELD' if pa else 'FAILED'}")
    print(f"(b) joint <= +1.2 at <= 0.1M: {'HELD' if pb else 'FAILED'}")
    print(f"(c) refit lever transfers (>=15%): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
