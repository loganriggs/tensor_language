"""HILLCLIMB strategy 2 (user-directed): circuit-informed rank allocation vs
blind variance. At equal parameters (rank-8 stand-ins, layers 5-16, sequential
refit), two arms: (A) SVD truncation of the refit map (variance-dominant
directions -- the blind baseline); (B) reader-aligned truncation: project the
refit map's OUTPUT onto the NEXT layer's measured input watch-list (its
input-mode Gram top-8) -- "write what your reader reads," the relay/watch-list
circuit knowledge as an allocation rule.

REGISTERED PREDICTIONS: (a) individually, reader-aligned beats variance at
>= 7/12 layers by >= 0.01; (b) jointly, reader-aligned beats variance by
>= 0.05; alternative: variance wins -- the diffuseness verdict extends to
allocation, and blind search beats this circuit rule (also a benchmark
result)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV, orth, m
import bilin18_pipe_refit as PR
D=1152
LAYERS=list(range(5,17))
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_circuit_assign_results.json')

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
def reader_grams():
    """Next-layer MLP input-mode Gram top-8 per writer layer (reader = li+1)."""
    ins={li+1:[] for li in LAYERS}
    hs=[]
    for lj in ins:
        def mk(lj=lj):
            return lambda mod,inp: ins[lj].append(
                inp[0].detach().reshape(-1,D).float()) or None
        hs.append(m.transformer.h[lj].mlp.register_forward_pre_hook(mk()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    G8={}
    for lj,acc in ins.items():
        X=torch.cat(acc); S=X.T@X/X.shape[0]
        mlp=m.transformer.h[lj].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float(); DD=Dw.T@Dw
        G=L.T@(DD*(R@S@R.T))@L + R.T@(DD*(L@S@L.T))@R
        ev,U=torch.linalg.eigh(G.double())
        G8[lj]=orth(U[:,ev.argsort(descending=True)[:8]].float())
    return G8

@torch.no_grad()
def main():
    t0=time.time()
    PR.LINS={}
    base=ce()
    G8=reader_grams()
    def trunc_var(mp):
        U,S,Vh=torch.linalg.svd(mp['W'])
        return {'W':U[:,:8]@torch.diag(S[:8])@Vh[:8],'bx':mp['bx'],'by':mp['by']}
    def trunc_reader(mp,li):
        P=G8[li+1]@G8[li+1].T
        Wr=mp['W']@P            # output restricted to reader's watch-list
        U,S,Vh=torch.linalg.svd(Wr)
        return {'W':U[:,:8]@torch.diag(S[:8])@Vh[:8],'bx':mp['bx'],
                'by':mp['by']}
    res={'A':{},'B':{}}
    for arm,tr in (('A',lambda mp,li:trunc_var(mp)),
                   ('B',trunc_reader)):
        maps={}
        for li in LAYERS:
            PR.LINS={lj:maps[lj] for lj in maps}
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
            maps[li]=tr({'W':W,'bx':bx,'by':by},li)
        PR.LINS=maps
        joint=ce()-base
        res[arm]['joint']=joint
        # individual costs
        for li in LAYERS:
            PR.LINS={li:maps[li]}
            res[arm][li]=ce()-base
        PR.LINS={}
        print(f'arm {arm}: joint +{joint:.3f} | indiv '
              +' '.join(f'{res[arm][li]:.3f}' for li in LAYERS),flush=True)
    wins=sum(1 for li in LAYERS if res['B'][li]<=res['A'][li]-0.01)
    pa=wins>=7
    pb=(res['A']['joint']-res['B']['joint'])>=0.05
    out={'A':{str(k):v for k,v in res['A'].items()},
         'B':{str(k):v for k,v in res['B'].items()},
         'wins_B':wins,'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\n(a) reader-aligned wins >=7/12 individually: "
          f"{'HELD' if pa else 'FAILED'} ({wins}/12)")
    print(f"(b) jointly by >=0.05: {'HELD' if pb else 'FAILED'} "
          f"(A +{res['A']['joint']:.3f} vs B +{res['B']['joint']:.3f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
