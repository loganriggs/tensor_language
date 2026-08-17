"""The last empty cell: L0's MLP. Every effective-linearity and linearization
sweep started at L1+ (or L2+), but L0 is the true reset -- it writes at RMS 1436
into a stream of ~6, and the post-L0 stream IS its computation on raw
embeddings. Consistent protocol: (i) linear R^2 of L0's write from its block
input (top-16 output PCs, held out); (ii) linearization cost on fresh eval rows.

REGISTERED PREDICTIONS: (a) linearization cost >= 0.15 (front-loading extends
to the front-most layer; L1 is +0.28, L2 +0.11); (b) linear R^2 <= 0.75 (real
quadratic computation on embeddings); (c) sanity: the cost exceeds L4's +0.054."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
import bilin18_pipe_refit as PR
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l0_cell_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    tri=[];tei=[]
    for i in range(0,48,6): tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    for i in range(300,324,6): tei.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    Xt=torch.cat([a[0] for a,b in tri]); Yt=torch.cat([b[0] for a,b in tri])
    Xe=torch.cat([a[0] for a,b in tei]); Ye=torch.cat([b[0] for a,b in tei])
    Ytc=Yt-Yt.mean(0)
    _,_,Vh=torch.linalg.svd(Ytc[:20000],full_matrices=False)
    P=orth(Vh[:16].T)
    yt=Yt@P; ye=Ye@P
    Xc=Xt-Xt.mean(0); Xec=Xe-Xt.mean(0)
    ytc=yt-yt.mean(0); yec=ye-yt.mean(0)+yt.mean(0)-ye.mean(0)+ye.mean(0)-yt.mean(0)
    yec=ye-yt.mean(0)@P if False else ye-yt.mean(0)@torch.zeros(1) if False else ye-ye.mean(0)
    lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
    W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                         Xc.T@ytc/Xc.shape[0])
    r2=1-float(((yec-Xec@W)**2).mean()/(yec**2).mean())
    print(f'L0 linear R^2 (write, top-16 PCs, held out): {r2:+.3f}',flush=True)
    PR.LINS={}
    def ce():
        tot,n=0.0,0
        for i in range(384,448,4):
            b=FW[i:i+4,:257].to(DEV)
            lg,_=PR.fwd_lin(b[:,:-1].contiguous())
            c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
            tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
        return tot/n
    base=ce()
    PR.LINS={0:PR.fit_layer(0)}
    cost=ce()-base
    PR.LINS={}
    print(f'L0 linearization cost: +{cost:.4f}',flush=True)
    pa=cost>=0.15; pb=r2<=0.75; pc=cost>0.054
    out={'linear_r2':r2,'cost':cost,'pred_a':bool(pa),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f"(a) front-loading extends (cost >=0.15): {'HELD' if pa else 'FAILED'}")
    print(f"(b) real quadratic work (R^2 <=0.75): {'HELD' if pb else 'FAILED'}")
    print(f"(c) exceeds L4: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
