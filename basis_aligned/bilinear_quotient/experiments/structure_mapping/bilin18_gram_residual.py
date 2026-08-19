"""v2 in the GRAM basis. Section 112: the residual quadratic is not compact in
input PCA (19% at L17, 0% at L9). The layer's input-mode Lambda-Gram
G = L^T(D^TD o RSR^T)L + R^T(D^TD o LSL^T)R names the input directions the
quadratic actually uses, from weights. Same fit, Gram eigenbasis instead of PCA.
REGISTERED: (a) Gram-64 captures >= 2x PCA-64 at L17 (>= 0.38); (b) >= 0.25
absolute at L9; (c) monotone in k. If (a) and (b) fail, the residual is diffuse
in every natural basis and the compressed-quadratic direction is closed.

Original: compressed quadratic on the residual of the
linear fit (cf. Belrose et al.'s closed-form polynomial approximations of MLPs).
The MLPs here are exactly quadratic, so a full quadratic is exact by
construction; the question is COMPACTNESS: how few input directions does the
nonlinear residual need? For L17 (linear R^2 0.95 -- is the 5% skin a compact
quadratic?) and L9 (middle, linear R^2 0.60 -- is the big residual also
compact?): fit ridge linear from block input, then fit quadratic features
z_i z_j of the top-k input PCs (k = 16, 32, 64) to the residual, held-out R^2
on the residual's top-16 output PCs.

REGISTERED PREDICTIONS: (a) L17's residual is compact: quadratic on top-64 input
PCs captures >= 50% of held-out residual variance; (b) L9's is not: top-64
captures <= 30% (middle nonlinearity is high-rank -- matches its incompressible
reputation); (c) monotone in k for both."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_effective_linearity import fwd_all
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_gram_residual_results.json')

def quad_feats(Z):
    n,k=Z.shape
    idx=torch.triu_indices(k,k)
    return Z[:,idx[0]]*Z[:,idx[1]]

@torch.no_grad()
def main():
    t0=time.time()
    tri=[];tei=[]
    for i in range(0,48,6): tri.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    for i in range(300,324,6): tei.append(fwd_all(FW[i:i+6,:257].to(DEV)))
    out={}
    for li in (17,9):
        Xt=torch.cat([a[li] for a,b in tri]); Yt=torch.cat([b[li] for a,b in tri])
        Xe=torch.cat([a[li] for a,b in tei]); Ye=torch.cat([b[li] for a,b in tei])
        bx=Xt.mean(0); by=Yt.mean(0)
        Xc=Xt-bx; Yc=Yt-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        Rt=Yc-Xc@W; Re=(Ye-by)-(Xe-bx)@W
        _,_,Vh=torch.linalg.svd(Rt[:20000],full_matrices=False)
        P=orth(Vh[:16].T)
        rt=Rt@P; re=Re@P
        S=Xc[:20000].T@Xc[:20000]/20000
        mlp=m.transformer.h[li].mlp
        Lw=mlp.Left.weight.detach().float(); Rw=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float(); DD=Dw.T@Dw
        G=Lw.T@(DD*(Rw@S@Rw.T))@Lw + Rw.T@(DD*(Lw@S@Lw.T))@Rw
        ev,U_=torch.linalg.eigh(G.double())
        Ug=U_[:,ev.argsort(descending=True)].float()
        row={}
        for k in (16,32,64):
            U=orth(Ug[:,:k])
            Zt=quad_feats(Xc@U); Ze=quad_feats((Xe-bx)@U)
            Zt=Zt-Zt.mean(0); Ze=Ze-Ze.mean(0)
            lam2=1e-2*float((Zt**2).mean())*Zt.shape[1]/Zt.shape[0]
            A=torch.linalg.solve(Zt.T@Zt/Zt.shape[0]+
                                 lam2*torch.eye(Zt.shape[1],device=DEV),
                                 Zt.T@rt/Zt.shape[0])
            pred=Ze@A
            r2=1-float(((re-pred)**2).mean()/(re**2).mean())
            row[k]=r2
            print(f'L{li} quad-on-top-{k:2d}: residual R^2 {r2:+.3f}',flush=True)
        out[str(li)]=row
    pa=out['17'][64]>=0.38
    pb=out['9'][64]>=0.25
    pc=all(out[l][16]<=out[l][32]<=out[l][64] for l in ('17','9'))
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f"\n(a) Gram-64 doubles PCA at L17 (>=0.38): {'HELD' if pa else 'FAILED'}")
    print(f"(b) L9 Gram-64 >= 0.25: {'HELD' if pb else 'FAILED'}")
    print(f"(c) monotone in k: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
