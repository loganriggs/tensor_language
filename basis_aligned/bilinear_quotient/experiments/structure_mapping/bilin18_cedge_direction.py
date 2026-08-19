"""Weight-side prediction of the coupling anisotropy (Phase-D of the refined law).

§42: on the 3->4 edge, coupling strength per unit damage varies ~5x with WHICH L3
directions are damaged. §31's coupling operator K(3->4) = C3^{1/2} G2(4) C3^{1/2} is
the weights+S candidate for that anisotropy. REGISTERED PREDICTION: damaging L3 along
K's top-8 eigendirections (mapped to output space) produces coupling constant
c = excess/(d3*d4) at least 1.5x the constant for damaging along L3's top-8 PCA
directions, with L4's damage held fixed (PCA-32 span). Control: bottom-K 8-dim span,
predicted lower c than either."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_cedge_direction_results.json')

@torch.no_grad()
def collect(li, what):
    ins,outs=[],[]
    def hook(mod,inp,o):
        if what!='out': ins.append(inp[0].detach().reshape(-1,D).float())
        if what!='in': outs.append(o.detach().reshape(-1,D).float())
    h=m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    return (torch.cat(ins) if what=='in' else torch.cat(outs))

def main():
    t0=time.time()
    base=held()
    Y3=collect(3,'out'); Yb3=Y3.mean(0); Y3c=Y3-Yb3
    C3=(Y3c.T@Y3c/Y3c.shape[0]).double()
    X4=collect(4,'in'); S4=X4.T@X4/X4.shape[0]
    mlp4=m.transformer.h[4].mlp
    L=mlp4.Left.weight.detach().float(); R=mlp4.Right.weight.detach().float()
    Dw=mlp4.Down.weight.detach().float(); DD=Dw.T@Dw
    G24=L.T@(DD*(R@S4@R.T))@L + R.T@(DD*(L@S4@L.T))@R
    ev3,U3=torch.linalg.eigh(C3); ev3=ev3.clamp_min(0)
    C3h=((U3*ev3.sqrt())@U3.T).float()
    K=C3h@G24@C3h
    evK,UK=torch.linalg.eigh(K.double())
    idxK=evK.argsort(descending=True)
    topK=orth((C3h.double()@UK[:,idxK[:8]]).float())
    botK=orth((C3h.double()@UK[:,idxK[-8:]]).float())
    _,_,Vh3=torch.linalg.svd(Y3c.float(), full_matrices=False)
    pca8=orth(Vh3[:8].T)
    Y4=collect(4,'out'); Yb4=Y4.mean(0)
    _,_,Vh4=torch.linalg.svd((Y4-Yb4).float(), full_matrices=False)
    span4=orth(Vh4[:32].T)
    def val(patches):
        for li,Q,Yb in patches: PATCH[li]=(Q,Yb@Q)
        try: return float((held()-base).mean())
        finally:
            for li,_,_ in patches: PATCH.pop(li)
    d4=val([(4,span4,Yb4)])
    out={'d4':d4,'arms':{}}
    print(f'd4 (PCA-32) = {d4:+.4f}')
    for tag,Q in (('K-top8',topK),('PCA-8',pca8),('K-bot8',botK)):
        d3=val([(3,Q,Yb3)])
        j=val([(3,Q,Yb3),(4,span4,Yb4)])
        exc=j-d3-d4
        c=exc/max(d3*d4,1e-9)
        out['arms'][tag]={'d3':d3,'joint':j,'excess':exc,'c':c}
        print(f'{tag:>7}: d3 {d3:+.4f} excess {exc:+.4f}  c={c:+.1f}',flush=True)
    cK=out['arms']['K-top8']['c']; cP=out['arms']['PCA-8']['c']
    cB=out['arms']['K-bot8']['c']
    pa=cK>1.5*cP; pb=cB<min(cK,cP)
    out['pred_main']=bool(pa); out['pred_control']=bool(pb)
    print(f"\nprediction (c_K > 1.5x c_PCA): {'HELD' if pa else 'FAILED'} "
          f"({cK:.1f} vs {cP:.1f})")
    print(f"control (K-bottom lowest): {'HELD' if pb else 'FAILED'} ({cB:.1f})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
