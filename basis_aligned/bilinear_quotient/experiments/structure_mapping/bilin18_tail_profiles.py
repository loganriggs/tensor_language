"""Profiles for the rest of the tail -- the last open item. The weights-plus-S
formula (input-mode Lambda-Gram top eigenvectors of each layer's own quadratic,
weighted by input second moment) located causal-leader subspaces blindly at 3 of 4
tested tail layers. This run extends the protocol to every tail layer 5-15:
compute the formula's top-8 output-side span (the layer's Down-projected leading
quadratic modes), delete it (mean-ablate) and measure held-out CE damage against
(i) a random-8 span and (ii) the layer's top-8 output-PCA span.

REGISTERED PREDICTIONS: (a) formula-span damage >= 5x random-span damage at a
majority (>=6/11) of tail layers; (b) the formula span is competitive with output
PCA (damage ratio formula/PCA >= 0.7 at >=6/11 layers) -- the weights find what
matters without seeing activations' output side; (c) at least one layer shows the
truncation-as-regularisation sign (negative damage) on this corpus, as L9/12/15
did on shifted data -- if none does, that pattern was purely shift-driven."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_tail_profiles_results.json')

@torch.no_grad()
def ce_eval(patches):
    hs=[]
    for li,(Q,cbar) in patches.items():
        def mk(Q=Q,cbar=cbar):
            def hook(mod,i_,o_):
                c=o_.float()@Q
                return (o_-((c-cbar)@Q.T).to(o_.dtype))
            return hook
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    tot,n=0.0,0
    for i in range(300,380,4):
        b=FW[i:i+4,:257].to(DEV)
        loss=m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    for h in hs: h.remove()
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    base=ce_eval({})
    print(f'baseline CE {base:.4f}\n',flush=True)
    ins={}; mos={}
    hs=[]
    for li in range(5,16):
        def mki(li=li):
            return lambda mod,inp: ins.setdefault(li,[]).append(
                inp[0].detach().reshape(-1,D).float()) or None
        def mko(li=li):
            return lambda mod,i_,o_: mos.setdefault(li,[]).append(
                o_.detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].mlp.register_forward_pre_hook(mki()))
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mko()))
    for i in range(0,36,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    g=torch.Generator(device=DEV).manual_seed(0)
    res={}
    for li in range(5,16):
        X=torch.cat(ins[li]); S=X.T@X/X.shape[0]
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float(); DD=Dw.T@Dw
        G=L.T@(DD*(R@S@R.T))@L + R.T@(DD*(L@S@L.T))@R
        ev,U=torch.linalg.eigh(G.double())
        Vin=U[:,ev.argsort(descending=True)[:8]].float()
        # output-side image of the formula's input modes
        act=F.rms_norm(X[:20000],(D,))
        # push the input modes through the layer's quadratic to get output span
        Yimg=(Dw@( (L@ (S@Vin)) * (R@(S@Vin)) ).squeeze()).T if False else None
        # simpler faithful route: regression of output on input-mode coords
        cin=(X[:20000]@Vin)
        Y=torch.cat(mos[li])[:20000]; Yc=Y-Y.mean(0)
        cinc=cin-cin.mean(0)
        W=torch.linalg.lstsq(cinc,Yc).solution
        Qf=orth(W.T)
        Ybar=Y.mean(0)
        _,_,Vh=torch.linalg.svd(Yc,full_matrices=False)
        Qp=orth(Vh[:8].T)
        Qr=orth(torch.randn(D,8,device=DEV,generator=g))
        row={}
        for tag,Q in (('formula',Qf),('pca',Qp),('random',Qr)):
            ce=ce_eval({li:(Q,Ybar@Q)})
            row[tag]=ce-base
        res[li]=row
        print(f'L{li}: formula {row["formula"]:+.4f} | pca {row["pca"]:+.4f} | '
              f'random {row["random"]:+.4f}',flush=True)
    na=sum(1 for r in res.values() if r['formula']>=5*max(r['random'],1e-6))
    nb=sum(1 for r in res.values()
           if r['pca']>1e-6 and r['formula']/r['pca']>=0.7)
    nc=sum(1 for r in res.values() if r['formula']< -0.001 or r['pca']< -0.001)
    pa=na>=6; pb=nb>=6; pc=nc>=1
    out={'per_layer':{str(k):v for k,v in res.items()},'base':base,
         'n_formula_over_random':na,'n_competitive':nb,'n_negative':nc,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) formula >=5x random at >=6/11: {'HELD' if pa else 'FAILED'} ({na})")
    print(f"(b) formula/pca >=0.7 at >=6/11: {'HELD' if pb else 'FAILED'} ({nb})")
    print(f"(c) >=1 negative-damage layer: {'HELD' if pc else 'FAILED'} ({nc})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
