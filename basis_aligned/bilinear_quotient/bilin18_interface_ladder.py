"""Interface-size ladder on the L5->L6 cargo edge (benchmark instrument demo):
how many interface dimensions does the edge need? Delete L5's write within the
top-k dims of L6's attention watch-list, k in (2,4,8,16,32), plus random-k
controls. REGISTERED: (a) the effect saturates by k=8 (within 20% of k=32's);
(b) random-k stays <= 0.005 at every k; (c) k=2 already carries >= 40% of k=32
(the interface is genuinely small).

Prior context -- causal validation of the watch-lists. Sections 130-132 established, by subspace
geometry, that each layer's attention and MLP maintain separate reading
institutions. Causal test: delete the component of layer li's MLP write that lies
in the NEXT layer's attention watch-list (top-8 of stacked score filters) versus
in the next layer's MLP watch-list (input-mode Gram top-8), for li in (5,9,13).

REGISTERED PREDICTIONS: (a) per li, deleting within the MLP watch-list costs more
than within the attention watch-list at >= 2/3 layers (MLP reading dominates CE
-- the functional-vocabulary consumers); (b) both institutional deletions exceed
a random-8 deletion at >= 2/3 layers (the watch-lists are causally real, not
just geometric); null (c): random-8 <= 0.005 mean."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_interface_ladder_results.json')

@torch.no_grad()
def ce_eval(li, Q, cbar):
    hs=[]
    if Q is not None:
        def hook(mod,i_,o_):
            c=o_.float()@Q
            return (o_-((c-cbar)@Q.T).to(o_.dtype))
        hs.append(m.transformer.h[li].mlp.register_forward_hook(hook))
    tot,n=0.0,0
    for i in range(300,364,4):
        b=FW[i:i+4,:257].to(DEV)
        loss=m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        ntok=(b.shape[1]-1)*b.shape[0]
        tot+=float(loss)*ntok; n+=ntok
    for h in hs: h.remove()
    return tot/n

@torch.no_grad()
def main():
    t0=time.time()
    base=ce_eval(0,None,None)
    print(f'base {base:.4f}\n',flush=True)
    # stream stats entering each reader layer (for Ch and for Gram's S)
    caps={}; mlpin={}
    hs=[]
    for lj in (6,10,14):
        def mka(lj=lj):
            return lambda mod,inp: caps.setdefault(lj,[]).append(
                F.rms_norm(inp[0].detach().reshape(-1,D).float(),(D,))) or None
        def mkm(lj=lj):
            return lambda mod,inp: mlpin.setdefault(lj,[]).append(
                inp[0].detach().reshape(-1,D).float()) or None
        hs.append(m.transformer.h[lj].attn.register_forward_pre_hook(mka()))
        hs.append(m.transformer.h[lj].mlp.register_forward_pre_hook(mkm()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    g=torch.Generator(device=DEV).manual_seed(0)
    res={}; wins_a=0; wins_b=0; rnds=[]
    for li,lj in ((5,6),):
        X=torch.cat(caps[lj]); Xc=X-X.mean(0)
        C=Xc.T@Xc/Xc.shape[0]
        ev,U=torch.linalg.eigh(C.double())
        Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
        a=m.transformer.h[lj].attn
        mats=[]
        for h in range(NH):
            for wq,wk in ((a.c_q,a.c_k),(a.c_q2,a.c_k2)):
                Wq=wq.weight.detach().float().view(NH,HD,D)[h]
                Wk=wk.weight.detach().float().view(NH,HD,D)[h]
                K=Ch@Wq.T@Wk@Ch
                Uk,S,Vk=torch.linalg.svd(K)
                mats.append(torch.cat([Uk[:,:4],Vk[:4].T],dim=1))
        Sf=torch.cat(mats,dim=1)
        Ua,_,_=torch.linalg.svd(Sf@Sf.T)
        A8=orth(Ua[:,:8])
        Xm=torch.cat(mlpin[lj]); S=Xm.T@Xm/Xm.shape[0]
        mlp=m.transformer.h[lj].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float(); DD=Dw.T@Dw
        G=L.T@(DD*(R@S@R.T))@L + R.T@(DD*(L@S@L.T))@R
        evg,Ug=torch.linalg.eigh(G.double())
        G8=orth(Ug[:,evg.argsort(descending=True)[:8]].float())
        # full watch-list ordering: use top-32 of the stacked filters
        Ua,_,_=torch.linalg.svd(Sf@Sf.T)
        A32=Ua[:,:32]
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Ybar=torch.cat(accs).mean(0)
        curve={};rcurve={}
        for k in (2,4,8,16,32):
            Qk=orth(A32[:,:k])
            curve[k]=ce_eval(li,Qk,Ybar@Qk)-base
            Rk=orth(torch.randn(D,k,device=DEV,generator=g))
            rcurve[k]=ce_eval(li,Rk,Ybar@Rk)-base
            print(f'k={k:2d}: watch +{curve[k]:.4f} | random +{rcurve[k]:.4f}',
                  flush=True)
    pa=curve[8]>=0.8*curve[32]
    pb=all(abs(v)<=0.005 for v in rcurve.values())
    pc=curve[2]>=0.4*curve[32]
    out={'curve':{str(k):v for k,v in curve.items()},
         'random':{str(k):v for k,v in rcurve.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) saturates by k=8: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random floor: {'HELD' if pb else 'VIOLATED'}")
    print(f"(c) k=2 >= 40%: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
