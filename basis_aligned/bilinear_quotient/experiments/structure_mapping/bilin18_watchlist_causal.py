"""Causal validation of the watch-lists. Sections 130-132 established, by subspace
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
     'bilin18_watchlist_causal_results.json')

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
    for li,lj in ((5,6),(9,10),(13,14)):
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
        R8=orth(torch.randn(D,8,device=DEV,generator=g))
        accs=[]
        for i in range(0,36,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
        Ybar=torch.cat(accs).mean(0)
        row={}
        for tag,Q in (('attn_watch',A8),('mlp_watch',G8),('random',R8)):
            row[tag]=ce_eval(li,Q,Ybar@Q)-base
        res[f'{li}->{lj}']=row
        print(f'L{li}->L{lj}: attn-watch +{row["attn_watch"]:.4f} | '
              f'mlp-watch +{row["mlp_watch"]:.4f} | random +{row["random"]:.4f}',
              flush=True)
        if row['mlp_watch']>row['attn_watch']: wins_a+=1
        if min(row['mlp_watch'],row['attn_watch'])>row['random']: wins_b+=1
        rnds.append(row['random'])
    pa=wins_a>=2; pb=wins_b>=2; pc=sum(rnds)/3<=0.005
    out={'per_edge':res,'pred_a':bool(pa),'pred_b':bool(pb),'null_c':bool(pc)}
    print(f"\n(a) mlp-watch deletion costlier (>=2/3): {'HELD' if pa else 'FAILED'} ({wins_a}/3)")
    print(f"(b) both beat random (>=2/3): {'HELD' if pb else 'FAILED'} ({wins_b}/3)")
    print(f"(c) random floor <=0.005: {'HELD' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
