"""The both-directions typed-edge audit, 'keep-only' direction (benchmark rule,
section 171): for the coordinate-typed L5 interface, does keeping ONLY the
watched dims preserve the layer's downstream value? Arms: (i) delete L5's whole
MLP write (mean-ablate all 1152 dims) = D_full; (ii) keep only the top-8 of
L6's attention watch-list (mean-ablate the 1144-dim complement) = D_keep8;
(iii) keep a random 8 = D_rand8.

REGISTERED PREDICTIONS: (a) D_keep8 <= 0.6 * D_full (the 8-dim interface
carries >= 40% of L5's whole-write value); (b) D_rand8 >= 0.9 * D_full (random
8 dims preserve almost nothing); (c) sanity: D_full >= 0.08 (L5's write
matters at all -- its rank-0 stand-in cost +0.095)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_attn_norm_share import attn_mean  # noqa (import side effects none)
from tier2_model import rope_tables, apply_rot
NH,HD,D=9,128,1152
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_keep_only_results.json')

@torch.no_grad()
def ce_eval(Q,cbar):
    hs=[]
    if Q is not None:
        def hook(mod,i_,o_):
            c=o_.float()@Q
            return (o_-((c-cbar)@Q.T).to(o_.dtype))
        hs.append(m.transformer.h[5].mlp.register_forward_hook(hook))
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
    # L6 attention watch-list top-8 (as in interface ladder)
    caps=[]
    h=m.transformer.h[6].attn.register_forward_pre_hook(
        lambda mod,inp: caps.append(
            F.rms_norm(inp[0].detach().reshape(-1,D).float(),(D,))) or None)
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    X=torch.cat(caps); Xc=X-X.mean(0)
    C=Xc.T@Xc/Xc.shape[0]
    ev,U=torch.linalg.eigh(C.double())
    Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
    a=m.transformer.h[6].attn
    mats=[]
    for h_ in range(NH):
        for wq,wk in ((a.c_q,a.c_k),(a.c_q2,a.c_k2)):
            Wq=wq.weight.detach().float().view(NH,HD,D)[h_]
            Wk=wk.weight.detach().float().view(NH,HD,D)[h_]
            Uk,S,Vk=torch.linalg.svd(Ch@Wq.T@Wk@Ch)
            mats.append(torch.cat([Uk[:,:4],Vk[:4].T],dim=1))
    Sf=torch.cat(mats,dim=1)
    Ua,_,_=torch.linalg.svd(Sf@Sf.T)
    W8=orth(Ua[:,:8])
    g=torch.Generator(device=DEV).manual_seed(0)
    R8=orth(torch.randn(D,8,device=DEV,generator=g))
    accs=[]
    for i in range(0,36,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=5, acc=acc); accs.append(acc[0])
    Ybar=torch.cat(accs).mean(0)
    base=ce_eval(None,None)
    I=torch.eye(D,device=DEV)
    D_full=ce_eval(I,Ybar@I)-base
    def comp(Q):
        P=torch.eye(D,device=DEV)-Q@Q.T
        Uc,S,_=torch.linalg.svd(P)
        Qc=Uc[:,:D-8]
        return orth(Qc)
    Qc8=comp(W8); Qcr=comp(R8)
    D_keep8=ce_eval(Qc8,Ybar@Qc8)-base
    D_rand8=ce_eval(Qcr,Ybar@Qcr)-base
    print(f'base {base:.4f} | delete-all +{D_full:.4f} | keep-watched-8 '
          f'+{D_keep8:.4f} | keep-random-8 +{D_rand8:.4f}')
    pa=D_keep8<=0.6*D_full; pb=D_rand8>=0.9*D_full; pc=D_full>=0.08
    out={'base':base,'D_full':D_full,'D_keep8':D_keep8,'D_rand8':D_rand8,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"(a) watched-8 preserves >=40%: {'HELD' if pa else 'FAILED'}")
    print(f"(b) random-8 preserves ~nothing: {'HELD' if pb else 'FAILED'}")
    print(f"(c) L5 write matters: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
