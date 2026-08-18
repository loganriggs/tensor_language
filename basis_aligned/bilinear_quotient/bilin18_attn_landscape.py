"""The unscanned half of the sharing landscape: ATTENTION writers. Every
landscape measurement so far used MLP outputs as the writer signal. Same
instrument over attention outputs: behavioral LORO over attn writers
(1,4,6,8,12,16) -- attn6 included deliberately (is privacy a property of the
LAYER or of its MLP?), readers the standard MLP set (2,3,5,9,13,17) minus
the co-located reader where applicable. Nulls measured per the ledger-17 rule.

REGISTERED PREDICTIONS: (a) attention writers are shared (median across the
six >= 0.40 -- the vocabulary is signal-generic, readers decode whatever is
in the stream); (b) attn6 specifically is NOT private (>= 0.35): privacy
belongs to the MLP's quadratic content, not to layer-6 real estate --
consistent with section 219 (contested directions) and the relay picture
(attention transports, MLPs originate); (c) measured random-basis nulls
<= 0.1 everywhere."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab as grab_mlp
D=1152; K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_attn_landscape_results.json')

@torch.no_grad()
def grab_attn(li, r0, r1):
    outs=[]
    h=m.transformer.h[li].attn.register_forward_hook(
        lambda mo_,i_,o_: outs.append(
            (o_[0] if isinstance(o_,tuple) else o_)
            .detach().reshape(-1,D).float()))
    for i in range(r0,r1,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    h.remove()
    return torch.cat(outs)

@torch.no_grad()
def loro_attn(Wl):
    readers=tuple(r for r in (2,3,5,9,13,17) if r!=Wl)
    Yw=grab_attn(Wl,0,300); mu=Yw.mean(0)
    _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
    V=orth(Vh[:K].T)
    yproj=((grab_attn(Wl,384,448)-mu).float()@V)[:20000]
    fams={}
    for j in readers:
        Yj=grab_mlp(j,0,60)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        fams[j]=[0.5*(M+M.T) for M in
                 (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R) for f in range(NF))]
    g=torch.Generator(device=DEV).manual_seed(0)
    r2s=[]; r2r=[]
    for jout in readers:
        X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                       if j2!=jout for Mm in Ms])
        _,_,W=torch.linalg.svd(X, full_matrices=False)
        Basis=W[:80]
        Rb=torch.randn(80,K*K,device=DEV,generator=g)
        Rb=Rb/Rb.norm(dim=1,keepdim=True)
        for Mm in fams[jout][:12]:
            c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
            vt=c_true.var().clamp_min(1e-12)
            for Bset,acc_ in ((Basis,r2s),(Rb,r2r)):
                Mre=((Bset@Mm.flatten())@Bset).view(K,K)
                c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                acc_.append(1-float(((c_hat-c_true)**2).mean()/vt))
    return sorted(r2s)[len(r2s)//2], sorted(r2r)[len(r2r)//2]

@torch.no_grad()
def main():
    t0=time.time()
    res={}
    for Wl in (1,4,6,8,12,16):
        med,rnd=loro_attn(Wl)
        res[Wl]=(med,rnd)
        print(f'attn writer L{Wl:2d}: LORO {med:+.3f} (random {rnd:+.3f})',
              flush=True)
    meds=sorted(v[0] for v in res.values())
    pa=meds[len(meds)//2]>=0.40
    pb=res[6][0]>=0.35
    pc=all(v[1]<=0.1 for v in res.values())
    out={'writers':{str(k):{'loro':v[0],'random':v[1]} for k,v in res.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) attention shared, median >=0.40: {'HELD' if pa else 'FAILED'}"
          f" ({meds[len(meds)//2]:+.3f})")
    print(f"(b) attn6 not private >=0.35: {'HELD' if pb else 'FAILED'}")
    print(f"(c) measured nulls <=0.1: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
