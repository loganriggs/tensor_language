"""Deciding section 223's fork on-box (called off-box too hastily -- the
section-212 lesson repeating): is attn6's privacy INTRINSIC or BORROWED from
the mlp6 span directions it shares real estate with? Behavioral LORO over
attn6's output coords orthogonalized against the mlp6 top-8 span (project the
8 directions out of the activation data before the writer PCA), and the same
orthogonalization applied to the shared control attn12. If attn6's low
consensus lives in those 8 directions, orthogonalizing recovers it; if layer
6's attention emits its own private content, it stays low.

REGISTERED PREDICTIONS: (a) control attn12 is unmoved by orthogonalization
(>= 0.38, was 0.47 -- removing 8 of 1152 directions is negligible for a
normal writer); (b) fork: attn6-orthogonalized >= 0.30 = privacy BORROWED
(the layer-level notch reduces to the one 8-dim object) / <= 0.20 = privacy
INTRINSIC to the layer's attention too (the notch is genuinely layer-wide);
between = mixed, report as such. (c) measured random-basis nulls <= 0.1."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab as grab_mlp
from bilin18_attn_landscape import grab_attn
D=1152; K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_attn6_borrowed_results.json')

@torch.no_grad()
def loro_orth(Wl, Qspan):
    readers=tuple(r for r in (2,3,5,9,13,17) if r!=Wl)
    Yw=grab_attn(Wl,0,300); mu=Yw.mean(0)
    Ywc=(Yw-mu).float()
    Ywc=Ywc-(Ywc@Qspan)@Qspan.T
    _,_,Vh=torch.linalg.svd(Ywc, full_matrices=False)
    V=orth(Vh[:K].T)
    Yf=(grab_attn(Wl,384,448)-mu).float()
    Yf=Yf-(Yf@Qspan)@Qspan.T
    yproj=(Yf@V)[:20000]
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
    Y6=grab_mlp(6,0,120); mu6=Y6.mean(0)
    _,_,Vh6=torch.linalg.svd((Y6-mu6).float(), full_matrices=False)
    Qspan=orth(Vh6[:8].T)
    res={}
    for Wl in (6,12):
        med,rnd=loro_orth(Wl,Qspan)
        res[Wl]=(med,rnd)
        print(f'attn{Wl} orth-to-span: LORO {med:+.3f} (random {rnd:+.3f})',
              flush=True)
    pa=res[12][0]>=0.38
    a6=res[6][0]
    verdict=('BORROWED' if a6>=0.30 else
             'INTRINSIC' if a6<=0.20 else 'MIXED')
    pc=all(v[1]<=0.1 for v in res.values())
    out={'attn6_orth':a6,'attn12_orth':res[12][0],'verdict':verdict,
         'pred_a':bool(pa),'pred_c':bool(pc)}
    print(f"\n(a) control unmoved >=0.38: {'HELD' if pa else 'FAILED'}")
    print(f"(b) fork verdict: {verdict} (attn6 orth {a6:+.3f}; was 0.126)")
    print(f"(c) nulls <=0.1: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
