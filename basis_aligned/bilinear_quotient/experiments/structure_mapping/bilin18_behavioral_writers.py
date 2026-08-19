"""Section 114 re-measured with the right instrument. Section 114 concluded
cross-reader vocabulary sharing is L1-SPECIFIC (writers L0/L9 at R^2
0.12-0.25) -- but it used the matrix-element metric, which section 208 just
showed underestimates behavioral sharing 2.5x (L1: 0.26 elementwise vs 0.64
activation-weighted). Re-run LORO for writers L0, L9, with L1 as anchor, all
on the activation-weighted metric with fresh evaluation rows (384-448).
Readers (2,3,5,9,13,17); 9->15 swap when the writer is L9.

REGISTERED PREDICTIONS (section 114's claim standing): (a) behavioral LORO
< 0.45 for BOTH L0 and L9 (L1-specificity is real, not a metric artifact);
(b) L1 anchor >= 0.55 (section 208 machinery sanity); (c) random basis <= 0.1
for every writer. If (a) fails at >= 0.55 for either writer, section 114
earns ledger entry #16 and vocabulary sharing is writer-general."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_behavioral_writers_results.json')

@torch.no_grad()
def grab(li, r0, r1):
    accs=[]
    for i in range(r0,r1,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc); accs.append(acc[0])
    return torch.cat(accs)

@torch.no_grad()
def loro(Wl, readers):
    Yw=grab(Wl,0,300); mu=Yw.mean(0)
    _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
    V=orth(Vh[:K].T)
    yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
    fams={}
    for j in readers:
        Yj=grab(j,0,60)
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
    for Wl,readers in ((1,(2,3,5,9,13,17)),(0,(2,3,5,9,13,17)),
                       (9,(2,3,5,15,13,17))):
        med,rnd=loro(Wl,readers)
        res[Wl]=(med,rnd)
        print(f'writer L{Wl}: behavioral LORO {med:+.3f} (random {rnd:+.3f})',
              flush=True)
    pa=res[0][0]<0.45 and res[9][0]<0.45
    pb=res[1][0]>=0.55
    pc=all(v[1]<=0.1 for v in res.values())
    out={'writers':{str(k):{'loro':v[0],'random':v[1]} for k,v in res.items()},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) L0,L9 both <0.45 (L1-specific): {'HELD' if pa else 'FAILED -- ledger #16'}")
    print(f"(b) L1 anchor >=0.55: {'HELD' if pb else 'FAILED'}")
    print(f"(c) randoms <=0.1: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
