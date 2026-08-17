"""The discriminator section 211 called for -- and prematurely called
unbuildable. Is L6's privacy concentrated in its top-8 (regularizer) span, or
does it cover the whole code? Behavioral LORO with the writer coords V taken
as SVD components 9-48 of the writer's output (the 40 non-principal coords,
past the regularizer span), readers (2,3,5,9,13,17) minus self, fresh eval
rows. Writers: L6 (the private one) and L9 (the control -- equally
regularizer-flavored, shared at 0.54 on full coords).

REGISTERED PREDICTIONS: (a) L6 tail-coords LORO >= 0.40 (privacy lives in the
top-8 span; the refined regularizer story revives: readers shun the span, not
the writer); alternative if < 0.25: whole-code privacy, story stays dead.
(b) control: L9 tail-coords LORO >= 0.40 (dropping the span does not destroy
sharing for a normal writer). (c) null: random 40-dim basis in coefficient
space <= 0.1 for both."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; NF=40; KT=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_l6_tailcoords_results.json')

@torch.no_grad()
def loro_tail(Wl, readers):
    Yw=grab(Wl,0,300); mu=Yw.mean(0)
    _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
    V=orth(Vh[8:8+KT].T)                     # components 9-48 only
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
        Rb=torch.randn(80,KT*KT,device=DEV,generator=g)
        Rb=Rb/Rb.norm(dim=1,keepdim=True)
        for Mm in fams[jout][:12]:
            c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
            vt=c_true.var().clamp_min(1e-12)
            for Bset,acc_ in ((Basis,r2s),(Rb,r2r)):
                Mre=((Bset@Mm.flatten())@Bset).view(KT,KT)
                c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                acc_.append(1-float(((c_hat-c_true)**2).mean()/vt))
    return sorted(r2s)[len(r2s)//2], sorted(r2r)[len(r2r)//2]

@torch.no_grad()
def main():
    t0=time.time()
    res={}
    for Wl in (6,9):
        readers=tuple(r for r in (2,3,5,9,13,17) if r!=Wl)
        med,rnd=loro_tail(Wl,readers)
        res[Wl]=(med,rnd)
        print(f'writer L{Wl} coords 9-48: LORO {med:+.3f} (random {rnd:+.3f})',
              flush=True)
    pa=res[6][0]>=0.40; alt=res[6][0]<0.25
    pb=res[9][0]>=0.40
    pc=all(v[1]<=0.1 for v in res.values())
    out={'writers':{str(k):{'loro':v[0],'random':v[1]} for k,v in res.items()},
         'pred_a':bool(pa),'alt_whole_code':bool(alt),'pred_b':bool(pb),
         'pred_c':bool(pc)}
    print(f"\n(a) L6 tail >=0.40 span-concentrated: {'HELD' if pa else 'FAILED'}"
          f"{' (alt <0.25: whole-code privacy)' if alt else ''}")
    print(f"(b) L9 control >=0.40: {'HELD' if pb else 'FAILED'}")
    print(f"(c) randoms <=0.1: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
