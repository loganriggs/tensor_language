"""Completes the cross-model private-writer signature. bilin18's L6 privacy
is concentrated in its top-8 span (tail coords 9-48 shared at 0.41, section
212). Is bilin12's L4 the same shape? Behavioral LORO for writer L4 with
coords restricted to SVD components 9-48, control writer L8 (shared at 0.48
on full coords), readers (1,3,5,7,9,11) minus self.

Note recorded with the registration: bilin12-L4's top-8 span is NOT
deletion-improving (+0.036, family_regularizer_results) -- privacy and
regularizer character already dissociate cross-model regardless of this
run's outcome.

REGISTERED PREDICTIONS: (a) L4 tail-coords LORO >= 0.30 (privacy is
span-concentrated in both models -- the signature transfers whole);
alternative < 0.15: bilin12's privacy is whole-code, the signature differs.
(b) control L8 tail-coords >= 0.30. (c) random 80-basis <= 0.1 both."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
K=40; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin12_l4_tailcoords_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    m2,_=load_elriggs('bilin12', device=DEV)
    D=m2.transformer.wte.weight.shape[1]
    def grab(li, r0, r1):
        outs=[]
        h=m2.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(r0,r1,6):
            b=FW[i:i+6,:513].to(DEV)
            m2(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    res={}
    for Wl in (4,8):
        readers=tuple(r for r in (1,3,5,7,9,11) if r!=Wl)
        Yw=grab(Wl,0,300); mu=Yw.mean(0)
        _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
        V=orth(Vh[8:8+K].T)
        yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
        fams={}
        for j in readers:
            Yj=grab(j,0,60)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(),
                                     full_matrices=False)
            P=orth(Vhj[:NF].T)
            mlp=m2.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            fams[j]=[0.5*(M+M.T) for M in
                     (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                      for f in range(NF))]
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
        med=sorted(r2s)[len(r2s)//2]; rnd=sorted(r2r)[len(r2r)//2]
        res[Wl]=(med,rnd)
        print(f'writer L{Wl} coords 9-48: LORO {med:+.3f} (random {rnd:+.3f})',
              flush=True)
    pa=res[4][0]>=0.30; alt=res[4][0]<0.15
    pb=res[8][0]>=0.30
    pc=all(v[1]<=0.1 for v in res.values())
    out={'writers':{str(k):{'loro':v[0],'random':v[1]}
                    for k,v in res.items()},
         'pred_a':bool(pa),'alt_whole_code':bool(alt),
         'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) L4 tail >=0.30 span-concentrated: {'HELD' if pa else 'FAILED'}"
          f"{' (alt <0.15: whole-code)' if alt else ''}")
    print(f"(b) control L8 >=0.30: {'HELD' if pb else 'FAILED'}")
    print(f"(c) randoms <=0.1: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
