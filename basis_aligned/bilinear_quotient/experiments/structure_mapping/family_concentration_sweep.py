"""Ledger-19 follow-through: the span-concentration claim (privacy lives in
the top-8 span; tail coords 9-48 share normally -- sections 212/216) was
measured under ONE reader ensemble in each model. Sweep it: bilin18 L6 full
vs tail-coords pooled LORO under ensembles A=(2,3,5,9,13,17),
C=(2,4,5,8,13,16), D=(3,7,9,12,15,17); bilin12 L4 under A12=(1,3,5,7,9,11)
and C12=(2,3,5,7,9,10). Pooled aggregation only (the ledger-19 rule).

REGISTERED PREDICTIONS: (a) bilin18: tail >= 1.5x full in ALL three
ensembles (concentration is construction-robust); (b) bilin12: tail >= 1.5x
full in both ensembles WHERE full <= 0.3 (concentration only meaningful
where the notch shows; C12 full was 0.19); (c) tail-coords absolute value
>= 0.30 everywhere (the tail genuinely participates, not just "less bad")."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from bilin18_joint_removal import FW, orth, DEV
from tier2_model import load_elriggs
K=40; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'family_concentration_sweep_results.json')

@torch.no_grad()
def sweep(name, Wl, ensembles):
    m2,_=load_elriggs(name, device=DEV)
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
    Yw=grab(Wl,0,300); mu=Yw.mean(0)
    _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
    Yf=(grab(Wl,384,448)-mu).float()
    cacheP={}
    def getP(j):
        if j not in cacheP:
            Yj=grab(j,0,60)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(),
                                     full_matrices=False)
            cacheP[j]=orth(Vhj[:NF].T)
        return cacheP[j]
    def loro(readers, comp0):
        V=orth(Vh[comp0:comp0+K].T)
        yproj=(Yf@V)[:20000]
        fams={}
        for j in readers:
            P=getP(j)
            mlp=m2.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            fams[j]=[0.5*(M+M.T) for M in
                     (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                      for f in range(NF))]
        r2s=[]
        for jout in readers:
            X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                           if j2!=jout for Mm in Ms])
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            B=W[:80]
            for Mm in fams[jout][:12]:
                ct=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=ct.var().clamp_min(1e-12)
                Mre=((B@Mm.flatten())@B).view(K,K)
                ch=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                r2s.append(1-float(((ch-ct)**2).mean()/vt))
        return sorted(r2s)[len(r2s)//2]
    out={}
    for tag,rs in ensembles.items():
        rs=tuple(r for r in rs if r!=Wl)
        full=loro(rs,0); tail=loro(rs,8)
        out[tag]={'full':full,'tail':tail}
        print(f'{name} L{Wl} set {tag}: full {full:+.3f} tail {tail:+.3f} '
              f'ratio {tail/max(full,1e-3):.1f}',flush=True)
    del m2; torch.cuda.empty_cache()
    return out

@torch.no_grad()
def main():
    t0=time.time()
    r18=sweep('bilin18',6,{'A':(2,3,5,9,13,17),'C':(2,4,5,8,13,16),
                            'D':(3,7,9,12,15,17)})
    r12=sweep('bilin12',4,{'A12':(1,3,5,7,9,11),'C12':(2,3,5,7,9,10)})
    pa=all(v['tail']>=1.5*max(v['full'],1e-3) for v in r18.values())
    b12rel=[v for v in r12.values() if v['full']<=0.3]
    pb=all(v['tail']>=1.5*max(v['full'],1e-3) for v in b12rel)
    pc=all(v['tail']>=0.30 for v in list(r18.values())+list(r12.values()))
    out={'bilin18':r18,'bilin12':r12,
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) b18 tail>=1.5x full all sets: {'HELD' if pa else 'FAILED'}")
    print(f"(b) b12 same where notch shows: {'HELD' if pb else 'FAILED'}")
    print(f"(c) tail >=0.30 everywhere: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
