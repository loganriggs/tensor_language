"""bilin12's ordinal solitary-reader claim just failed its ensemble sweep
(v2 scorecard: L11 worst for 1/10 writers in one ensemble). bilin18's
stronger claim -- L17 SECEDES (worst fold for every writer, median ~0.11,
sections 210/218, published) -- was measured under ensemble A only. Sweep
it: per-fold tables for writers (0,1,5,8,12) under three ensembles that
contain reader 17: A=(2,3,5,9,13,17), D=(3,7,9,12,15,17), F=(1,4,8,11,14,17).

REGISTERED PREDICTIONS: (a) L17 is the worst fold for >= 4/5 writers in ALL
three ensembles (secession ensemble-robust); (b) L17's median fold <= 0.25
in all three (the categorical bar); (c) if either fails, sections 210/218's
bilin18 secession claim gains a construction-dependence correction like
ledger 19, and P5 is rewritten."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_solitary_sweep_results.json')
ENS={'A':(2,3,5,9,13,17),'D':(3,7,9,12,15,17),'F':(1,4,8,11,14,17)}
WRITERS=(0,1,5,8,12)

@torch.no_grad()
def main():
    t0=time.time()
    cacheP={}
    def getP(j):
        if j not in cacheP:
            Yj=grab(j,0,60)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(),
                                     full_matrices=False)
            cacheP[j]=orth(Vhj[:NF].T)
        return cacheP[j]
    cacheW={}
    def wassets(Wl):
        if Wl not in cacheW:
            Yw=grab(Wl,0,300); mu=Yw.mean(0)
            _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
            cacheW[Wl]=(orth(Vh[:K].T),(grab(Wl,384,448)-mu).float())
        return cacheW[Wl]
    res={}
    for tag,ens in ENS.items():
        worst17=0; med17=[]
        for Wl in WRITERS:
            readers=tuple(r for r in ens if r!=Wl)
            V,Yf=wassets(Wl)
            yproj=(Yf@V)[:20000]
            fams={}
            for j in readers:
                P=getP(j)
                mlp=m.transformer.h[j].mlp
                L=mlp.Left.weight.detach().float()@V
                R=mlp.Right.weight.detach().float()@V
                DwP=mlp.Down.weight.detach().float().T@P
                fams[j]=[0.5*(M+M.T) for M in
                         (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                          for f in range(NF))]
            folds={}
            for jout in readers:
                X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                               if j2!=jout for Mm in Ms])
                _,_,W=torch.linalg.svd(X, full_matrices=False)
                B=W[:80]
                r2s=[]
                for Mm in fams[jout][:12]:
                    ct=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                    vt=ct.var().clamp_min(1e-12)
                    Mre=((B@Mm.flatten())@B).view(K,K)
                    ch=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                    r2s.append(1-float(((ch-ct)**2).mean()/vt))
                folds[jout]=sorted(r2s)[len(r2s)//2]
            if min(folds,key=folds.get)==17: worst17+=1
            med17.append(folds[17])
        m17=sorted(med17)[len(med17)//2]
        res[tag]={'worst17':worst17,'l17_median':m17,
                  'l17_folds':med17}
        print(f'ens {tag}: L17 worst {worst17}/5, median fold {m17:+.3f} '
              f'({[f"{x:+.2f}" for x in med17]})',flush=True)
    pa=all(v['worst17']>=4 for v in res.values())
    pb=all(v['l17_median']<=0.25 for v in res.values())
    out={'ens':res,'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\n(a) worst >=4/5 all ensembles: {'HELD' if pa else 'FAILED'}")
    print(f"(b) median <=0.25 all ensembles: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
