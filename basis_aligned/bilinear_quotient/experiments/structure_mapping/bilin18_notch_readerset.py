"""THE CRITICAL CHECK: bilin12's L4 notch is reader-ensemble-dependent
(present with readers {1,3,5,9,11}+, absent when 5 leaves or {1,11} thins,
and sensitive to pooled-vs-fold-median aggregation). Is bilin18's L6 notch --
the section-215 universality headline, published in the report -- fragile
the same way? Writer L6 (and control L9) under four reader ensembles:
A=(2,3,5,9,13,17) [original], B=(1,4,7,10,12,16), C=(2,4,5,8,13,16),
D=(3,7,9,12,15,17); both aggregations reported (pooled median and
median-of-fold-medians).

REGISTERED PREDICTIONS: (a) ROBUST: L6 pooled <= 0.25 in all four ensembles
(the bilin18 notch is ensemble-independent; the fragility is bilin12-only
and 215 needs a bilin12-scope note); (b) FRAGILE: L6 >= 0.4 in any ensemble
= the notch construct itself is ensemble-relative, ledger entry, report
correction, PREREGISTRATION P3 rewritten to specify the ensemble; (c)
control L9 varies < 0.15 pooled across ensembles."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_notch_readerset_results.json')
SETS={'A':(2,3,5,9,13,17),'B':(1,4,7,10,12,16),
      'C':(2,4,5,8,13,16),'D':(3,7,9,12,15,17)}

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
    def loro2(Wl, readers):
        readers=tuple(r for r in readers if r!=Wl)
        Yw=grab(Wl,0,300); mu=Yw.mean(0)
        _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
        V=orth(Vh[:K].T)
        yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
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
        pooled=[]; foldmeds=[]
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
            pooled+=r2s
            foldmeds.append(sorted(r2s)[len(r2s)//2])
        return (sorted(pooled)[len(pooled)//2],
                sorted(foldmeds)[len(foldmeds)//2])
    res={}
    for tag,rs in SETS.items():
        p6,f6=loro2(6,rs); p9,f9=loro2(9,rs)
        res[tag]={'L6_pooled':p6,'L6_foldmed':f6,
                  'L9_pooled':p9,'L9_foldmed':f9}
        print(f'set {tag}: L6 pooled {p6:+.3f} foldmed {f6:+.3f} | '
              f'L9 pooled {p9:+.3f} foldmed {f9:+.3f}',flush=True)
    l6=[v['L6_pooled'] for v in res.values()]
    l9=[v['L9_pooled'] for v in res.values()]
    pa=all(x<=0.25 for x in l6)
    pb=any(x>=0.4 for x in l6)
    pc=max(l9)-min(l9)<0.15
    out={'sets':res,'pred_a_robust':bool(pa),'pred_b_fragile':bool(pb),
         'pred_c':bool(pc)}
    print(f"\n(a) robust (all <=0.25): {'HELD' if pa else 'FAILED'}")
    print(f"(b) fragile (any >=0.4): {'YES -- ledger' if pb else 'no'}")
    print(f"(c) L9 stable <0.15: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
