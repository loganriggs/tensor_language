"""Last headline under the ledger-19 audit: the dialects result (section
220 -- each reader's own basis reconstructs its held-out forms over the
private span far better than the population basis; gap +0.56, control
-0.10). Measured then with one form split (even/odd), one rank (18), one
ensemble. Sweep: splits {even/odd, first/second half} x ranks {12, 18, 24}
x ensembles {A=(2,3,5,9,13,17), C=(2,4,5,8,13,16)} -- 12 variants -- for
the private span (writer L6, top-8 coords) and control writer L9.

REGISTERED PREDICTIONS: (a) private-span median self-cross gap >= 0.30 in
ALL 12 variants (dialects construction-robust); (b) control writer gap
within +/-0.15 in >= 10/12 variants (a shared code shows no self advantage);
(c) worst-case private gap across variants reported; if any variant drops
below 0.15 the headline gains a construction-dependence note like 228's."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; NF=40; KS=8
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_dialects_sweep_results.json')
ENS={'A':(2,3,5,9,13,17),'C':(2,4,5,8,13,16)}

@torch.no_grad()
def main():
    t0=time.time()
    cacheF={}
    def fams_for(Wl, readers):
        key=(Wl,readers)
        if key in cacheF: return cacheF[key]
        Yw=grab(Wl,0,300); mu=Yw.mean(0)
        _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
        V=orth(Vh[:KS].T)
        yproj=((grab(Wl,384,448)-mu).float()@V)[:20000]
        fams={}
        for j in readers:
            Yj=grab(j,0,60)
            _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(),
                                     full_matrices=False)
            P=orth(Vhj[:NF].T)
            mlp=m.transformer.h[j].mlp
            L=mlp.Left.weight.detach().float()@V
            R=mlp.Right.weight.detach().float()@V
            DwP=mlp.Down.weight.detach().float().T@P
            fams[j]=torch.stack([(0.5*(M+M.T)).flatten() for M in
                     (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                      for f in range(NF))])
        cacheF[key]=(fams,yproj)
        return cacheF[key]
    def gap(Wl, readers, split, RB):
        fams,yproj=fams_for(Wl,readers)
        gaps=[]
        for j in readers:
            if split=='eo': tr=fams[j][0::2]; te=fams[j][1::2]
            else: tr=fams[j][:NF//2]; te=fams[j][NF//2:]
            _,_,Wt=torch.linalg.svd(tr, full_matrices=False)
            Bs=Wt[:RB]
            X=torch.cat([fams[j2] for j2 in readers if j2!=j])
            _,_,Wx=torch.linalg.svd(X, full_matrices=False)
            Bc=Wx[:RB]
            med={}
            for tag,B in (('s',Bs),('c',Bc)):
                r2s=[]
                for v in te:
                    Mm=v.view(KS,KS)
                    ct=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                    vt=ct.var().clamp_min(1e-12)
                    Mre=((B@v)@B).view(KS,KS)
                    ch=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                    r2s.append(1-float(((ch-ct)**2).mean()/vt))
                med[tag]=sorted(r2s)[len(r2s)//2]
            gaps.append(med['s']-med['c'])
        return sorted(gaps)[len(gaps)//2]
    res={'L6':{},'L9':{}}
    for ens,rs in ENS.items():
        for split in ('eo','fh'):
            for RB in (12,18,24):
                tag=f'{ens}/{split}/r{RB}'
                for Wl in (6,9):
                    readers=tuple(r for r in rs if r!=Wl)
                    g=gap(Wl,readers,split,RB)
                    res[f'L{Wl}'][tag]=g
                print(f'{tag}: L6 gap {res["L6"][tag]:+.3f} | '
                      f'L9 gap {res["L9"][tag]:+.3f}',flush=True)
    g6=list(res['L6'].values()); g9=list(res['L9'].values())
    pa=all(g>=0.30 for g in g6)
    pb=sum(1 for g in g9 if abs(g)<=0.15)>=10
    worst=min(g6)
    out={'gaps':res,'worst_private':worst,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"\nworst private gap {worst:+.3f}")
    print(f"(a) all 12 >= 0.30: {'HELD' if pa else 'FAILED'}")
    print(f"(b) control within 0.15 in >=10/12: {'HELD' if pb else 'FAILED'}"
          f" ({sum(1 for g in g9 if abs(g)<=0.15)}/12)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
