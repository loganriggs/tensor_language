"""Convenience-sample check: the vocabulary with ALL 16 downstream MLP readers.

The 240-functional family used 6 readers chosen for depth coverage (user flagged
this). REGISTERED PREDICTIONS: (a) with all 16 readers (16x40 = 640 functionals),
family eff-rank stays in [70, 130] (the code is not an artifact of the six);
(b) readers OUTSIDE the original six reconstruct from the six-reader top-80 basis at
median R^2 >= 0.55 (slightly below the within-six 0.71 allowed);
(c) no single excluded reader falls below R^2 0.3 (no reader speaks a different
language)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
SIX=(2,3,5,9,13,17)
ALL=tuple(range(2,18))
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_all_readers_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    def collect(li):
        outs=[]
        h=m.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            m(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Y1=collect(1); Y1c=Y1-Y1.mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    yproj=(Y1c@V)[:20000]
    fams={}
    for j in ALL:
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        mats=[]
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            Ms=0.5*(M+M.T)
            mats.append(Ms/Ms.norm().clamp_min(1e-12))
        fams[j]=mats
    Xall=torch.stack([Mm.flatten() for j in ALL for Mm in fams[j]])
    sv=torch.linalg.svdvals(Xall); e=sv**2
    er=float(e.sum()**2/(e**2).sum())
    Xsix=torch.stack([Mm.flatten() for j in SIX for Mm in fams[j]])
    _,_,W6=torch.linalg.svd(Xsix, full_matrices=False)
    B=W6[:80]
    out={'effrank_all16':er,'excluded':{}}
    meds=[]
    for j in [x for x in ALL if x not in SIX]:
        r2s=[]
        for Mm in fams[j][:12]:
            c=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
            co=B@Mm.flatten(); Mre=(co@B).view(K,K)
            ch=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
            r2s.append(1-float(((ch-c)**2).mean()/c.var().clamp_min(1e-12)))
        med=sorted(r2s)[len(r2s)//2]
        out['excluded'][j]=med
        meds.append(med)
        print(f'excluded reader L{j:2d}: median R^2 from six-reader basis {med:.2f}',
              flush=True)
    overall=sorted(meds)[len(meds)//2]
    pa=70<=er<=130; pb=overall>=0.55; pc=min(meds)>=0.3
    out['median_excluded_r2']=overall
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['pred_c']=bool(pc)
    print(f'\nall-16 family eff-rank: {er:.0f}')
    print(f"(a) eff-rank in [70,130]: {'HELD' if pa else 'FAILED'} | "
          f"(b) excluded median >= 0.55: {'HELD' if pb else 'FAILED'} | "
          f"(c) min >= 0.3: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
