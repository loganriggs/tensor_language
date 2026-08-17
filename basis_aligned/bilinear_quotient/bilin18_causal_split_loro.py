"""Discriminator for the L6-vs-L12 split (weak_writers): for mid-depth writers
some readers sit UPSTREAM and never see the writer's output -- their LORO
folds are acausal. Per-fold behavioral R^2 (held-out reader x writer) for
writers (0,1,6,9,12), readers (2,3,5,9,13,17) minus self, folds split into
DOWNSTREAM (reader index > writer) vs UPSTREAM.

REGISTERED PREDICTIONS: (a) pooled downstream-fold median >= 0.5 for every
writer including L6 (sharing is general among readers that actually read the
writer; L6's 0.16 was upstream folds dragging the median); (b) pooled
upstream-fold median <= 0.3 (acausal folds are the weak ones); (c) if (a)
fails at L6 specifically, the L6 exception is real and earns its own section.
"""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_behavioral_writers import grab
D=1152; K=48; NF=40
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_causal_split_loro_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    down=[]; up=[]; table={}
    for Wl in (0,1,6,9,12):
        readers=tuple(r for r in (2,3,5,9,13,17) if r!=Wl)
        Yw=grab(Wl,0,300); mu=Yw.mean(0)
        _,_,Vh=torch.linalg.svd((Yw-mu).float(), full_matrices=False)
        V=orth(Vh[:K].T)
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
            fams[j]=[0.5*(M+M.T) for M in
                     (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
                      for f in range(NF))]
        table[Wl]={}
        for jout in readers:
            X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                           if j2!=jout for Mm in Ms])
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            Basis=W[:80]
            r2s=[]
            for Mm in fams[jout][:12]:
                c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=c_true.var().clamp_min(1e-12)
                Mre=((Basis@Mm.flatten())@Basis).view(K,K)
                c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                r2s.append(1-float(((c_hat-c_true)**2).mean()/vt))
            med=sorted(r2s)[len(r2s)//2]
            table[Wl][jout]=med
            (down if jout>Wl else up).append((Wl,jout,med))
        row=' '.join(f'L{j}:{table[Wl][j]:+.2f}' for j in readers)
        print(f'writer L{Wl:2d}: {row}',flush=True)
    dmed=sorted(x[2] for x in down)[len(down)//2]
    umed=sorted(x[2] for x in up)[len(up)//2] if up else float('nan')
    wl6_down=[x[2] for x in down if x[0]==6]
    wl6d=sorted(wl6_down)[len(wl6_down)//2]
    pa=all(sorted([x[2] for x in down if x[0]==Wl])
           [len([x for x in down if x[0]==Wl])//2]>=0.5
           for Wl in (0,1,6,9,12))
    pb=umed<=0.3
    out={'table':{str(k):{str(j):v for j,v in d.items()}
                  for k,d in table.items()},
         'down_median':dmed,'up_median':umed,'l6_down_median':wl6d,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'\npooled downstream median {dmed:+.3f} | upstream {umed:+.3f} | '
          f'L6 downstream {wl6d:+.3f}')
    print(f"(a) every writer's downstream median >=0.5: "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) upstream pooled <=0.3: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
