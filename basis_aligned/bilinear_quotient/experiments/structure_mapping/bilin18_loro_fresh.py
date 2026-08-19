"""Faithful fresh-rows replication of section 61's LORO 0.71 -- construction
copied verbatim from bilin18_functional_basis_fidelity.py with ONE change: the
held-out activation rows for the R^2 evaluation come from FRESH data (rows
384-448, never used for the writer coords V or the reader spans), removing any
row overlap between basis construction and evaluation. Companion to
bilin18_loro_replicate.py, whose bar (a) failure was a mis-registered metric
(it measured matrix-element R^2, a stronger quantity section 61 never claimed).

REGISTERED PREDICTIONS: (a) r=80 activation-weighted LORO median R^2 >= 0.55
on fresh rows (section 61 replicates; original 0.711); (b) random-basis
control <= 0.1; (c) if (a) fails, section 61 earns a ledger correction --
row-overlap-dependent."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_loro_fresh_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs); Y1c=(Y1-Y1.mean(0)).float()
    mu=Y1.mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    fresh=[]
    for i in range(384,448,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); fresh.append(acc[0])
    yproj=((torch.cat(fresh)-mu).float()@V)[:20000]
    fams={}
    for j in READERS:
        accs=[]
        for i in range(0,60,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=j, acc=acc); accs.append(acc[0])
        Yj=torch.cat(accs)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        P=orth(Vhj[:NF].T)
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        fams[j]=[0.5*(M+M.T) for M in
                 (torch.einsum('k,ka,kb->ab',DwP[:,f],L,R) for f in range(NF))]
    g=torch.Generator(device=DEV).manual_seed(0)
    r=80; r2s=[]; r2r=[]
    for jout in READERS:
        X=torch.stack([Mm.flatten() for j2,Ms in fams.items()
                       if j2!=jout for Mm in Ms])
        _,_,W=torch.linalg.svd(X, full_matrices=False)
        Basis=W[:r]
        Rb=torch.randn(r,K*K,device=DEV,generator=g)
        Rb=Rb/Rb.norm(dim=1,keepdim=True)
        for Mm in fams[jout][:12]:
            c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
            vt=c_true.var().clamp_min(1e-12)
            for Bset,acc_ in ((Basis,r2s),(Rb,r2r)):
                Mre=((Bset@Mm.flatten())@Bset).view(K,K)
                c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                acc_.append(1-float(((c_hat-c_true)**2).mean()/vt))
    med=sorted(r2s)[len(r2s)//2]; medr=sorted(r2r)[len(r2r)//2]
    pa=med>=0.55; pb=medr<=0.1
    out={'loro_fresh':med,'random':medr,'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f'fresh-rows LORO r=80: {med:.3f} (orig 0.711) | random {medr:.3f}')
    print(f"(a) >=0.55: {'HELD -- section 61 replicates' if pa else 'FAILED -- ledger correction'}")
    print(f"(b) random <=0.1: {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
