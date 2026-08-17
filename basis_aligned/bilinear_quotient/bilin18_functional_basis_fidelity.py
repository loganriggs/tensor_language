"""Does the ~80-functional basis actually carry the readers' L1-dependence?

§58: the 240 sampled coupling matrices have effective rank 80. The functional-
compression claim needs a fidelity gate: reconstruct each reader-form's coupling
matrix from the top-r principal functionals (fit on OTHER readers' forms -- leave-one-
reader-out, so the basis is not fit to what it reconstructs) and measure how much of
the form's L1-quadratic component survives, as R^2 of the coefficient contribution
y^T B y over held-out activations. REGISTERED PREDICTIONS:
  (a) at r=80, median leave-one-reader-out R^2 >= 0.6 (the basis is shared structure,
      not per-reader overfit);
  (b) r=8 achieves under half of r=80's median R^2 (the diversity is real -- no tiny
      basis suffices);
Control: a random-matrix basis of matched rank, predicted R^2 <= 0.1 at r=80."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_functional_basis_fidelity_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    accs=[]
    for i in range(0,300,6):
        acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=1, acc=acc); accs.append(acc[0])
    Y1=torch.cat(accs); Y1c=(Y1-Y1.mean(0)).float()
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    V=orth(Vh[:K].T)
    yproj=(Y1c@V)[:20000]                    # held-out L1 coords for R^2 eval
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
        mats=[]
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            mats.append(0.5*(M+M.T))
        fams[j]=mats
    g=torch.Generator(device=DEV).manual_seed(0)
    out={'r':{}}
    for r in (8,80):
        r2s=[]; r2r=[]
        for jout in READERS:
            train=[Mm.flatten() for j2,Ms in fams.items() if j2!=jout for Mm in Ms]
            X=torch.stack(train)
            _,_,W=torch.linalg.svd(X, full_matrices=False)
            Basis=W[:r]                       # (r, K*K)
            Rb=torch.randn(r,K*K,device=DEV,generator=g)
            Rb=Rb/Rb.norm(dim=1,keepdim=True)
            for Mm in fams[jout][:12]:
                c_true=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
                vt=c_true.var().clamp_min(1e-12)
                for tag,Bset,acc_ in (('basis',Basis,r2s),('random',Rb,r2r)):
                    co=Bset@Mm.flatten()
                    Mre=(co@Bset).view(K,K)
                    c_hat=torch.einsum('na,ab,nb->n',yproj,Mre,yproj)
                    acc_.append(1-float(((c_hat-c_true)**2).mean()/vt))
        med=sorted(r2s)[len(r2s)//2]; medr=sorted(r2r)[len(r2r)//2]
        out['r'][r]={'median_r2':med,'random_r2':medr}
        print(f'r={r:3d}: leave-one-reader-out median R^2 {med:.3f} '
              f'(random basis {medr:.3f})',flush=True)
    pa=out['r'][80]['median_r2']>=0.6
    pb=out['r'][8]['median_r2']<0.5*out['r'][80]['median_r2']
    pc=out['r'][80]['random_r2']<=0.1
    out['pred_a']=bool(pa); out['pred_b']=bool(pb); out['ctrl']=bool(pc)
    print(f"\n(a) r=80 R^2 >= 0.6: {'HELD' if pa else 'FAILED'} | "
          f"(b) r=8 under half of r=80: {'HELD' if pb else 'FAILED'} | "
          f"control random <= 0.1: {'OK' if pc else 'VIOLATED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
