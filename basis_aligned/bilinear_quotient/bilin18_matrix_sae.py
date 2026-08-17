"""The last open door on vocabulary structure: a sparse dictionary over MATRICES.

§68/§63: the 80 principal words are medium-rank (11-20) and only partly nameable. The
untested compression: the 240 functionals might be SPARSE combinations of an
overcomplete set of SIMPLE (low-rank) atoms even though no small orthogonal basis is
low-rank -- the matrix-SAE door. Fit a 300-atom L1 dictionary over the 240
unit-normalised coupling matrices (vectors in 1176-d). REGISTERED PREDICTIONS:
  (a) reconstruction R^2 >= 0.75 at L0 <= 12 active atoms per functional (a sparser
      per-functional description than the dense 80-code);
  (b) the used atoms are simple: median eff-rank of the top-40 most-used atoms <= 6
      (the sparse door delivers low-rank nameable pieces that PCA words are not);
Control: the same fit on the weight-shuffled family should need more atoms per
functional at matched R^2 (sparsity is a trained property), registered L0 ratio
shuffled/trained >= 1.3."""
import json, sys, time, torch, itertools
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, FW, DEV
from tier2_model import load_elriggs
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_matrix_sae_results.json')

@torch.no_grad()
def family(mdl):
    def collect(li):
        outs=[]
        h=mdl.transformer.h[li].mlp.register_forward_hook(
            lambda mo_,i_,o_: outs.append(o_.detach().reshape(-1,D).float()))
        for i in range(0,60,6):
            b=FW[i:i+6,:513].to(DEV)
            mdl(b[:,:-1].contiguous(), b[:,1:].contiguous())
        h.remove()
        return torch.cat(outs)
    Y1=collect(1); Y1c=Y1-Y1.mean(0)
    _,_,Vh=torch.linalg.svd(Y1c, full_matrices=False)
    Q,_=torch.linalg.qr(Vh[:K].T); V=Q[:,:K]
    rows=[]
    for j in READERS:
        Yj=collect(j)
        _,_,Vhj=torch.linalg.svd((Yj-Yj.mean(0)).float(), full_matrices=False)
        Pq,_=torch.linalg.qr(Vhj[:NF].T); P=Pq[:,:NF]
        mlp=mdl.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float()@V
        R=mlp.Right.weight.detach().float()@V
        DwP=mlp.Down.weight.detach().float().T@P
        for f in range(NF):
            M=torch.einsum('k,ka,kb->ab',DwP[:,f],L,R)
            Ms=0.5*(M+M.T)
            rows.append((Ms/Ms.norm().clamp_min(1e-12)).flatten())
    return torch.stack(rows)

def fit_sae(X, n_atoms=300, steps=6000, lr=3e-3):
    n,d=X.shape
    g=torch.Generator(device=DEV).manual_seed(0)
    Wd=torch.nn.Parameter(torch.randn(n_atoms,d,device=DEV,generator=g)/d**0.5)
    We=torch.nn.Parameter(torch.randn(d,n_atoms,device=DEV,generator=g)/d**0.5)
    b=torch.nn.Parameter(torch.zeros(n_atoms,device=DEV))
    opt=torch.optim.Adam([Wd,We,b],lr=lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    best=None
    for l1 in (0.001,0.003,0.01,0.03):
        for p_ in (Wd,We,b):
            with torch.no_grad():
                if p_ is b: p_.zero_()
                else: p_.normal_(0,1/d**0.5)
        opt=torch.optim.Adam([Wd,We,b],lr=lr)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
        for t in range(steps):
            a=torch.relu(X@We+b)
            rec=a@Wd
            loss=((rec-X)**2).sum(1).mean()+l1*a.abs().sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            a=torch.relu(X@We+b)
            l0=float((a>1e-4).float().sum(1).mean())
            r2=1-float(((a@Wd-X)**2).sum()/ (X**2).sum())
        if best is None or (r2>=0.75 and l0<best[1]) or (best[2]<0.75 and r2>best[2]):
            best=(l1,l0,r2,Wd.detach().clone(),
                  torch.relu(X@We+b).detach().clone())
    return best

def main():
    t0=time.time()
    mdl,cfg=load_elriggs('bilin18', device=DEV)
    Xt=family(mdl)
    l1t,l0t,r2t,Wd,acts=fit_sae(Xt)
    print(f'trained family: l1={l1t} -> L0 {l0t:.1f} atoms/functional, R^2 {r2t:.2f}')
    usage=(acts>1e-4).float().sum(0)
    top=usage.argsort(descending=True)[:40]
    ers=[]
    for i in top:
        A=Wd[i].view(K,K); A=0.5*(A+A.T)
        e=torch.linalg.eigvalsh(A.double()).abs()
        ers.append(float(e.sum()**2/(e**2).sum()))
    med_er=sorted(ers)[len(ers)//2]
    g=torch.Generator(device=DEV).manual_seed(0)
    for blk in mdl.transformer.h:
        for W in (blk.mlp.Left.weight, blk.mlp.Right.weight, blk.mlp.Down.weight):
            flat=W.data.flatten()
            W.data=flat[torch.randperm(flat.numel(),device=DEV,generator=g)]\
                .view_as(W.data)
    Xs=family(mdl)
    l1s,l0s,r2s,_,_=fit_sae(Xs)
    print(f'shuffled family: l1={l1s} -> L0 {l0s:.1f}, R^2 {r2s:.2f}')
    out={'trained':{'l0':l0t,'r2':r2t},'shuffled':{'l0':l0s,'r2':r2s},
         'atom_effrank_median':med_er}
    pa=r2t>=0.75 and l0t<=12
    pb=med_er<=6
    pc=(l0s/max(l0t,1e-9))>=1.3 if r2s>=0.7 else None
    out['pred_a']=bool(pa); out['pred_b']=bool(pb)
    out['pred_c']=pc
    print(f'\natom eff-rank median (top-40 used): {med_er:.1f}')
    print(f"(a) sparse+faithful (L0<=12, R2>=0.75): {'HELD' if pa else 'FAILED'}")
    print(f"(b) atoms simple (<=6): {'HELD' if pb else 'FAILED'}")
    print(f"(c) sparsity trained (ratio>=1.3): "
          f"{'HELD' if pc else ('FAILED' if pc is not None else 'N/A (shuffled fit poor)')}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
