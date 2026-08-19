"""Matrix-SAE v2: held-out, honestly sized.

v1 was vacuous (300 atoms > 240 samples). v2: 120 atoms fit on five readers' 200
functionals; score the SIXTH reader's held-out 40. Baseline: the dense orthogonal
80-basis achieves LORO R^2 0.71. REGISTERED PREDICTIONS: (a) held-out sparse coding
reaches R^2 >= 0.6 at L0 <= 15 (competitive with dense at ~5x fewer active
components per functional); (b) if (a) holds, atoms are still NOT simple (median
eff-rank > 6) -- the §63/§68 medium-rank character is expected to persist; a simple-
atom outcome would be the surprise worth having."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; K=48; NF=40
READERS=(2,3,5,9,13,17)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_matrix_sae2_results.json')

@torch.no_grad()
def build_fams():
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
    for j in READERS:
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
    return fams,yproj

def fit_dict(X, n_atoms=120, l1=0.01, steps=6000, lr=3e-3, seed=0):
    n,d=X.shape
    g=torch.Generator(device=DEV).manual_seed(seed)
    Wd=torch.nn.Parameter(torch.randn(n_atoms,d,device=DEV,generator=g)/d**0.5)
    We=torch.nn.Parameter(torch.randn(d,n_atoms,device=DEV,generator=g)/d**0.5)
    b=torch.nn.Parameter(torch.zeros(n_atoms,device=DEV))
    opt=torch.optim.Adam([Wd,We,b],lr=lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,steps)
    for t in range(steps):
        a=torch.relu(X@We+b)
        loss=((a@Wd-X)**2).sum(1).mean()+l1*a.abs().sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return Wd.detach(),We.detach(),b.detach()

def sparse_code(x, Wd, n_iter=400, l1=0.01):
    a=torch.zeros(Wd.shape[0],device=DEV,requires_grad=True)
    opt=torch.optim.Adam([a],lr=0.05)
    for t in range(n_iter):
        loss=((a@Wd-x)**2).sum()+l1*a.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return a.detach()

def main():
    t0=time.time()
    fams,yproj=build_fams()
    r2s=[]; l0s=[]
    Wd_last=None
    for jout in (5,13):     # two held-out readers, cost control
        X=torch.stack([Mm.flatten() for j in READERS if j!=jout
                       for Mm in fams[j]])
        Wd,We,b=fit_dict(X)
        Wd_last=Wd
        for Mm in fams[jout][:12]:
            x=Mm.flatten()
            a=sparse_code(x,Wd)
            rec=(a@Wd).view(K,K)
            c=torch.einsum('na,ab,nb->n',yproj,Mm,yproj)
            ch=torch.einsum('na,ab,nb->n',yproj,rec,yproj)
            r2s.append(1-float(((ch-c)**2).mean()/c.var().clamp_min(1e-12)))
            l0s.append(int((a.abs()>1e-3).sum()))
    med_r2=sorted(r2s)[len(r2s)//2]
    med_l0=sorted(l0s)[len(l0s)//2]
    ers=[]
    usage=torch.randn(0)
    for i in range(40):
        A=Wd_last[i].view(K,K); A=0.5*(A+A.T)
        e=torch.linalg.eigvalsh(A.double()).abs()
        ers.append(float(e.sum()**2/(e**2).sum()))
    med_er=sorted(ers)[len(ers)//2]
    out={'heldout_median_r2':med_r2,'median_l0':med_l0,
         'atom_effrank_median':med_er}
    pa=med_r2>=0.6 and med_l0<=15
    pb=med_er>6
    out['pred_a']=bool(pa); out['pred_b_atoms_still_complex']=bool(pb)
    print(f'held-out sparse coding: median R^2 {med_r2:.2f} at L0 {med_l0} '
          f'(dense-80 baseline: 0.71)')
    print(f'atom eff-rank median: {med_er:.1f}')
    print(f"(a) competitive sparse (R2>=0.6, L0<=15): {'HELD' if pa else 'FAILED'}")
    print(f"(b) atoms still complex (>6): "
          f"{'HELD' if pb else 'SURPRISE -- simple atoms found'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
