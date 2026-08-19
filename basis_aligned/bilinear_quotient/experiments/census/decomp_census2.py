"""DECOMPOSITION-TYPE CENSUS v2 -- v1 crashed on Down's orientation and
gave low-rank an unfair budget on non-square matrices (rank chosen for
square shapes). v2: every method gets the SAME budget P = D^2/8 per
matrix regardless of shape: lowrank rank r = P/(m+n); sparse top-P
entries; blockdiag nb = mn/P blocks in the shared stream basis (rotating
only the D-sized side for non-square); diag+lowrank square-only.
REGISTERED PREDICTIONS (as v1):
  (a) winning type consistent within family for >=13/18 matrices, for
      >=5 of 7 families (decomposition type IS a family motif);
  (b) at least one family's winner is NOT lowrank;
  (c) full error table reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'decomp_census2_results.json'
CA=300; NB=8
P=D*D//8; R=P//(2*D); NBLK=8; BS=D//NBLK

@torch.no_grad()
def main():
    t0=time.time()
    cov=torch.zeros(D,D,device=DEV)
    def h(mo,i_,o_):
        pass
    xs=[]
    hh=m.transformer.h[9].register_forward_pre_hook(
        lambda mo,args: xs.append(args[0].detach().float()
                                  .reshape(-1,D)))
    for i in range(CA,CA+NB*4,4):
        bb=FW[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    hh.remove()
    X=torch.cat(xs); cov=X.T@X
    ev,SB=torch.linalg.eigh(cov)          # shared stream basis
    def relerr(Wm,Wa):
        return float((Wm-Wa).pow(2).sum()/Wm.pow(2).sum())
    def fit_lowrank(Wm):
        mn,nn=Wm.shape
        r=max(1,P//(mn+nn))
        U,S,Vh=torch.linalg.svd(Wm,full_matrices=False)
        return (U[:,:r]*S[:r])@Vh[:r]
    def fit_sparse(Wm):
        flat=Wm.abs().flatten()
        th=flat.kthvalue(max(1,flat.numel()-P)).values
        return Wm*(Wm.abs()>th)
    def fit_block(Wm):
        mn,nn=Wm.shape
        nb=max(2,(mn*nn)//P)
        if mn==D: Wb=SB.T@Wm
        else: Wb=Wm
        if nn==D: Wb=Wb@SB
        M=torch.zeros_like(Wb)
        b0=mn//nb; b1=nn//nb
        for b in range(nb):
            M[b*b0:(b+1)*b0,b*b1:(b+1)*b1]=                Wb[b*b0:(b+1)*b0,b*b1:(b+1)*b1]
        if mn==D: M=SB@M
        if nn==D: M=M@SB.T
        return M
    def fit_diaglr(Wm):
        if Wm.shape[0]!=Wm.shape[1]: return None
        dg=torch.diagonal(Wm).clone()
        Rr=Wm-torch.diag(dg)
        r2=max(1,(P-D)//(2*D))
        U,S,Vh=torch.linalg.svd(Rr,full_matrices=False)
        return torch.diag(dg)+(U[:,:r2]*S[:r2])@Vh[:r2]
    FAMS={'c_q':lambda b:b.attn.c_q.weight,
          'c_k':lambda b:b.attn.c_k.weight,
          'c_v':lambda b:b.attn.c_v.weight,
          'c_proj':lambda b:b.attn.c_proj.weight,
          'Left':lambda b:b.mlp.Left.weight,
          'Right':lambda b:b.mlp.Right.weight,
          'Down':lambda b:b.mlp.Down.weight}
    METH={'lowrank':fit_lowrank,'sparse':fit_sparse,
          'blockdiag':fit_block,'diaglr':fit_diaglr}
    tab={}; winners={}
    for fname,getw in FAMS.items():
        tab[fname]=[]; winners[fname]=[]
        for li in range(18):
            Wm=getw(m.transformer.h[li]).detach().float()
            errs={}
            for mn,fn in METH.items():
                Wa=fn(Wm)
                if Wa is None: continue
                errs[mn]=relerr(Wm,Wa)
            tab[fname].append({k:round(v,4) for k,v in errs.items()})
            winners[fname].append(min(errs,key=errs.get))
        wc={}
        for w in winners[fname]: wc[w]=wc.get(w,0)+1
        top=max(wc,key=wc.get)
        print(f'{fname:7s}: winner {top} ({wc[top]}/18) all={wc}',
              flush=True)
    cons=sum(1 for f in FAMS
             if max((winners[f].count(w) for w in set(winners[f])))>=13)
    tops={f:max(set(winners[f]),key=winners[f].count) for f in FAMS}
    pa=cons>=5
    pb=any(t!='lowrank' for t in tops.values())
    out={'winners':winners,'family_top':tops,'table':tab,
         'consistent_families':cons,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) >=5/7 families consistent (13/18): "
          f"{'HELD' if pa else 'FAILED'} ({cons}/7)")
    print(f"(b) some family not lowrank: {'HELD' if pb else 'FAILED'} "
          f"({tops})")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
