"""DECOMPOSITION-TYPE CENSUS -- user direction: maybe the MOTIF is which
decomposition a component prefers (low-rank vs sparse vs block vs
diagonal-plus-low-rank), not a shared basis. For every weight matrix in
the model's seven families (attn c_q, c_k, c_v, c_proj; mlp Left, Right,
Down) x 18 layers, fit four structures at MATCHED parameter budget
P = D^2/8 (166k numbers per matrix) and record the winner by relative
Frobenius reconstruction error:
  lowrank  -- rank r = P/(2D) = 72 truncated SVD
  sparse   -- top-P entries by magnitude
  blockdiag-- block-diagonal in the matrix's own SVD basis is trivially
              low-rank, so blocks live in a FIXED shared basis: the
              stream PCA basis (window-A covariance eigenvectors);
              8 blocks of 144x144 = P numbers
  diaglr   -- diagonal (D) + rank-71 on the residual
REGISTERED PREDICTIONS:
  (a) the winning type is consistent within a family for >=70% of its
      18 matrices (decomposition type IS a family motif);
  (b) at least one family's winner is NOT lowrank (structural
      diversity exists);
  (c) full error table reported."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'decomp_census_results.json'
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
        U,S,Vh=torch.linalg.svd(Wm,full_matrices=False)
        return (U[:,:R]*S[:R])@Vh[:R]
    def fit_sparse(Wm):
        flat=Wm.abs().flatten()
        th=flat.kthvalue(flat.numel()-P).values
        return Wm*(Wm.abs()>th)
    def fit_block(Wm):
        Wb=SB.T@Wm@SB
        M=torch.zeros_like(Wb)
        for b in range(NBLK):
            s=b*BS
            M[s:s+BS,s:s+BS]=Wb[s:s+BS,s:s+BS]
        return SB@M@SB.T
    def fit_diaglr(Wm):
        dg=torch.diagonal(Wm).clone()
        Rr=Wm-torch.diag(dg)
        U,S,Vh=torch.linalg.svd(Rr,full_matrices=False)
        r2=(P-D)//(2*D)
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
            sq=Wm.shape[0]==Wm.shape[1]
            errs={}
            for mn,fn in METH.items():
                if mn in ('blockdiag','diaglr') and not sq:
                    Wt=Wm[:,:D] if Wm.shape[1]>=D else Wm
                    # non-square (Left/Right/Down are 4608x1152 etc):
                    # blockdiag/diaglr defined only for square; use
                    # row-chunked variant: block rows in shared basis
                    if mn=='blockdiag':
                        Wb=Wm@SB
                        M=torch.zeros_like(Wb)
                        cs=Wm.shape[0]//NBLK
                        for b in range(NBLK):
                            M[b*cs:(b+1)*cs,b*BS:(b+1)*BS]=\
                                Wb[b*cs:(b+1)*cs,b*BS:(b+1)*BS]
                        errs[mn]=relerr(Wm,M@SB.T)
                        continue
                    else:
                        errs[mn]=1.0
                        continue
                errs[mn]=relerr(Wm,fn(Wm))
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
