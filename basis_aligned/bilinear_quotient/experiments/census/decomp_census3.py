"""DECOMPOSITION-TYPE CENSUS v3 -- v2 found sparse wins 126/126, but
sparse beats low-rank on iid Gaussian matrices too at this budget, so
the claim needs its null (ledger 21). v3 adds, per matrix: the same
four fits on a shape-matched iid Gaussian matrix (seeded), and the
excess kurtosis of the model matrix's entries.
REGISTERED PREDICTIONS:
  (a) model sparse-error <= 0.8x the Gaussian sparse-error for >=5/7
      families (heavy-tail structure beyond chance);
  (b) mean excess kurtosis > 1 for >=5/7 families;
  (c) the sparse-vs-lowrank preference itself survives on Gaussian
      (reported; the TYPE census only counts where model beats null)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import m, FW, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'decomp_census3_results.json'
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
            gg=torch.Generator(device=DEV).manual_seed(li*7+1)
            G=torch.randn(Wm.shape,device=DEV,generator=gg)
            errs['g_sparse']=relerr(G,fit_sparse(G))
            errs['g_lowrank']=relerr(G,fit_lowrank(G))
            zz=(Wm-Wm.mean())/Wm.std()
            errs['kurt']=float((zz**4).mean())-3.0
            tab[fname].append({k:round(v,4) for k,v in errs.items()})
            winners[fname].append(min(errs,key=errs.get))
        wc={}
        for w in winners[fname]: wc[w]=wc.get(w,0)+1
        top=max(wc,key=wc.get)
        se=[e['sparse'] for e in tab[fname]]
        ge=[e['g_sparse'] for e in tab[fname]]
        ku=[e['kurt'] for e in tab[fname]]
        print(f'{fname:7s}: winner {top} ({wc[top]}/18) | sparse-err '
              f'{sum(se)/18:.3f} vs gaussian {sum(ge)/18:.3f} | '
              f'kurtosis {sum(ku)/18:.2f}',flush=True)
    fam_ok=0; fam_ku=0
    for f in FAMS:
        se=sum(e['sparse'] for e in tab[f])/18
        ge=sum(e['g_sparse'] for e in tab[f])/18
        ku=sum(e['kurt'] for e in tab[f])/18
        if se<=0.8*ge: fam_ok+=1
        if ku>1: fam_ku+=1
    pa=fam_ok>=5; pb=fam_ku>=5
    out={'winners':winners,'table':tab,
         'fam_beats_gaussian':fam_ok,'fam_kurtotic':fam_ku,
         'pred_a':bool(pa),'pred_b':bool(pb)}
    print(f"(a) sparse beats gaussian-null 0.8x ({fam_ok}/7): "
          f"{'HELD' if pa else 'FAILED'}")
    print(f"(b) kurtosis > 1 ({fam_ku}/7): {'HELD' if pb else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
