#!/usr/bin/env python3
"""MLP0 context-only (quadratic) branch effective rank (CPU, exact). Completes §2673/§2675.

Context-only branch B(d)=D[(Ld)*(Rd)], output o = d^T M_o d, M_o = sum_h D[o,h] L[h,:]^T R[h,:]. The family
{M_o} is linear in the D rows; its effective rank = nonzero spectrum of (D^TD)^{1/2}(LL^T (.) RR^T)(D^TD)^{1/2}
(Hadamard product; 4608x4608; bound <= d_model=1152). Uniform output weighting; exact weight-space structural
rank of the quadratic map. Noise-free, no forwards.
"""
import torch, json, time
BLOB="/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin"
def main():
    t0=time.time(); sd=torch.load(BLOB,map_location="cpu",weights_only=False)
    if hasattr(sd,'state_dict'): sd=sd.state_dict()
    L=sd["transformer.h.0.mlp.Left.weight"].double(); R=sd["transformer.h.0.mlp.Right.weight"].double(); D=sd["transformer.h.0.mlp.Down.weight"].double()
    d=L.shape[1]
    WtW=(L@L.T)*(R@R.T); DtD=D.T@D
    ew,ev=torch.linalg.eigh(DtD); ew=torch.clamp(ew,min=0); S12=ev@torch.diag(torch.sqrt(ew))@ev.T
    M=S12@WtW@S12; M=0.5*(M+M.T)
    lam=torch.linalg.eigvalsh(M); lam=torch.clamp(lam,min=0).flip(0); lam=lam/lam.sum()
    ent=float(torch.exp(-(lam*torch.log(lam+1e-300)).sum())); cum=torch.cumsum(lam,0)
    res={"status":"complete","analysis":"mlp0_context_only_quadratic_branch_effective_rank","d_model_bound":d,
         "effective_rank_entropy":ent,"rank_90pct":int((cum<0.90).sum())+1,"rank_99pct":int((cum<0.99).sum())+1,
         "top1_energy":float(lam[0]),"frozen_threshold_compressible_below":288,
         "verdict":("low_dim_compressible" if ent<288 else "high_rank_not_compressible"),
         "runtime_s":time.time()-t0}
    json.dump(res,open("mlp0_context_branch_rank_results.json","w"),indent=2)
    print(json.dumps({k:res[k] for k in ("verdict","effective_rank_entropy","rank_90pct","runtime_s")},indent=2))
if __name__=="__main__": main()
