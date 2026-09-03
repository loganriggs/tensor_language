#!/usr/bin/env python3
"""Token-context operator-family effective rank across ALL 18 MLP blocks (CPU, exact).

Extends §2673 (MLP0 high-rank) to every block: is the high-rank token-context operator a universal property of
bilin18's bilinear MLPs, or block-specific (a low-rank block = a compressible target for the smaller program)?
Same exact reduction: family cov nonzero spectrum = eig(Sig^{1/2} Gram Sig^{1/2}), Gram[k,l]=tr(G_k^T G_l), per
block's L/R/D. Sig = token-embedding 2nd moment (same input basis proxy for all blocks; the rank is a property
of each block's bilinear weights). Noise-free, no forwards.
"""
import torch, json, time
BLOB="/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin"

def block_rank(L,R,D,Sig):
    P=D.T@D; Gram=L.T@((P*(R@R.T))@L)+R.T@((P*(L@L.T))@R)
    cross=L.T@((P*(R@L.T))@R); Gram=Gram+cross+cross.T; Gram=0.5*(Gram+Gram.T)
    ew,ev=torch.linalg.eigh(Sig); ew=torch.clamp(ew,min=0); S12=ev@torch.diag(torch.sqrt(ew))@ev.T
    M=S12@Gram@S12; M=0.5*(M+M.T)
    lam=torch.linalg.eigvalsh(M); lam=torch.clamp(lam,min=0).flip(0); lam=lam/lam.sum()
    ent=float(torch.exp(-(lam*torch.log(lam+1e-300)).sum())); cum=torch.cumsum(lam,0)
    return ent, int((cum<0.90).sum())+1, float(lam[0])

def main():
    t0=time.time(); sd=torch.load(BLOB,map_location="cpu",weights_only=False)
    if hasattr(sd,'state_dict'): sd=sd.state_dict()
    wte=sd["transformer.wte.weight"].double(); V=wte.shape[0]; Sig=(wte.T@wte)/V
    d=Sig.shape[0]; rows=[]
    for b in range(18):
        L=sd[f"transformer.h.{b}.mlp.Left.weight"].double()
        R=sd[f"transformer.h.{b}.mlp.Right.weight"].double()
        D=sd[f"transformer.h.{b}.mlp.Down.weight"].double()
        ent,r90,top1=block_rank(L,R,D,Sig)
        rows.append({"block":b,"eff_rank_entropy":ent,"rank_90pct":r90,"top1_energy":top1,
                     "compressible": ent<288})
        print(f"  MLP{b:2d}: eff_rank={ent:6.1f}  r90={r90:4d}  top1={top1:.3f}  {'LOW-DIM' if ent<288 else 'high-rank'}")
    ents=[r["eff_rank_entropy"] for r in rows]; ncomp=sum(r["compressible"] for r in rows)
    res={"status":"complete","analysis":"all_mlp_operator_family_effective_rank","d_model":d,
         "frozen_threshold_compressible_below":288,"per_block":rows,
         "min_eff_rank":min(ents),"max_eff_rank":max(ents),"n_compressible_blocks":ncomp,
         "verdict":("some_mlp_blocks_low_dim_compressible" if ncomp>0 else
                    "all_18_mlp_blocks_high_rank_token_context_operators"),
         "runtime_s":time.time()-t0}
    json.dump(res,open("all_mlp_operator_family_rank_results.json","w"),indent=2)
    print(json.dumps({k:res[k] for k in ("verdict","min_eff_rank","max_eff_rank","n_compressible_blocks","runtime_s")},indent=2))

if __name__=="__main__": main()
