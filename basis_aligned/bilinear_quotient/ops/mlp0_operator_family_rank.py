#!/usr/bin/env python3
"""EXACT effective rank of the MLP0 token-context operator family (CPU, math-review analysis).

Insight: for the bilinear MLP0(z)=D[(Lz)*(Rz)]+bias, the token-conditioned linear operator on a context
deviation delta, K_t delta = D[diag(L x_t)(R delta) + diag(R x_t)(L delta)], is LINEAR in the token embedding
x_t. So K_t = sum_k x_t[k] G_k, G_k = D diag(L[:,k]) R + D diag(R[:,k]) L, and the whole 50k-operator family
lies in a subspace of dimension <= d_model=1152. rung525 showed operators do not GROUP (are not equal); this
computes the family's EFFECTIVE RANK exactly (the real compressibility question), noise-free.

Reduction: family covariance nonzero spectrum = eig of Sig^{1/2} Gram Sig^{1/2}, Sig = E_t[x_t x_t^T] (embedding
2nd moment), Gram[k,l]=tr(G_k^T G_l) = L^T(P*QRR)L + R^T(P*QLL)R + L^T(P*QRL)R + transpose, P=D^TD, QRR=RR^T,
QLL=LL^T, QRL=RL^T (* = elementwise). CAVEAT: uses the token embedding as x_t and uniform vocab weighting; the
RANK is a property of the bilinear weights, robust to Codex's exact centering/gauge/attention0-context basis
(near-orthogonal transforms) — this is the intrinsic operator-family rank, representative not byte-identical.
"""
import torch, json, time
BLOB="/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin"

def main():
    t0=time.time()
    sd=torch.load(BLOB,map_location="cpu",weights_only=False)
    if hasattr(sd,'state_dict'): sd=sd.state_dict()
    L=sd["transformer.h.0.mlp.Left.weight"].double()
    R=sd["transformer.h.0.mlp.Right.weight"].double()
    D=sd["transformer.h.0.mlp.Down.weight"].double()
    wte=sd["transformer.wte.weight"].double()
    H,d=L.shape; V=wte.shape[0]
    P=D.T@D; QRR=R@R.T; QLL=L@L.T; QRL=R@L.T
    Gram=L.T@((P*QRR)@L)+R.T@((P*QLL)@R)
    cross=L.T@((P*QRL)@R); Gram=Gram+cross+cross.T; Gram=0.5*(Gram+Gram.T)
    Sig=(wte.T@wte)/V
    ew,ev=torch.linalg.eigh(Sig); ew=torch.clamp(ew,min=0)
    S12=ev@torch.diag(torch.sqrt(ew))@ev.T
    M=S12@Gram@S12; M=0.5*(M+M.T)
    lam=torch.linalg.eigvalsh(M); lam=torch.clamp(lam,min=0).flip(0); lam=lam/lam.sum()
    ent=float(torch.exp(-(lam*torch.log(lam+1e-300)).sum()))
    pr=float(1.0/(lam**2).sum()); cum=torch.cumsum(lam,0)
    r50=int((cum<0.50).sum())+1; r90=int((cum<0.90).sum())+1; r99=int((cum<0.99).sum())+1
    res={"status":"complete","analysis":"mlp0_operator_family_effective_rank","d_model_bound":d,
         "effective_rank_entropy":ent,"effective_rank_participation_ratio":pr,
         "top1_energy":float(lam[0]),"rank_50pct":r50,"rank_90pct":r90,"rank_99pct":r99,
         "frozen_threshold_compressible_below":288,
         "verdict":("low_dim_compressible_operator_family" if ent<288 else
                    "high_rank_operator_family_not_compressible"),
         "caveat":"intrinsic rank from bilinear weights; token-embedding x_t, uniform vocab; robust to gauge/centering",
         "runtime_s":time.time()-t0}
    json.dump(res,open("mlp0_operator_family_rank_results.json","w"),indent=2)
    print(json.dumps({k:res[k] for k in ("verdict","effective_rank_entropy","rank_90pct","rank_99pct","top1_energy","runtime_s")},indent=2))

if __name__=="__main__": main()
