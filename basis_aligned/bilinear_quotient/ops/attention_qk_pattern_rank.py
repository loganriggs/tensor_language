#!/usr/bin/env python3
"""Per-head QK squared-bilinear pattern effective rank across all 18 blocks (CPU, exact). Capstone to §2673/§2675/§2676.

Attention pattern bilinear form per head h: score(x_i,x_j) = x_i^T (c_q_h^T c_k_h) x_j (and the squared c_q2/c_k2
term). Effective rank of W_h = c_q_h^T c_k_h (bounded by head_dim=128) measures attention's per-head pattern
compressibility. Compare to the MLP token-context operator rank (438-929 of 1152, no bottleneck, §2673/§2675).
Unifies bilin18's compressibility map: attention = 128-dim head bottleneck (compressible, the §312 frontier);
MLP = full-1152 high-rank (not compressible). Noise-free, no forwards.
"""
import torch, json, time, statistics as st
BLOB="/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin"
def eff_rank(W):
    s=torch.linalg.svdvals(W.double()); p=(s**2); p=p/p.sum()
    return float(torch.exp(-(p*torch.log(p+1e-30)).sum()))
def main():
    t0=time.time(); sd=torch.load(BLOB,map_location="cpu",weights_only=False)
    if hasattr(sd,'state_dict'): sd=sd.state_dict()
    nh,hd,de=9,128,1152; rows=[]
    for b in range(18):
        cq=sd[f"transformer.h.{b}.attn.c_q.weight"].reshape(nh,hd,de); ck=sd[f"transformer.h.{b}.attn.c_k.weight"].reshape(nh,hd,de)
        cq2=sd[f"transformer.h.{b}.attn.c_q2.weight"].reshape(nh,hd,de); ck2=sd[f"transformer.h.{b}.attn.c_k2.weight"].reshape(nh,hd,de)
        r1=[eff_rank(cq[h].T@ck[h]) for h in range(nh)]; r2=[eff_rank(cq2[h].T@ck2[h]) for h in range(nh)]
        rows.append({"block":b,"qk1_median_eff_rank":st.median(r1),"qk1_max":max(r1),"qk2_median_eff_rank":st.median(r2)})
    allmed=[r["qk1_median_eff_rank"] for r in rows]
    res={"status":"complete","analysis":"attention_qk_pattern_per_head_effective_rank","head_dim_bound":hd,
         "per_block":rows,"qk1_median_over_blocks":st.median(allmed),"min":min(allmed),"max":max(allmed),
         "mlp_operator_rank_reference_2675":"438-749 of 1152 (no bottleneck)",
         "interpretation":"attention pattern is head-dim(128)-bottlenecked and compressible (frontier §312); MLP token-context operators are full-1152 high-rank (not compressible, §2673/§2675/§2676)",
         "runtime_s":time.time()-t0}
    json.dump(res,open("attention_qk_pattern_rank_results.json","w"),indent=2)
    print(json.dumps({k:res[k] for k in ("qk1_median_over_blocks","min","max","head_dim_bound","runtime_s")},indent=2))
if __name__=="__main__": main()
