#!/usr/bin/env python3
"""Do the 18 MLP blocks share a low-dim INPUT subspace (a cross-block dictionary)? (CPU, exact). Extends §2675/§2679.

Per-block input subspace = top-k eigenvectors of (L_b^T L_b + R_b^T R_b) (the input-importance metric). Test
pairwise principal-angle overlap (mean singular value of U_a^T U_b; 1=identical, ~k/d for random) and the
effective rank of the COMBINED input-importance sum_b(L_b^T L_b+R_b^T R_b). If blocks share a low-dim input
dictionary, overlaps >> random and the combined rank << 1152. If near-orthogonal, MLPs read independently ->
no cross-block input compression. Noise-free, no forwards.
"""
import torch, json, time, itertools, statistics as st
BLOB="/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin"
def eff_rank(vals):
    p=vals/vals.sum(); return float(torch.exp(-(p*torch.log(p+1e-30)).sum()))
def main():
    t0=time.time(); sd=torch.load(BLOB,map_location="cpu",weights_only=False)
    if hasattr(sd,'state_dict'): sd=sd.state_dict()
    d=1152; K=64
    Ms={}; subs={}
    comb=torch.zeros(d,d,dtype=torch.float64)
    for b in range(18):
        L=sd[f"transformer.h.{b}.mlp.Left.weight"].double(); R=sd[f"transformer.h.{b}.mlp.Right.weight"].double()
        M=L.T@L+R.T@R; Ms[b]=M; comb=comb+M
        w,V=torch.linalg.eigh(M); subs[b]=V[:,-K:]
    ovs=[float(torch.mean(torch.linalg.svdvals(subs[a].T@subs[c]))) for a,c in itertools.combinations(range(18),2)]
    # random baseline
    rnd=[]
    for _ in range(20):
        g=torch.linalg.qr(torch.randn(d,K,dtype=torch.float64))[0]; g2=torch.linalg.qr(torch.randn(d,K,dtype=torch.float64))[0]
        rnd.append(float(torch.mean(torch.linalg.svdvals(g.T@g2))))
    comb_rank=eff_rank(torch.linalg.eigvalsh(comb).flip(0).clamp(min=0))
    res={"status":"complete","analysis":"mlp_cross_block_input_subspace_sharing","d_model":d,"top_k":K,
         "pairwise_overlap_mean":st.mean(ovs),"pairwise_overlap_min":min(ovs),"pairwise_overlap_max":max(ovs),
         "random_baseline_mean":st.mean(rnd),"combined_input_effective_rank":comb_rank,
         "verdict":("shared_low_dim_input_dictionary" if st.mean(ovs) > 2*st.mean(rnd) else
                    "blocks_read_near_orthogonal_inputs_no_shared_dictionary"),
         "runtime_s":time.time()-t0}
    json.dump(res,open("mlp_cross_block_input_sharing_results.json","w"),indent=2)
    print(json.dumps({k:res[k] for k in ("verdict","pairwise_overlap_mean","pairwise_overlap_max","random_baseline_mean","combined_input_effective_rank","runtime_s")},indent=2))
if __name__=="__main__": main()
