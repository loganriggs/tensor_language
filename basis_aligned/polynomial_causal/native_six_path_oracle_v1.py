"""Exact source-pair expansion, native entire-output MLP17, no sparse-fit claim.
A six unordered terms sum to native FP64 quadratic relative<=1e-10.
B mixed pair equals source-addition finite difference relative<=1e-10.
All1152 outputs imply every linear unembedding row, no vocab tensor materialized.
"""
import json
from pathlib import Path
import torch
from native_suffix_mlp16_producer_v1 import digest


def source_pairs(sources,left,right,down):
    """sources [...,source,d]; return dictionary of unordered output interactions."""
    l=sources@left.T;r=sources@right.T
    return {(i,j):((l[...,i,:]*r[...,j,:])+(l[...,j,:]*r[...,i,:] if i!=j else 0))@down.T
            for i in range(sources.shape[-2]) for j in range(i,sources.shape[-2])}


@torch.no_grad()
def main():
    torch.set_num_threads(2);p=Path(__file__).parent;out=p/'NATIVE_SIX_PATH_ORACLE_V1.json';assert not out.exists()
    bind=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_BINDING.json').read_text())['files'];assert all(digest(k)==v for k,v in bind.items())
    state=torch.load(next(k for k in bind if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    left,right,down=[state['transformer.h.17.mlp.'+n+'.weight'].double() for n in ('Left','Right','Down')]
    up=torch.load(p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt',weights_only=True,map_location='cpu')
    receipt=json.loads((p/'NATIVE_SUFFIX_UPSTREAM_V1_RESULT.json').read_text());assert digest(p/'NATIVE_SUFFIX_UPSTREAM_V1_PORTS.pt')==receipt['cache_sha256']
    cache=torch.load(p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')
    r=up['ports']['residual'].double();producer=up['block17_lambdas'][0].double()*up['ports']['mlp16_output'].double()
    pre=cache['ports']['pre'].double();sources=torch.stack([r-producer,producer,pre-r],1)
    parts=source_pairs(sources,left,right,down)
    q=lambda x:((x@left.T)*(x@right.T))@down.T
    target=q(pre);error=float((sum(parts.values())-target).norm()/target.norm());mixed=[]
    for (i,j),value in parts.items():
        if i!=j:
            ref=q(sources[:,i]+sources[:,j])-q(sources[:,i])-q(sources[:,j])
            mixed.append(float((value-ref).norm()/ref.norm()))
    result=dict(pred_a=error<=1e-10,pred_b=max(mixed)<=1e-10,sum_relative_error=error,mixed_relative_errors=mixed,
        source_names=['remaining_incoming_residual','scaled_mlp16_output','attention17_output'],endpoints=len(pre),output_coordinates=1152,
        price=dict(native_products=4608,explicit_pair_terms=6,no_persisted_dense_tensor=True,source_linear_reads=3*2*4608,
                   expanded_pair_product_multiplications=9*4608,factored_execution_product_multiplications=4608),
        source_script_sha256=digest(__file__),scope='Exact all-output polynomial numerator identity on cached sources. Shared input RMS squared, Down bias, direct residual and final RMS/tanh remain explicit. No fitted sparsity, independent source realizability, or behavioral circuit claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
