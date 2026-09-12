"""CPU audit of projected source energy overlap, with no fitting."""
import json,time,sys
from cubic_secant_block_v1 import align_pair
from shared_cubic_source_projection_v1 import capture
from pathlib import Path
import torch
from folded_normalized_router_v1 import rotary
from sparse_path_stability_atlas_v1 import digest
from shared_cubic_source_projection_v1 import atom_gram,cross_factors,query_output_gram
from source_energy_overlap_v1 import overlap,operators,literal_shared
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);tic=time.perf_counter()
    binding=json.loads((P/'SHARED_CUBIC_SOURCE_NATIVE_V1_BINDING.json').read_text())['files']
    sd=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def w(k):return sd[k].double()
    q1,k1,q2,k2=[w('transformer.h.17.attn.'+key+'.weight').reshape(9,128,1152) for key in ('c_q','c_k','c_q2','c_k2')]
    mix=float(sd['transformer.h.17.attn.lamb']);v=torch.cat([(1-mix)*w('transformer.h.17.attn.c_v.weight'),mix*w('transformer.h.0.attn.c_v.weight')],-1).reshape(9,128,2304)
    g0=1/(q1.flatten(1).norm(dim=1)*k1.flatten(1).norm(dim=1)*q2.flatten(1).norm(dim=1)*k2.flatten(1).norm(dim=1))
    o=w('transformer.h.17.attn.c_proj.weight').reshape(1152,9,128)*g0[None,:,None];qr=rotary(8,128);weights={}
    for pos in (7,0):
        rotation=qr.T@rotary(pos,128);ka=torch.cat([torch.einsum('ab,hbd->had',rotation,k1),torch.zeros_like(k1)],-1);kb=torch.cat([torch.einsum('ab,hbd->had',rotation,k2),torch.zeros_like(k2)],-1)
        weights[pos]=(q1,ka,q2,kb,v,o)
    saved=torch.load(P/'SHARED_CUBIC_SOURCE_CONTINUE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');atoms=saved['atoms'][1]
    a=atoms[5];b=align_pair(a,atoms[6]);base=(a+b)/2;delta=(a-b)/2;axis=int(delta.norm(dim=-1).argmax());others=[r for r in range(16) if r not in (5,6)]
    pair=base.repeat(2,1,1);pair[1,axis]=delta[axis]/delta[axis].norm();collapsed=torch.cat([atoms[others],pair]);reports=[]
    for pos in (7,0):
        with torch.no_grad():
            original=float(capture(atoms,*weights[pos]));no_pair=float(capture(atoms[others],*weights[pos]));value=float(capture(collapsed,*weights[pos]));g=atom_gram(collapsed);k=query_output_gram(cross_factors(collapsed,*weights[pos]));inv=torch.linalg.inv(g);e=(inv[None]@k@inv[None]).diagonal(dim1=-2,dim2=-1)*g.diagonal()[None]
            reports.append(dict(position=pos,original_capture=original,no_pair_capture=no_pair,collapsed_capture=value,relative_total_change=(value-original)/original,marginal_retained=(value-no_pair)/(original-no_pair),gram_condition=float(torch.linalg.cond(g)),block_head_energies=e[:,-2:].sum(-1).tolist()))
    out=P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_RESULT.json';art=P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt';assert not out.exists() and not art.exists();torch.save(dict(atoms=collapsed,shared_reader_indices=[i for i in range(3) if i!=axis],block_indices=[14,15]),art)
    result=dict(reports=reports,reader_cosines=(a*b).sum(-1).tolist(),variable_axis=axis,source_reader_vectors_before=48,source_reader_vectors_after_with_explicit_sharing=46,artifact_sha=digest(art),scope='Weight-only reader collapse, private writers recomputed, no optimization or native behavioral test.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
