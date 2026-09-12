"""CPU audit of projected source energy overlap, with no fitting."""
import json,time,sys
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
    a=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'];b=torch.load(P/'SHARED_CUBIC_SOURCE_CONTINUE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'][0]
    ia=[14,15];ib=[11,2];allatoms=torch.cat([a,b]);g=atom_gram(allatoms);ga,gb,gab=g[:16,:16],g[16:,16:],g[:16,16:];inva=torch.linalg.inv(ga);invb=torch.linalg.inv(gb);reports=[]
    for pos in (7,0):
        with torch.no_grad():
            k=query_output_gram(cross_factors(allatoms,*weights[pos])).sum(0);aa=inva@k[:16,:16]@inva;bb=invb@k[16:,16:]@invb;ab=inva@k[:16,16:]@invb
            na=float((ga[ia][:,ia]*aa[ia][:,ia]).sum());nb=float((gb[ib][:,ib]*bb[ib][:,ib]).sum());inner=float((gab[ia][:,ib]*ab[ia][:,ib]).sum());error=na+nb-2*inner
            reports.append(dict(position=pos,target_norm_squared=na,other_fit_norm_squared=nb,inner_product=inner,function_cosine=inner/(na*nb)**.5,relative_norm_error=max(error/na,0)**.5))
    result=dict(reports=reports,all_errors_below10pct=all(r['relative_norm_error']<=.1 for r in reports),scope='Both independent fits keep their own full-dictionary private query writers; selected block only, head-separated coefficient metric with fixed reference gates, not native text behavior.')
    out=P/'REGIONAL_SOURCE_CROSS_START_FUNCTION_V1_AUDIT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
