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
    pack=torch.load(P/'CUBIC_SECANT_BLOCK_NATIVE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');base=pack['exact_secant'];simple=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms']
    candidates=dict(tangent=pack['tangent_limit'],common_quadratic=dict(components=simple,mix=torch.eye(16,dtype=torch.float64)));reports=[]
    for name,candidate in candidates.items():
        a,b=base['components'],candidate['components'];mix=torch.block_diag(base['mix'],candidate['mix']);allatoms=torch.cat([a,b]);g=mix@atom_gram(allatoms)@mix.T;g1,g2,g12=g[:16,:16],g[16:,16:],g[:16,16:]
        for pos in (7,0):
            with torch.no_grad():
                k=(mix[None]@query_output_gram(cross_factors(allatoms,*weights[pos]))@mix.T[None]).sum(0)
                n1=float(torch.linalg.solve(g1,k[:16,:16]).trace());n2=float(torch.linalg.solve(g2,k[16:,16:]).trace());cross=torch.linalg.solve(g1,k[:16,16:])@torch.linalg.inv(g2);inner=float((g12*cross).sum());error=n1+n2-2*inner
                reports.append(dict(candidate=name,position=pos,reference_norm_squared=n1,candidate_norm_squared=n2,inner_product=inner,squared_error=error,relative_squared_error=error/n1,relative_norm_error=max(error/n1,0)**.5))
    result=dict(reports=reports,scope='Exact coefficient-space projected-function comparison; roundoff may make tiny squared errors signed. No native normalization or text-distribution guarantee.')
    out=P/'CUBIC_SOURCE_BLOCK_FUNCTION_COMPARISON_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
