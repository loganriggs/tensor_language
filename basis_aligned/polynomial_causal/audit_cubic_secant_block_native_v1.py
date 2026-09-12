"""CPU audit of projected source energy overlap, with no fitting."""
import json,time,sys
from cubic_secant_block_v1 import encode,gram_kernel
from shared_cubic_source_projection_v1 import capture as original_capture
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
    artifact=P/'SHARED_CUBIC_SOURCE_CONTINUE_V1_ARTIFACT.pt';atoms=torch.load(artifact,weights_only=True,map_location='cpu')['atoms'][1]
    reports=[];packed={}
    for tangent in (False,True):
        components,mix,info=encode(atoms,tangent=tangent);label='tangent_limit' if tangent else 'exact_secant';packed[label]=dict(components=components,mix=mix,**info)
        for pos in (7,0):
            with torch.no_grad():
                g,k=gram_kernel(components,mix,weights[pos]);cap=float(torch.linalg.solve(g,k.sum(0)).trace());old=float(original_capture(atoms,*weights[pos]));inv=torch.linalg.inv(g)
                e=(inv[None]@k@inv[None]).diagonal(dim1=-2,dim2=-1)*g.diagonal()[None]
                reports.append(dict(label=label,position=pos,capture=cap,original_capture=old,relative_capture_change=(cap-old)/old,gram_condition=float(torch.linalg.cond(g)),individual_energy_ratio=float(e.sum()/cap),block_feature_head_energies=e[:,-2:].tolist(),separation=info['separation']))
    out=P/'CUBIC_SECANT_BLOCK_NATIVE_V1_RESULT.json';art=P/'CUBIC_SECANT_BLOCK_NATIVE_V1_ARTIFACT.pt';assert not out.exists() and not art.exists();torch.save(packed,art)
    result=dict(reports=reports,source_artifact_sha=digest(artifact),artifact_sha=digest(art),seconds=time.perf_counter()-tic,scope='Exact secant is a basis change; tangent limit changes the source function space. Private query writers eliminated again in each space, no reader fitting.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
