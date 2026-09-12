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
    reports=[]
    for stem in sys.argv[1:]:
        artifact=P/(stem+'_ARTIFACT.pt');saved=torch.load(artifact,weights_only=True,map_location='cpu')
        for arm,atoms in enumerate(saved['atoms']):
            for pos in (7,0):
                with torch.no_grad():
                    gram=atom_gram(atoms);kernel=query_output_gram(cross_factors(atoms,*weights[pos]));scores=overlap(gram,kernel);m=operators(gram,kernel)
                    pairs=[(float(scores[i,j]),i,j) for i in range(9) for j in range(i+1,9)]
                    pairs.sort(reverse=True)
                    reports.append(dict(stem=stem,artifact_sha=digest(artifact),arm=arm,position=pos,top_pairs=pairs[:8],all_pair_overlaps=[v for v,i,j in pairs],head_capture=m.diagonal(dim1=-2,dim2=-1).sum(-1).tolist(),literal_shared=literal_shared(gram,kernel)))
    result=dict(reports=reports,seconds=time.perf_counter()-tic,scope='Within fitted source dictionary only; no global source overlap or circuit claim. Normalized operator overlap can emphasize weak projected head energy.')
    out=P/('SOURCE_ENERGY_OVERLAP_'+sys.argv[1]+'_RESULT.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
