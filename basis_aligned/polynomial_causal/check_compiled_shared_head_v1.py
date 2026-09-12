"""Native-weight compilation identity at a declared normalized-state interface.

Seed1450,16 independent input pairs,positions0/1/2/4/8/16; relative<=1e-10.
No native text generalization or upstream-closure claim from these probes.
"""
import json
from pathlib import Path
import torch
from compiled_shared_head_v1 import compile_head,execute_head
from shared_cubic_source_projection_v1 import cross_factors
from common_quadratic_ports_v1 import source_ports,private_writers
from factorial_source_ports_v1 import write
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(1450)
    bind=json.loads((P/'REGIONAL_PRIVATE_CONSUMERS_V1_BINDING.json').read_text())['files']
    sd=torch.load(next(x for x in bind if x.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def w(n):return sd['transformer.h.17.attn.'+n+'.weight'].double().reshape(9,128,1152)[2]
    q1,k1,q2,k2=[w(n) for n in ('c_q','c_k','c_q2','c_k2')]
    lam=float(sd['transformer.h.17.attn.lamb']);value=torch.cat([(1-lam)*w('c_v'),lam*sd['transformer.h.0.attn.c_v.weight'].double().reshape(9,128,1152)[2]],-1)
    output=sd['transformer.h.17.attn.c_proj.weight'].double().reshape(1152,9,128)[:,2]
    atoms=torch.load(P/'COMMON_QUADRATIC_SOURCE_BLOCK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['atoms'].double()
    program=compile_head(atoms,q1,k1,q2,k2,value,output);eps=torch.finfo(torch.float32).eps
    query=torch.randn(16,1152,dtype=torch.float64);source=torch.randn(16,2304,dtype=torch.float64)
    qa=query@q1.T;qb=query@q2.T;ka=source[:,:1152]@k1.T;kb=source[:,:1152]@k2.T
    gate=1/(128**2*((qa.square().mean(-1)+eps)*(qb.square().mean(-1)+eps)*(ka.square().mean(-1)+eps)*(kb.square().mean(-1)+eps)).sqrt())
    parent,children=source_ports(source,atoms);cells=[]
    for position in (0,1,2,4,8,16):
        rot=rotary(position,128);keys=[torch.cat([rot@k,torch.zeros_like(k)],-1)[None] for k in (k1,k2)]
        factors=cross_factors(atoms,q1[None],keys[0],q2[None],keys[1],value[None],output[:,None])
        reference=write(gate[:,None],parent,children,private_writers(query,atoms,factors))
        actual=execute_head(query,source,rot,program)
        cells.append(dict(position=position,relative_error=float((actual-reference).norm()/reference.norm())))
    scalar_count=sum(t.numel() for t in program.values());assert scalar_count==666656
    result=dict(passed=max(c['relative_error'] for c in cells)<=1e-10,cells=cells,stored_scalars=scalar_count,fp32_storage_bytes=4*scalar_count,
        parts={k:v.numel() for k,v in program.items()},
        scope='Exact weight-derived selected shared-head program. Requires query1152,source2304 and fixed rounded relative rotary map. No original attention module or inverse solve at runtime. Upstream state generation, final suffix/unembedding and positional-map generation are outside this package and remain required.')
    torch.save(program,P/'COMPILED_SHARED_HEAD2_V1_ARTIFACT.pt')
    (P/'COMPILED_SHARED_HEAD2_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
