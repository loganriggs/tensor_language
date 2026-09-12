"""Compile four first-state readings for every vocabulary ID, with no corpus.

Registered independent256-probe FP32 write replay<=1e-5; child partition<=1e-6.
Full vocabulary lookup is deterministic weight folding, not empirical fitting.
"""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from compiled_shared_head_v1 import execute_head
from compiled_token_shared_head_v1 import execute_token_head
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(1502);eps=torch.finfo(torch.float32).eps
    old=torch.load(P/'extracted_circuits/regional_shared_head2_v1/program.pt',weights_only=True,map_location='cpu')
    rows=torch.cat([old['parent'],old['children']]);binding=json.loads((P/'COMPILED_SHARED_HEAD2_NATIVE_V1_BINDING.json').read_text())['files']
    ck=next(x for x in binding if x.endswith('pytorch_model.bin'));sd=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
    embedding=sd['transformer.wte.weight'];lam=sd['transformer.h.0.lambdas'].float()
    def first(ids):
        x0=F.rms_norm(embedding[ids].float(),(1152,),eps=eps)
        return F.rms_norm(lam[0]*x0+lam[1]*x0,(1152,),eps=eps)
    table=torch.empty(len(embedding),4)
    for start in range(0,len(table),1024):
        ids=torch.arange(start,min(start+1024,len(table)));table[ids]=first(ids)@rows[:,1152:].T
    program={k:v.clone() for k,v in old.items() if k not in ('parent','children')}
    program.update(current_readers=rows[:,:1152].clone(),token_reads=table)
    ids=torch.randint(len(table),(256,));query=torch.randn(256,1152);current=torch.randn_like(query);source=torch.cat([current,first(ids)],-1)
    rot=rotary(7,128).float();reference=execute_head(query,source,rot,old);actual=execute_token_head(query,current,ids,rot,program)
    error=float((actual-reference).norm()/reference.norm());child=sum(execute_token_head(query,current,ids,rot,program,mask) for mask in ((1.,0.),(0.,1.)))
    partition=float((child-actual).norm()/actual.norm());scalars=sum(t.numel() for t in program.values());assert scalars==863264
    result=dict(passed=error<=1e-5 and partition<=1e-6,write_relative_error=error,child_partition_relative_error=partition,
                vocabulary_rows=len(table),lookup_scalars=table.numel(),old_package_scalars=sum(t.numel() for t in old.values()),new_package_scalars=scalars,
                scope='Complete vocabulary first-state four-read lookup plus identical query/current branch. Closes token-only first-state input; query and current-state generators, rotation and final suffix remain external. Random interface probes are not native text validation.')
    torch.save(program,P/'COMPILED_TOKEN_SHARED_HEAD2_V1_ARTIFACT.pt')
    (P/'COMPILED_TOKEN_SHARED_HEAD2_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
