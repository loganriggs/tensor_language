"""Frozen weights-only key-span compilation, then CPU native-cache replay."""
import json,time,hashlib,importlib.util,signal
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from key_span_reuse_v1 import KeySpanParent
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    out=P/'KEY_SPAN_REUSE_V1_RESULT.json';assert not out.exists();signal.alarm(120)
    torch.set_num_threads(2);torch.manual_seed(170234000);start=time.perf_counter()
    original_path=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt'
    original=torch.load(original_path,weights_only=True)
    root=P/'extracted_circuits/sparse_even_key_producers_8_2_9_8_v1'
    spec=importlib.util.spec_from_file_location('original_executor',root/'execute.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    sparse=old.load_program(root/'program.pt')
    candidate={k:v.clone() for k,v in original.items() if k!='key_basis'}
    coordinates=[];spans=[]
    for i in range(2):
        k=torch.cat([original['k1'][i],original['k2'][i]]).double()
        b=original['key_basis'][i].double();c=torch.linalg.lstsq(k.T,b).solution
        bs=sparse['key_basis'][i];cs=torch.linalg.lstsq(k.T,bs).solution
        spans.append(dict(index=i,original_residual=float((k.T@c-b).norm()/b.norm()),
                          sparse_residual=float((k.T@cs-bs).norm()/bs.norm()),condition=float(torch.linalg.cond(k))))
        coordinates.append(c.clone())
    candidate['key_coordinates']=torch.stack(coordinates)
    program=P/'KEY_SPAN_REUSE_V1_PROGRAM.pt';assert not program.exists();torch.save(candidate,program)
    sha=hashlib.sha256(program.read_bytes()).hexdigest()
    new=KeySpanParent(torch.load(program,weights_only=True));checks=[]
    for i in [0,1]:
        x=torch.randn(2,32,1152);ids=torch.randint(0,50000,(2,32))
        a=old.scalar(x,ids,original,i);b=new.scalar(x,ids,i)
        checks.append(dict(inputs='independent',index=i,relative=float((a-b).norm()/a.norm())))
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    for context in [0,1]:
        x=F.rms_norm(cache['raw9'][context].float(),(1152,),eps=torch.finfo(torch.float32).eps)
        ids=torch.zeros(x.shape[:2],dtype=torch.long)
        a=old.scalar(x,ids,original,1);b=new.scalar(x,ids,1)
        checks.append(dict(inputs='cached_native',context=context,relative=float((a-b).norm()/a.norm())))
    payload=lambda p:sum(v.numel()*v.element_size() for v in p.values())
    result=dict(utc=datetime.now(timezone.utc).isoformat(),spans=spans,checks=checks,
                pred_a=max([s['original_residual'] for s in spans]+[c['relative'] for c in checks])<=1e-10,
                pred_b=program.stat().st_size<min(original_path.stat().st_size,(root/'program.pt').stat().st_size),
                program_sha=sha,program_bytes=program.stat().st_size,original_file_bytes=original_path.stat().st_size,
                sparse_file_bytes=(root/'program.pt').stat().st_size,original_payload_bytes=payload(original),candidate_payload_bytes=payload(candidate),
                original_common_fp32_payload=sum(v.numel()*4 for v in original.values()),candidate_common_fp32_payload=sum(v.numel()*4 for v in candidate.values()),
                resident_adapter_bytes=sum(v.numel()*v.element_size() for v in new.adapters),seconds=time.perf_counter()-start,
                scope='Exact original unsparsified parent; native normalized contexts/full QK maps/values/writers/suffix remain. Coordinates derived from weights before cached scoring. Sparse basis residual descriptive only, no projection rescue or whole-model saving.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result));signal.alarm(0)


if __name__=='__main__':main()
