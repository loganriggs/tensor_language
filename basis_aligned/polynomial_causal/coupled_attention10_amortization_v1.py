"""Matched FP64 query timing with native dense maps already cached."""
import json,time,signal,statistics
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from coupled_routing_value_mlp9_v1 import execute as source_execute
from coupled_attention10_ports_v1 import compile_program,attention_write,MAPS,EPS
from coupled_attention10_ports_v2 import execute as attention_execute


@torch.no_grad()
def main():
    out=P/'COUPLED_ATTENTION10_AMORTIZATION_V1_RESULT.json';assert not out.exists()
    signal.alarm(120);torch.set_num_threads(2)
    state=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    source0=torch.load(P/'COUPLED_NATIVE_SUFFIX_V1_EXAMPLE_PROGRAM.pt',weights_only=True)
    original_attention=torch.load(P/'COUPLED_ATTENTION10_PORTS_V1_PROGRAM.pt',weights_only=True)
    row=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows'][0]
    ids=torch.tensor([row['ids']]);n=ids.shape[1]
    x0base=F.rms_norm(F.embedding(ids,state['transformer.wte.weight']).float(),(1152,),eps=EPS).double()
    first0=original_attention['first_values'].double()
    prefix='transformer.h.10.attn.'
    # Strong baseline: no repeated dense-weight casts inside query measurements.
    weights={k:state[prefix+k+'.weight'].double() for k in [*MAPS.values(),'c_proj']}
    weights['lamb']=state[prefix+'lamb'].double();lambdas=state['transformer.h.10.lambdas'].double()
    grid=[(a,b) for a in [-1,0,.5,1,2] for b in [-1,0,.5,1,2]]
    records=[];prices=[]
    for batch in [1,8]:
        source={k:(v if k=='bias' else v.repeat(batch,*([1]*(v.ndim-1)))) for k,v in source0.items()}
        x0=x0base.repeat(batch,1,1);first=first0.repeat(batch,1,1)
        def prepare():
            program=compile_program(source,x0,lambdas,weights,first)
            # These static maps/values are shared, not copied per query.
            program['output']=weights['c_proj'];program['first_values']=first
            return program
        program=prepare()
        def native(a,b):
            y=source_execute(source,a,b);raw=lambdas[0]*y+lambdas[1]*x0
            normalized=F.rms_norm(raw,(1152,),eps=EPS);ports={}
            for name,native_name in MAPS.items():
                v=(normalized@weights[native_name].T).reshape(batch,n,9,128)
                ports[name]=v if name=='v' else F.rms_norm(v,(128,),eps=EPS)
            return torch.stack([y,attention_write(ports,weights['c_proj'],weights['lamb'],first)])
        def compiled(p,a,b):return torch.stack([source_execute(source,a,b),attention_execute(p,a,b)])
        prices.append(dict(batch=batch,source_bytes=sum(v.numel()*v.element_size() for v in source.values()),
                           cached_dense_attention_bytes=sum(v.numel()*v.element_size() for v in weights.values()),
                           attention_program_resident_bytes=sum(v.numel()*v.element_size() for v in program.values()),
                           shared_output_bytes=weights['c_proj'].numel()*8,shared_first_bytes=first.numel()*8))
        for queries in [1,8,32]:
            strengths=[grid[j%25] for j in range(queries)]
            def dense():return torch.stack([native(a,b) for a,b in strengths])
            def cold():
                p=prepare();return torch.stack([compiled(p,a,b) for a,b in strengths])
            def warm():return torch.stack([compiled(program,a,b) for a,b in strengths])
            funcs=[dense,cold,warm];reference=dense();errors=[]
            for fn in [cold,warm]:
                actual=fn();errors.append(float((actual-reference).norm()/reference.norm()))
            del reference,actual
            samples=[[],[],[]]
            for trial in range(3):
                for j in [(trial+k)%3 for k in range(3)]:
                    tic=time.perf_counter();value=funcs[j]();samples[j].append(time.perf_counter()-tic);del value
            medians=[statistics.median(s) for s in samples]
            record=dict(batch=batch,queries=queries,max_error=max(errors),median_seconds=medians,
                        cold_speedup=medians[0]/medians[1],warm_speedup=medians[0]/medians[2],samples=samples)
            records.append(record)
            print(json.dumps({k:v for k,v in record.items() if k!='samples'}),flush=True)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(r['max_error']<=1e-10 for r in records),
                pred_b=all(r['cold_speedup']>=1.1 for r in records if r['queries']==32),records=records,prices=prices,
                arm_order=['cached_dense','compiled_including_attention_preparation','compiled_warm'],
                scope='FP64 CPU/two threads, one18-token prefix repeated atbatch8. Three interleaved trials. Source program/context generation common and excluded; cold includes all attention port preparation. Dense weights cached before timing. Returns both residual source state and attention write. No whole-model/GPU speed or static weight saving claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','prices']}));signal.alarm(0)


if __name__=='__main__':main()
