"""Full attention10 CPU consumer check and literal conditional runtime price."""
import json,sys,time,signal
from datetime import datetime,timezone
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from coupled_routing_value_mlp9_v1 import execute as source_execute
from coupled_attention10_ports_v1 import compile_program,normalized_ports,attention_write,execute,MAPS,EPS
sys.path.insert(0,str(P.parents[1]))
from jacclust.tt_model import CausalBilinearSelfAttention


@torch.no_grad()
def main():
    out=P/'COUPLED_ATTENTION10_PORTS_V1_RESULT.json';artifact=P/'COUPLED_ATTENTION10_PORTS_V1_PROGRAM.pt'
    assert not out.exists();signal.alarm(120);torch.set_num_threads(2);start=time.perf_counter()
    s=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    source=torch.load(P/'COUPLED_NATIVE_SUFFIX_V1_EXAMPLE_PROGRAM.pt',weights_only=True)
    row=json.loads((P/'SRO_ARTICLE_CORRECTION_V1_ROWS.json').read_text())['rows'][0]
    ids=torch.tensor([row['ids']]);n=ids.shape[1]
    x0=F.rms_norm(F.embedding(ids,s['transformer.wte.weight']).float(),(1152,),eps=EPS)
    l0=s['transformer.h.0.lambdas'].float()
    first_input=F.rms_norm((l0[0]+l0[1])*x0,(1152,),eps=EPS)
    first=F.linear(first_input,s['transformer.h.0.attn.c_v.weight'].float())
    prefix='transformer.h.10.attn.'
    w={name:s[prefix+name+'.weight'] for name in [*MAPS.values(),'c_proj']};w['lamb']=s[prefix+'lamb']
    lambdas=s['transformer.h.10.lambdas']
    program=compile_program(source,x0,lambdas,w,first)
    native=CausalBilinearSelfAttention(SimpleNamespace(n_head=9,n_embd=1152,squared_attn=True,bilinear_attn=True)).eval()
    native.load_state_dict({k[len(prefix):]:v for k,v in s.items() if k.startswith(prefix)})
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    records=[]
    for a in [-1,0,.5,1,2]:
        for b in [-1,0,.5,1,2]:
            y=source_execute(source,a,b);raw=lambdas[0].double()*y+lambdas[1].double()*x0.double()
            normalized=F.rms_norm(raw,(1152,),eps=EPS)
            direct={}
            for name,native_name in MAPS.items():
                v=(normalized@w[native_name].double().T).reshape(1,n,9,128)
                direct[name]=v if name=='v' else F.rms_norm(v,(128,),eps=EPS)
            folded=normalized_ports(program,a,b)
            reference=attention_write(direct,w['c_proj'],w['lamb'],first)
            pred=execute(program,a,b)
            native_write,_=native(F.rms_norm(raw.float(),(1152,),eps=EPS),first)
            records.append(dict(a=a,b=b,max_port_error=max(rel(folded[k],direct[k]) for k in MAPS),
                                write_error=rel(pred,reference),native_fp32_error=rel(pred,native_write.double())))
    payload=lambda d:sum(t.numel()*t.element_size() for t in d.values())
    compiled_bytes=payload(program)
    independent_bytes=payload(source)+x0.numel()*x0.element_size()+first.numel()*first.element_size()+payload(w)+lambdas.numel()*lambdas.element_size()
    if artifact.exists():
        # Recover the first attempt's metadata-only failure without overwriting
        # the already serialized program or accepting changed tensor values.
        saved=torch.load(artifact,weights_only=True)
        assert saved.keys()==program.keys() and all(torch.equal(saved[k],program[k]) for k in program)
    else:torch.save(program,artifact)
    # Formula scales the same per-token caches, not a new timing/fidelity experiment.
    fixed_compiled=payload({k:program[k] for k in ['output','mixture']})
    fixed_independent=payload(w)+source['bias'].numel()*source['bias'].element_size()+lambdas.numel()*lambdas.element_size()
    per_compiled=(compiled_bytes-fixed_compiled)//n;per_independent=(independent_bytes-fixed_independent)//n
    break_even=[t for t in range(1,257) if fixed_compiled+t*per_compiled<fixed_independent+t*per_independent]
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(max(r['max_port_error'],r['write_error'])<=1e-10 for r in records),
                pred_b=all(r['native_fp32_error']<=1e-5 for r in records),pred_c=compiled_bytes<=.99*independent_bytes,
                maxima={k:max(r[k] for r in records) for k in ['max_port_error','write_error','native_fp32_error']},
                compiled_payload_bytes=compiled_bytes,independent_payload_bytes=independent_bytes,saving=1-compiled_bytes/independent_bytes,
                serialized_program_bytes=artifact.stat().st_size,native_weight_dtypes={k:str(v.dtype) for k,v in w.items()},
                output_cast_cache_bytes=program['output'].numel()*8,independent_all_attention_cast_cache_bytes=sum(v.numel()*8 for v in w.values()),
                length_price_formula=dict(compiled_fixed=fixed_compiled,compiled_per_token=per_compiled,independent_fixed=fixed_independent,independent_per_token=per_independent,
                                          maximum_saving_length_under257=max(break_even) if break_even else None),
                records=records,seconds=time.perf_counter()-start,
                scope='One native18-token context/25coupled cases, CPU attention10 only. Projected polynomial ports replace runtime QKV dense maps for this context; original source/model/context generators remain required. No global parameter saving, long-prefix win, whole-suffix or execution-speed claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}));signal.alarm(0)


if __name__=='__main__':main()
