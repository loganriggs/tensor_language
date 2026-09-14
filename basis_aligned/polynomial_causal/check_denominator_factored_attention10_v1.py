"""Short/long native-context exact basis check, retaining failed long prices."""
from datetime import datetime,timezone
from types import SimpleNamespace
import json,sys,time,signal,torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from coupled_routing_value_mlp9_v1 import execute as source_execute
from coupled_attention10_ports_v1 import compile_program as expanded,normalized_ports as expanded_ports,MAPS,EPS,EXPS9,EXPS4
from coupled_attention10_ports_v2 import execute as expanded_execute
from denominator_factored_attention10_v1 import compile_program,normalized_ports,execute
sys.path.insert(0,str(P.parents[1]))
from jacclust.tt_model import CausalBilinearSelfAttention


@torch.no_grad()
def main():
    out=P/'DENOMINATOR_FACTORED_ATTENTION10_V1_RESULT.json';artifact=P/'DENOMINATOR_FACTORED_ATTENTION10_V1_SHORT_PROGRAM.pt'
    assert not out.exists() and not artifact.exists();torch.set_num_threads(2);signal.alarm(120);start=time.perf_counter()
    s=torch.load(CHECKPOINT,weights_only=True,mmap=True);prefix='transformer.h.10.attn.'
    w={name:s[prefix+name+'.weight'] for name in [*MAPS.values(),'c_proj']};w['lamb']=s[prefix+'lamb']
    lambdas=s['transformer.h.10.lambdas']
    native=CausalBilinearSelfAttention(SimpleNamespace(n_head=9,n_embd=1152,squared_attn=True,bilinear_attn=True)).eval()
    native.load_state_dict({k[len(prefix):]:v for k,v in s.items() if k.startswith(prefix)})
    payload=lambda p:sum(t.numel()*t.element_size() for t in p.values())
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    # Polynomial coefficient columns: rho=(1+a^2)(1+b^2); old13 rank12,
    # because rho*B0 duplicates a combination of the nine monomial columns.
    rho_coeff={(0,0):1,(2,0):1,(0,2):1,(2,2):1}
    design=torch.zeros(16,13,dtype=torch.float64)
    for k,(i,j) in enumerate(EXPS4):
        for (u,v),c in rho_coeff.items():design[4*(i+u)+j+v,k]+=c
    for k,(i,j) in enumerate(EXPS9):design[4*i+j,4+k]=1
    ranks=[int(torch.linalg.matrix_rank(matrix,tol=1e-10)) for matrix in [design,design[:,1:]]]
    records=[];prices=[]
    cases=[('short','COUPLED_ATTENTION10_NATIVE_V2_EXAMPLE_PROGRAM.pt','SRO_ARTICLE_CORRECTION_V1_ROWS.json'),
           ('long','COUPLED_NATURAL_LENGTH_V1_EXAMPLE_PROGRAM.pt','COUPLED_NATURAL_LENGTH_V1_ROWS.json')]
    for label,filename,rowfile in cases:
        saved=torch.load(P/filename,weights_only=True);source=saved['source'];first=saved['attention']['first_values']
        row=json.loads((P/rowfile).read_text())['rows'][0];ids=torch.tensor([row['ids']]);n=ids.shape[1]
        x0=F.rms_norm(F.embedding(ids,s['transformer.wte.weight']).float(),(1152,),eps=EPS)
        old=expanded(source,x0,lambdas,w,first);new=compile_program(source,x0,lambdas,w,first)
        for a in [-1,0,.5,1,2]:
            for b in [-1,0,.5,1,2]:
                op,np=expanded_ports(old,a,b),normalized_ports(new,a,b)
                actual=execute(new,a,b);reference=expanded_execute(old,a,b)
                y=source_execute(source,a,b);raw=lambdas[0].double()*y+lambdas[1].double()*x0.double()
                native_output,_=native(F.rms_norm(raw.float(),(1152,),eps=EPS),first)
                records.append(dict(context=label,a=a,b=b,port_error=max(rel(np[k],op[k]) for k in MAPS),
                                    write_error=rel(actual,reference),native_fp32_error=rel(actual,native_output.double())))
        combined=payload(source)+payload(new)+x0.numel()*x0.element_size()
        independent=payload(source)+payload(w)+x0.numel()*x0.element_size()+first.numel()*first.element_size()+lambdas.numel()*lambdas.element_size()
        prices.append(dict(context=label,tokens=n,expanded_consumer_bytes=payload(old),factored_consumer_bytes=payload(new),
                           consumer_ratio=payload(new)/payload(old),combined_bytes=combined,independent_bytes=independent,combined_ratio=combined/independent))
        if label=='short':torch.save(new,artifact)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(max(r['port_error'],r['write_error'])<=1e-10 for r in records) and ranks==[12,12],
                pred_b=all(r['consumer_ratio']<=.85 for r in prices),pred_c=all(r['native_fp32_error']<=1e-5 for r in records),
                design_ranks=ranks,maxima={k:max(r[k] for r in records) for k in ['port_error','write_error','native_fp32_error']},
                prices=prices,records=records,seconds=time.perf_counter()-start,
                scope='Direct weight/context compilation of12vector basis; no inverse or fitted outputs. Shared denominator retained, no interactions dropped. Two CPU native-context fixtures/50grid cases, not new whole-suffix validation. Original generators remain; long price failure reported.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}));signal.alarm(0)


if __name__=='__main__':main()
