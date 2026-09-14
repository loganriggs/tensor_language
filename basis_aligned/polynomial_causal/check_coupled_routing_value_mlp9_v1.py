"""Weight-derived coupled-port compilation, direct native-operation checks."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import json,sys,time,signal,torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from current_remainder_crossed_v1 import CurrentRemainder
from sro_mlp9_rational_v1 import prepare,execute as generic_execute,EPS
from coupled_routing_value_mlp9_v1 import compile_program,execute
sys.path.insert(0,str(P.parents[1]))
from jacclust.tt_model import CausalBilinearSelfAttention


@torch.no_grad()
def main():
    out=P/'COUPLED_ROUTING_VALUE_MLP9_V1_RESULT.json';assert not out.exists()
    signal.alarm(120);torch.set_num_threads(2);start=time.perf_counter()
    state=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    graph=CurrentRemainder(torch.load(P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True))
    attn=CausalBilinearSelfAttention(SimpleNamespace(n_head=9,n_embd=1152,squared_attn=True,bilinear_attn=True)).eval()
    prefix='transformer.h.9.attn.'
    attn.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    left,right,down=[state['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    bias=state['transformer.h.9.mlp.Down_bias'].double()
    rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    records=[];prices=[];serialized={}
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    for i in [0,18,36,54]:
        n=len(rows[i]['ids']);assert len(rows[i^1]['ids'])==n
        ids=torch.tensor([rows[i]['ids']]);raw=cache['raw9'][0,i,:n][None].float()
        x=F.rms_norm(raw,(1152,),eps=EPS)
        donor=F.rms_norm(cache['raw9'][0,i^1,:n][None].float(),(1152,),eps=EPS)
        x0=F.rms_norm(F.embedding(ids,state['transformer.wte.weight']).float(),(1152,),eps=EPS)
        lam=state['transformer.h.0.lambdas'].float()
        first_input=F.rms_norm((lam[0]+lam[1])*x0,(1152,),eps=EPS)
        first=F.linear(first_input,state['transformer.h.0.attn.c_v.weight'].float())
        attention,_=attn(x,first);z=(raw+attention).double()
        parts=graph.changes(graph.routing(x),graph.values(x),graph.routing(donor),graph.values(donor))
        directions=torch.stack([-graph.write(parts[k]) for k in ['routing','values','mixed']],-2)
        generic=prepare(z,directions,left,right,down,bias);coupled=compile_program(generic)
        def direct(a,b,dtype):
            gains=torch.tensor([a,b,a*b],dtype=dtype)
            edited=z.to(dtype)-(directions.to(dtype)*gains[None,None,:,None]).sum(-2)
            if dtype==torch.float32:
                normalized=F.rms_norm(edited,(1152,),eps=EPS)
                return edited+((normalized@left.float().T)*(normalized@right.float().T))@down.float().T+bias.float()
            rho=edited.square().mean(-1,keepdim=True)+EPS
            return edited+((edited@left.T)*(edited@right.T))@down.T/rho+bias
        base=direct(0,0,torch.float64);cb=execute(coupled,0,0)
        for a in [-1,0,.5,1,2]:
            for b in [-1,0,.5,1,2]:
                pred=execute(coupled,a,b);ref=direct(a,b,torch.float64);change=ref-base
                own=rel(pred-cb,change) if bool(change.norm()) else float((pred-cb).abs().max())
                records.append(dict(row=i,a=a,b=b,state_error=rel(pred,ref),own_change_error=own,
                                    generic_error=rel(pred,generic_execute(generic,[a,b,a*b])),
                                    native_fp32_error=rel(pred,direct(a,b,torch.float32).double())))
        sizes=[sum(t.numel()*t.element_size() for t in program.values()) for program in [generic,coupled]]
        prices.append(dict(row=i,tokens=n,generic_bytes=sizes[0],coupled_bytes=sizes[1],saving=1-sizes[1]/sizes[0]))
        if i==0:
            for name,program in [('GENERIC',generic),('COUPLED',coupled)]:
                f=P/('COUPLED_ROUTING_VALUE_MLP9_V1_'+name+'_EXAMPLE.pt');assert not f.exists()
                torch.save(program,f);serialized[name]=f.stat().st_size
    result=dict(utc=datetime.now(timezone.utc).isoformat(),
                pred_a=all(max(r['state_error'],r['generic_error'])<=1e-10 and r['own_change_error']<=1e-8 for r in records),
                pred_b=all(r['native_fp32_error']<=1e-5 for r in records),
                pred_c=all(p['saving']>=.07 for p in prices),
                maxima={k:max(r[k] for r in records) for k in ['state_error','generic_error','own_change_error','native_fp32_error']},
                records=records,prices=prices,serialized_example_bytes=serialized,
                original_mlp_scalars=left.numel()+right.numel()+down.numel()+bias.numel(),seconds=time.perf_counter()-start,
                scope='Four old cached paired contexts/100 coupled strength cases. Nine numerator and denominator coefficients replace ten only when mixed strength equals a*b. Original native generators/MLP remain; no fit, quantization or static whole-model saving.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}));signal.alarm(0)


if __name__=='__main__':main()
