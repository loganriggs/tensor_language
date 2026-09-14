"""Registered exact three-branch MLP9 response and RMS-origin triple control."""
import json,sys,time,signal
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from even_value_shared_graph_v1 import SharedGraph
from even_value_factorial_v1 import coefficients
from sro_mlp9_rational_v1 import prepare,execute,EPS
sys.path.insert(0,str(P.parents[1]))
from jacclust.tt_model import CausalBilinearSelfAttention


@torch.no_grad()
def main():
    out=P/'SRO_MLP9_RATIONAL_V1_RESULT.json';assert not out.exists()
    signal.alarm(120);torch.set_num_threads(2);tic=time.perf_counter()
    state=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    p=torch.load(P/'EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt',weights_only=True)
    graph=SharedGraph(p)
    attn=CausalBilinearSelfAttention(SimpleNamespace(n_head=9,n_embd=1152,squared_attn=True,bilinear_attn=True)).eval()
    prefix='transformer.h.9.attn.'
    attn.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    weights=[state['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    bias=state['transformer.h.9.mlp.Down_bias'].double();left,right,down=weights
    rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    amplitudes=[[int(bool(mask&bit)) for bit in [1,2,4]] for mask in range(8)]+[[-1,.5,1.5],[.25,.75,1.25],[2,-1,0]]
    records=[];triples=[];prices=[]
    def direct(z,a,dirs,dtype,freeze=False):
        z=z.to(dtype);dirs=dirs.to(dtype);a=torch.tensor(a,dtype=dtype)
        edited=z-(dirs*a[None,None,:,None]).sum(-2)
        rho=z.square().mean(-1,keepdim=True)+EPS if freeze else edited.square().mean(-1,keepdim=True)+EPS
        l,r,d=[v.to(dtype) for v in weights]
        return edited+((edited@l.T)*(edited@r.T))@d.T/rho+bias.to(dtype)
    error=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
    for i in [0,18,36,54]:
        row=rows[i];n=len(row['ids']);ids=torch.tensor([row['ids']])
        raw=cache['raw9'][0,i,:n][None].float()
        x=F.rms_norm(raw,(1152,),eps=EPS)
        x0=F.rms_norm(F.embedding(ids,state['transformer.wte.weight']).float(),(1152,),eps=EPS)
        lambdas=state['transformer.h.0.lambdas'].float()
        first_input=F.rms_norm((lambdas[0]+lambdas[1])*x0,(1152,),eps=EPS)
        first=F.linear(first_input,state['transformer.h.0.attn.c_v.weight'].float())
        attention,_=attn(x,first);z=(raw+attention).double()
        branches=graph.state(x,first_values=first.reshape(1,n,9,128)[:,:,8])
        dirs=torch.stack([branch@p['output'].double().T for branch in branches],-2)
        program=prepare(z,dirs,*weights,bias)
        # The reference is evaluated only after weight/context-derived compilation.
        base=direct(z,[0,0,0],dirs,torch.float64)
        compiled_base=execute(program,[0,0,0])
        full_cube=[];fixed_cube=[]
        for j,a in enumerate(amplitudes):
            predicted=execute(program,a);reference=direct(z,a,dirs,torch.float64)
            change=reference-base
            own_error=error(predicted-compiled_base,change) if bool(change.norm()) else float((predicted-compiled_base).abs().max())
            # Native FP32 operation order: RMS-normalize before bilinear projections.
            edited=z.float()-(dirs.float()*torch.tensor(a,dtype=torch.float32)[None,None,:,None]).sum(-2)
            normalized=F.rms_norm(edited,(1152,),eps=EPS)
            native=edited+((normalized@left.float().T)*(normalized@right.float().T))@down.float().T+bias.float()
            records.append(dict(row=i,amplitudes=a,state_error=error(predicted,reference),
                                own_change_error=own_error,native_fp32_error=error(predicted,native.double())))
            if j<8:
                full_cube.append(reference)
                fixed_cube.append(direct(z,a,dirs,torch.float64,True))
        full=torch.stack(full_cube);fixed=torch.stack(fixed_cube)
        full_triple=coefficients(full)[7].norm();fixed_triple=coefficients(fixed)[7].norm()
        change=(full[7]-full[0]).norm()
        triples.append(dict(row=i,live_triple_to_change=float(full_triple/change),
                            frozen_triple_to_change=float(fixed_triple/change),
                            passed=float(fixed_triple/change)<=1e-10 and float(full_triple)>=100*float(fixed_triple) and float(full_triple/change)>=1e-8))
        prices.append(dict(row=i,tokens=n,prepared_bytes=sum(v.numel()*v.element_size() for v in program.values()),
                           prepared_scalars=sum(v.numel() for v in program.values())))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),
                pred_a=all(r['state_error']<=1e-10 and r['own_change_error']<=1e-8 for r in records),
                pred_b=all(r['native_fp32_error']<=1e-5 for r in records),pred_c=all(r['passed'] for r in triples),
                max_state_error=max(r['state_error'] for r in records),max_own_change_error=max(r['own_change_error'] for r in records),
                max_native_fp32_error=max(r['native_fp32_error'] for r in records),records=records,triples=triples,prices=prices,
                original_mlp_scalars=sum(t.numel() for t in weights)+bias.numel(),seconds=time.perf_counter()-tic,
                scope='Exact per-context rational compiler from weights and supplied three sourcewrites; no fitted response targets. Ten quadratic numerator vectors, ten scalar denominator coefficients, four residual vectors and bias retained. Original MLP weights, native attention/context generators remain charged. Four old cached contexts/44strengthcases; no independent text model, parameter compression or runtime speed claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','prices']},indent=2));signal.alarm(0)


if __name__=='__main__':main()
