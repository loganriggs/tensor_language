"""Cold intervention-query price against an already shared-projection baseline."""
import json,time,statistics,signal
from datetime import datetime,timezone
import torch
from check_even_key_value_bank_v1 import P,CHECKPOINT
from sro_mlp9_rational_v1 import prepare,execute,EPS

STRENGTHS=[[int(bool(mask&bit)) for bit in [1,2,4]] for mask in range(8)]+[[-1,.5,1.5],[.25,.75,1.25],[2,-1,0]]


def prepare_cached(z,directions,left,right,down,bias):
    basis=torch.cat([z.unsqueeze(-2),-directions],-2)
    return dict(basis=basis,left=basis@left.T,right=basis@right.T,
                gram=(basis@basis.transpose(-1,-2))/z.shape[-1],bias=bias,down=down)


def cached_execute(p,a):
    c=torch.tensor([1.,*a],dtype=p['basis'].dtype)
    left=torch.einsum('...kh,k->...h',p['left'],c)
    right=torch.einsum('...kh,k->...h',p['right'],c)
    rho=torch.einsum('i,...ij,j->...',c,p['gram'],c)+EPS
    residual=torch.einsum('...kd,k->...d',p['basis'],c)
    return residual+(left*right)@p['down'].T/rho[...,None]+p['bias']


@torch.no_grad()
def main():
    out=P/'SRO_MLP9_AMORTIZATION_V1_RESULT.json';assert not out.exists()
    signal.alarm(120);torch.set_num_threads(2)
    stored=torch.load(P/'SRO_MLP9_SUFFIX_NATIVE_V1_EXAMPLE_PROGRAM.pt',weights_only=True)
    state=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    left,right,down=[state['transformer.h.9.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    bias=state['transformer.h.9.mlp.Down_bias'].double()
    base=stored['linear_basis'];z0=base[...,0,:];directions0=-base[...,1:,:]
    records=[];prices=[]
    for batch in [1,8]:
        z=z0.repeat(batch,1,1);directions=directions0.repeat(batch,1,1,1)
        prepared=prepare(z,directions,left,right,down,bias)
        cached=prepare_cached(z,directions,left,right,down,bias)
        prices.append(dict(batch=batch,length=z.shape[1],compiled_prepared_bytes=sum(v.numel()*v.element_size() for v in prepared.values()),
            cached_context_bytes=sum(v.numel()*v.element_size() for k,v in cached.items() if k!='down'),
            cached_retained_down_bytes=down.numel()*down.element_size(),
            common_original_mlp_bytes=sum(v.numel()*v.element_size() for v in [left,right,down,bias])))
        for queries in [1,8,32]:
            amps=[STRENGTHS[i%len(STRENGTHS)] for i in range(queries)]
            def compiled_cold():
                program=prepare(z,directions,left,right,down,bias)
                return torch.stack([execute(program,a) for a in amps])
            def cached_cold():
                program=prepare_cached(z,directions,left,right,down,bias)
                return torch.stack([cached_execute(program,a) for a in amps])
            def compiled_warm():return torch.stack([execute(prepared,a) for a in amps])
            def cached_warm():return torch.stack([cached_execute(cached,a) for a in amps])
            funcs=[cached_cold,compiled_cold,cached_warm,compiled_warm]
            values=[f() for f in funcs];ref=values[0]
            error=max(float((v-ref).norm()/ref.norm()) for v in values[1:])
            # Both cold paths contain their own preparation; alternate execution order.
            times=[[],[],[],[]]
            for trial in range(7):
                for j in [(trial+i)%4 for i in range(4)]:
                    start=time.perf_counter();funcs[j]();times[j].append(time.perf_counter()-start)
            med=list(map(statistics.median,times))
            records.append(dict(batch=batch,queries=queries,relative_error=error,median_seconds=med,samples=times,
                                cold_speedup=med[0]/med[1],warm_speedup=med[2]/med[3]))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(r['relative_error']<=1e-10 for r in records),
                pred_b=all(r['cold_speedup']>=1.05 for r in records if r['queries']==32),
                records=records,prices=prices,arm_order=['cached_cold','compiled_cold','cached_warm','compiled_warm'],
                scope='FP64 CPU/two threads; one saved native16-token prefix, batch8 repeats it. Cold includes preparation, warm explicitly excludes it. Full originalMLP/contextgenerators remain common; no wholemodel/GPU speed or staticparameter saving claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in records],indent=2));signal.alarm(0)


if __name__=='__main__':main()
