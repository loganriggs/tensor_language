"""Complete port equation versus fixed existing response; all signed strengths."""
import json,time,signal
from pathlib import Path
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
from complete_response_ports_v1 import compile_weights,project_ports,execute
from contracted_qk_response_v1 import features
from composed_key_span_v1 import scalar
P=Path(__file__).resolve().parent


@torch.no_grad()
def main():
    out=P/'COMPLETE_RESPONSE_PORTS_V1_RESULT.json';assert not out.exists();signal.alarm(120)
    torch.set_num_threads(2);start=time.perf_counter()
    p=torch.load(P/'COMPOSED_KEY_SPAN_V1_PROGRAM.pt',weights_only=True);w=compile_weights(p)
    checkpoint='/home/loganriggs/.local/share/bilin18/hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
    sd=torch.load(checkpoint,weights_only=True,mmap=True);lam=sd['transformer.h.9.lambdas'].double();bias=sd['transformer.h.8.mlp.Down_bias'].double()
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];records=[];portcounts=[]
    def error(a,b):
        norm=float(b.norm());delta=float((a-b).norm())
        return delta/norm if norm else delta
    for i,row in enumerate(rows):
        n=len(row['ids']);ids=torch.tensor([row['ids']]);z=cache['z8'][0,i,:n][None].double();h=cache['raw9'][0,i,:n][None].double();amp=cache['amplitude8'][0,i,:n][None]
        x0=F.rms_norm(F.embedding(ids,sd['transformer.wte.weight']),(1152,)).double();u=(h-lam[1]*x0)/lam[0]-z-bias
        ports=project_ports(z,h,u,p);portcounts.append(sum(t.numel() for t in ports.values())//n)
        if i==0:
            broken=dict(ports);del broken['cu']
            try:execute(broken,amp,w)
            except KeyError:pass
            else:raise AssertionError('missing compensation port accepted')
        rd0,rho0=features(z,h,u,amp*0,p);ref0=scalar(rd0,rho0,p);pred0,_=execute(ports,amp*0,w)
        for strength in [-2,-1,0,.5,1,2]:
            a=amp*strength;pred,rho=execute(ports,a,w);rd,reference_rho=features(z,h,u,a,p);ref=scalar(rd,reference_rho,p)
            own_ref=ref-ref0;own_pred=pred-pred0
            records.append(dict(row=i,strength=strength,scalar_error=error(pred,ref),norm_error=error(rho,reference_rho),
                                own_change_error=error(own_pred,own_ref),own_reference_norm=float(own_ref.norm()),
                                zero_own_pass=float((own_pred-own_ref).abs().max())<=1e-12 if not bool(own_ref.norm()) else True))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(max(r['scalar_error'],r['norm_error'],r['own_change_error'])<=1e-10 and r['zero_own_pass'] for r in records),
                pred_b=max(portcounts)<=.4*3456,ports_per_token=sorted(set(portcounts)),records=records,
                executor_weight_scalars=sum(v.numel() for v in w.values()),projection_weight_scalars=sum(p[k].numel() for k in ['readers','left','right','direction']),
                seconds=time.perf_counter()-start,scope='Complete compensation preserved, 432 cached signed-amplitude cases. Conditional ports, not independent native generators; rank64 approximation unchanged. Zero-change comparisons use absolute1e-12 control.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}));signal.alarm(0)


if __name__=='__main__':main()
