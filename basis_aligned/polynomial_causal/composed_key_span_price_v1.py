"""Fair primitive serialization and prepared-runtime comparison."""
import json,time,statistics
from pathlib import Path
from datetime import datetime,timezone
import torch
import torch.nn.functional as F
import shared_key_reads_v1 as old
import composed_key_span_v1 as new
from contracted_qk_response_v1 import features
P=Path(__file__).resolve().parent


def primitive(p):
    return {k:v.clone() for k,v in p.items() if k in ['readers','left','right','direction','gain','key_coordinates']}


def prepare(p):
    p=dict(p);r=p['readers'];l=p['left'];d=p['direction']
    p.update(read_left=r@l,read_direction=r@d,gram=l.T@l,left_direction=l.T@d,direction_norm2=d@d)
    k=torch.cat([r[128:256],r[384:512]])
    adapters=(k@k.T)@p['key_coordinates'] if 'key_coordinates' in p else k@r[512:576].T
    p['inside_adapters']=adapters.reshape(2,128,64)
    return p


@torch.no_grad()
def main():
    out=P/'COMPOSED_KEY_SPAN_PRICE_V1_RESULT.json';assert not out.exists();torch.set_num_threads(2)
    checkpoint='/home/loganriggs/.local/share/bilin18/hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
    sd=torch.load(checkpoint,weights_only=True,mmap=True)
    low=torch.load(P/'DIRECTIONAL_ROUTING_PREDICTOR_V1_TOP64.pt',weights_only=True)
    native=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
    lam=sd['transformer.h.9.lambdas'].double();bias=sd['transformer.h.8.mlp.Down_bias'].double()
    programs=[old.compile_program(native,low,lam[0]),new.compile_program(native,low,lam[0])]
    payloads=[];prepared=[];files=[];loadtimes=[]
    for name,p in zip(['OLD','NEW'],programs):
        file=P/('COMPOSED_KEY_SPAN_CANONICAL_'+name+'_V1_PROGRAM.pt');assert not file.exists()
        torch.save(primitive(p),file);files.append(file)
        start=time.perf_counter();loaded=torch.load(file,weights_only=True);prepared.append(prepare(loaded));loadtimes.append(time.perf_counter()-start)
        payloads.append(sum(v.numel()*v.element_size() for v in loaded.values()))
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    row=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'][0]
    n=len(row['ids']);ids=torch.tensor([row['ids']]);z=cache['z8'][0,0,:n][None].double();h=cache['raw9'][0,0,:n][None].double();a=cache['amplitude8'][0,0,:n][None]
    x0=F.rms_norm(F.embedding(ids,sd['transformer.wte.weight']),(1152,)).double()
    u=(h-lam[1]*x0)/lam[0]-z-bias;checks=[];timings=[]
    rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
    for batch in [1,8]:
        args=[x.repeat((batch,)+(1,)*(x.ndim-1)) for x in [z,h,u,a]]
        def run(p,module):
            rd,rho=features(*args,p);return module.scalar(rd,rho,p)
        references=[run(p,m) for p,m in zip(programs,[old,new])]
        values=[run(p,m) for p,m in zip(prepared,[old,new])]
        checks.extend([rel(v,r) for v,r in zip(values,references)]+[rel(values[1],values[0])])
        samples=[[],[]]
        for trial in range(7):
            for j in ([0,1] if trial%2==0 else [1,0]):
                start=time.perf_counter()
                for _ in range(20):run(prepared[j],[old,new][j])
                samples[j].append((time.perf_counter()-start)/20)
        med=list(map(statistics.median,samples))
        timings.append(dict(batch=batch,length=n,samples=samples,medians=med,speedup=med[0]/med[1]))
    sizes=[f.stat().st_size for f in files]
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=max(checks)<=1e-10,
                pred_b=1-sizes[1]/sizes[0]>=.07,pred_c=all(t['speedup']>=1.03 for t in timings),
                replay_errors=checks,canonical_file_bytes=sizes,canonical_payload_bytes=payloads,
                prepared_payload_bytes=[sum(v.numel()*v.element_size() for v in p.values()) for p in prepared],
                load_and_prepare_seconds=loadtimes,timings=timings,
                scope='Both programs omit and regenerate deterministic caches. CPU first cached prefix repeated atbatch8; not eight independent contexts. External context generation/native suffix excluded; rank64 approximation unchanged.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
