"""Four-context native-module even/odd recomposition, fixed before execution.

A: FP64 decomposition sum vs native FP32 <=1e-5, all four contexts.
B: sum vs direct FP64 algebra <=1e-10. C: omitted-odd tripwire >1e-8.
Conditional attention-module execution, no GPU or whole-model forwards.
"""
from datetime import datetime,timezone
from types import SimpleNamespace
import json,sys
import torch
import torch.nn.functional as F
from check_even_key_value_bank_v1 import P,CHECKPOINT
from even_key_value_bank_v1 import execute
sys.path.insert(0,str(P.parents[1]))
from jacclust.tt_model import CausalBilinearSelfAttention,Rotary,apply_rotary_emb


@torch.no_grad()
def main():
    out=P/'EVEN_KEY_NATIVE_RECOMPOSITION_V1_RESULT.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(2)
    p=torch.load(P/'EVEN_KEY_VALUE_BANK_V1_PROGRAM.pt',weights_only=True)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True)
    config=SimpleNamespace(n_head=9,n_embd=1152,squared_attn=True,bilinear_attn=True)
    native=CausalBilinearSelfAttention(config).eval()
    prefix='transformer.h.9.attn.'
    native.load_state_dict({k[len(prefix):]:v for k,v in sd.items() if k.startswith(prefix)})
    # A single native output head, all other output columns explicitly zero.
    native.c_proj.weight[:,:8*128]=0
    cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
    rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
    records=[]
    for i in [0,18,36,54]:
        row=rows[i];n=len(row['ids']);ids=torch.tensor([row['ids']])
        current=F.rms_norm(cache['raw9'][0,i,:n][None].float(),(1152,))
        initial=F.rms_norm(F.embedding(ids,sd['transformer.wte.weight']).float(),(1152,))
        first=F.linear(initial,sd['transformer.h.0.attn.c_v.weight'].float())
        reference,_=native(current,first)
        x=current.double();x0=initial.double();rotary=Rotary(128)
        factors=[];reflected=[]
        # Independent dense original basis projector (not key-coordinate reuse).
        original=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
        basis=original['key_basis'][1].double()
        for qn,kn in [('q1','k1'),('q2','k2')]:
            q=F.linear(x,p[qn].double());k=F.linear(x,p[kn].double())
            q=q/(q.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
            den=(k.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
            inside=(x@basis)@(p[kn].double()@basis).T
            co,si=rotary(q[:,:,None,:])
            qr=apply_rotary_emb(q[:,:,None,:],co,si)[:,:,0]
            kr=apply_rotary_emb((k/den)[:,:,None,:],co,si)[:,:,0]
            rr=apply_rotary_emb(((k-2*inside)/den)[:,:,None,:],co,si)[:,:,0]
            factors.append(qr@kr.transpose(-1,-2)/128)
            reflected.append(qr@rr.transpose(-1,-2)/128)
        full=factors[0]*factors[1];ref=reflected[0]*reflected[1]
        mask=torch.ones(n,n,dtype=torch.bool).tril()
        full=full.masked_fill(~mask,0);odd=((full-ref)/2).masked_fill(~mask,0)
        lam=p['mixture'].double()
        values=(1-lam)*(x@p['current_value'].double().T)+lam*(x0@p['first_value'].double().T)
        odd_write=(odd@values)@p['output'].double().T
        even,_=execute(x,x0,p)
        direct=(full@values)@p['output'].double().T
        rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
        records.append(dict(row=i,native_error=rel(even+odd_write,reference.double()),
                            algebra_error=rel(even+odd_write,direct),
                            omitted_odd_error=rel(even,direct),
                            even_norm=float(even.norm()),odd_norm=float(odd_write.norm()),
                            full_norm=float(direct.norm())))
    result=dict(utc=datetime.now(timezone.utc).isoformat(),pred_a=all(r['native_error']<=1e-5 for r in records),
                pred_b=all(r['algebra_error']<=1e-10 for r in records),
                pred_c=all(r['omitted_odd_error']>1e-8 for r in records),records=records,
                scope='Native attention class FP32 with other eight output heads zeroed; both values, mixture and full head9.8 writer retained. Dense-basis odd component plus compiled coordinate even component. Four old contexts, conditional module interface, no new text/OOD/selective suffix evidence.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
