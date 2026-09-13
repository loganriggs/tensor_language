"""Compare raw-projection attention response against actual FP32 attention10."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
import torch.nn.functional as F

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parents[1]))
from jacclust.tt_model import CausalBilinearSelfAttention
from head17_source_interface_v1 import CHECKPOINT
from raw_attention_response_v1 import prepare,changed,execute


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.manual_seed(349)
    sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')
    module=CausalBilinearSelfAttention(SimpleNamespace(n_head=9,n_embd=1152,
                                      squared_attn=True,bilinear_attn=True)).eval()
    prefix='transformer.h.10.attn.'
    module.load_state_dict({k[len(prefix):]:v for k,v in sd.items() if k.startswith(prefix)})
    matrices=[getattr(module,k).weight.double() for k in ('c_q','c_k','c_q2','c_k2','c_v')]
    raw=torch.randn(2,17,1152);first=torch.randn_like(raw)
    base,rho=prepare(raw,matrices)
    output=module.c_proj.weight.double();mix=float(module.lamb)
    baseline=execute(base,rho,first.double(),mix,output)
    native0=module(F.rms_norm(raw,(1152,)),first)[0].double()
    rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30))
    direction=torch.randn_like(raw);cells=[]
    for strength in (0.,.001,.03,.3,-.3):
        delta=strength*direction
        ports,newrho=changed(raw,delta,base,matrices)
        predicted=execute(ports,newrho,first.double(),mix,output)
        native=module(F.rms_norm(raw+delta,(1152,)),first)[0].double()
        directports,directrho=prepare(raw.double()+delta.double(),matrices)
        direct=execute(directports,directrho,first.double(),mix,output)
        cells.append(dict(strength=strength,projection_identity_error=rel(predicted,direct),
                          native_output_error=rel(predicted,native),
                          native_change_error=rel(predicted-baseline,native-native0),
                          maxabs_change_error=float(((predicted-baseline)-(native-native0)).abs().max())))
    result=dict(baseline_error=rel(baseline,native0),cells=cells,
                scope='Actual attention10 weights, FP32 native module versus FP64 projection algebra, '
                'two synthetic 17-token contexts. Both QK factors, all heads/positions, causal mask, '
                'rounded rotary and unchanged shared first values included. No native corpus or '
                'MLP9-generator accuracy claim; no fitting or state compression.')
    assert result['baseline_error']<1e-5
    assert all(c['projection_identity_error']<1e-12 and c['native_output_error']<1e-5 for c in cells)
    assert all(c['native_change_error']<.01 for c in cells)
    (P/'RAW_ATTENTION_RESPONSE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
