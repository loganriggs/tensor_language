"""Actual-weight synthetic control, frozen before execution.

A: every generated post-MLP10 branch relative error <=1e-10.
B: mixed response relative error <=1e-8 for nonzero amplitude pairs.
C: either zero edit yields mixed maxabs <=1e-10.
Direct reference recomputes MLP9, attention10 and MLP10 independently.
No native-text behavioral, suffix or speed claim.
"""
import json
import time
from pathlib import Path
import torch
from head17_source_interface_v1 import CHECKPOINT
from composed_mlp10_context_v1 import prepare
import composed_joint_response_v1 as joint
import raw_attention_response_v1 as attn

P = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(61373)
    start = time.perf_counter()
    sd = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    program = torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt', weights_only=True)
    z9 = torch.randn(1, 17, 1152, dtype=torch.float64)
    raw10 = torch.randn_like(z9)
    first = torch.randn_like(z9)
    def mlp_weights(index):
        prefix = f'transformer.h.{index}.mlp.'
        return [sd[prefix+k+'.weight'].double() for k in ['Left','Right','Down']]
    l9,r9,d9 = mlp_weights(9)
    l10,r10,d10 = mlp_weights(10)
    bias10 = sd['transformer.h.10.mlp.Down_bias'].double()
    def h9(z):
        return z+((z@l9.T)*(z@r9.T))@d9.T/(z.square().mean(-1,keepdim=True)+attn.EPS)
    h0 = h9(z9)
    baseline_mlp9 = h0-z9
    matrices = [sd['transformer.h.10.attn.'+k+'.weight'].double()
                for k in ['c_q','c_k','c_q2','c_k2','c_v']]
    output = sd['transformer.h.10.attn.c_proj.weight'].double()
    mixture = float(sd['transformer.h.10.attn.lamb'])
    scale = float(sd['transformer.h.10.lambdas'][0])
    def attend(raw):
        ports,rho = attn.prepare(raw,matrices)
        return attn.execute(ports,rho,first,mixture,output)
    z10 = raw10+attend(raw10)
    context = prepare(z9,baseline_mlp9,raw10,z10,first,program,scale,matrices,mixture,output)
    writer = context['response']['direction']
    def direct(amplitude):
        raw = raw10+scale*(h9(z9-amplitude*writer)-h0)
        z = raw+attend(raw)
        return joint.block_output(z,l10,r10,d10,bias10)
    a0 = torch.randn(1,17,1,dtype=z9.dtype)*.3
    b0 = torch.randn_like(a0)*.3
    rows=[]
    for alpha,beta in [(1,1),(-1,1),(.5,1.5),(0,1),(1,0)]:
        a,b = alpha*a0,beta*b0
        generated = joint.evaluate(a,b,context,l10,r10,d10,bias10)
        reference = {key:direct(value) for key,value in
                     [('native',torch.zeros_like(a)),('child',a),('remainder',b),('parent',a+b)]}
        mixed = reference['parent']-reference['child']-reference['remainder']+reference['native']
        branch_error = max(float((generated[k]-v).norm()/v.norm()) for k,v in reference.items())
        mixed_error = float((generated['mixed']-mixed).norm()/mixed.norm().clamp_min(1e-30)) if alpha*beta else None
        zero_maxabs = float(generated['mixed'].abs().max()) if not alpha*beta else None
        rows.append(dict(alpha=alpha,beta=beta,branch_error=branch_error,mixed_error=mixed_error,zero_maxabs=zero_maxabs))
    result=dict(pred_a=all(r['branch_error']<=1e-10 for r in rows),
                pred_b=all(r['mixed_error']<=1e-8 for r in rows if r['mixed_error'] is not None),
                pred_c=all(r['zero_maxabs']<=1e-10 for r in rows if r['zero_maxabs'] is not None),
                cells=rows,seconds=time.perf_counter()-start,
                scope='Actual weights, one synthetic 17-position context, five signed/amplitude pairs. Pristine context and dense weights supplied; generated changes include parent inherited response. No text or suffix validation.')
    (P/'COMPOSED_JOINT_RESPONSE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert result['pred_a'] and result['pred_b'] and result['pred_c']


if __name__=='__main__':
    main()
