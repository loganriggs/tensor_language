from pathlib import Path
import json,torch
from prepared_bilinear_response_tensor import compile,evaluate
from normalized_bilinear_secant import mlp_secant

torch.manual_seed(2601);torch.set_num_threads(2);b,d,m,r=3,11,19,4;dt=torch.float64;eps=torch.finfo(torch.float32).eps
L,R=torch.randn(m,d,dtype=dt),torch.randn(m,d,dtype=dt);D=torch.randn(d,m,dtype=dt);h=torch.randn(b,d,dtype=dt);W=torch.randn(b,d,r,dtype=dt);z=torch.randn(b,r,dtype=dt)*.2;delta=torch.einsum('bdr,br->bd',W,z);program=compile(L,R,D,h,W,eps)
expected=delta+mlp_secant(h,h+delta,delta,L,R,D,eps);error=float((evaluate(program,z)-expected).abs().max());zero=float(evaluate(program,z*0).abs().max())
# Normalization-aware contraction cancels a nonzero raw polynomial exactly.
v=torch.randn(d,dtype=dt);identity=torch.eye(d,dtype=dt);same=v[:,None].expand(d,d)
cancel=compile(identity,identity,same,h,W,0.);cancellation=float(cancel['coefficients'].abs().max());assert error<1e-10 and zero==0 and cancellation<1e-10
out=dict(secant_reference_error=error,zero_response=zero,normalized_constant_cancellation=cancellation,prices=dict(native_r=27,canonical_values_per_context=1152*405+1152*27+406,implicit_values_per_context=2*4608*27+2*4608+1152*27+406+1152,implicit_shared_down=1152*4608),scope='Exact context-prepared tensor and planted normalization cancellation; no native or sparse approximation claim')
p=Path(__file__).with_name('PREPARED_BILINEAR_RESPONSE_CPU_V1.json');assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(out)
