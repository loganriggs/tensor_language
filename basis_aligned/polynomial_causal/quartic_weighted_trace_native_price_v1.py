"""Two CPU native operator actions; validation and timing, no factor fitting."""
from pathlib import Path
import json,time,math,hashlib
import torch
from quartic_weighted_trace_v1 import producer_core,native_action
from composed_quartic_contraction_v1 import contract
P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.perf_counter()
assert json.loads((P/'QUARTIC_WEIGHTED_TRACE_V1_CONTROL.json').read_text())['pred_a']
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
tp=P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_PROGRAMS.pt';assert digest(tp)==json.loads((P/'QUARTIC_REPEATED_INPUT_NATIVE_V1_RESULT.json').read_text())['artifact_sha256']
target=torch.load(tp,weights_only=True)['programs'][0];writer=target['output_writers']
binding=json.loads((P/'COUPLED_QUARTIC_LBFGS_V1_BINDING.json').read_text())['files'];ck=next(k for k in binding if k.endswith('/pytorch_model.bin'));assert digest(ck)==binding[ck]
state=torch.load(ck,weights_only=True,mmap=True)
u=state['lm_head.weight'].double();mean=u.mean(0);reader=u.T@(u@writer[:,0])-len(u)*mean*torch.dot(mean,writer[:,0]);del u
l,r,down,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double() for layer in (16,17) for name in ('Left','Right','Down')]
coefficient=reader@d1;scale=float(state['transformer.h.17.lambdas'][0]);h=producer_core(down,l1,r1,coefficient,scale)
times=[];tic=time.perf_counter();identity=native_action(l,r,h,torch.eye(1152)/math.sqrt(1152));times.append(time.perf_counter()-tic)
reference=target['native_target_traces'][0]/math.sqrt(1152);identity_error=float((identity-reference).norm()/reference.norm())
print(json.dumps(dict(identity_error=identity_error,seconds=times[-1])),flush=True)
torch.manual_seed(91891);basis=torch.linalg.qr(torch.randn(1152,3),mode='reduced')[0];a,b,z=basis.T
q=(torch.outer(a,a)-torch.outer(b,b))/math.sqrt(2)
tic=time.perf_counter();answer=native_action(l,r,h,q);times.append(time.perf_counter()-tic)
slots=torch.stack([torch.stack([z,z,a,a]),torch.stack([z,z,b,b])])
values=contract(slots,l,r,down,l1,r1,coefficient[None],scale)[:,0]
expected=float((values[0]-values[1])/math.sqrt(2));actual=float(z@answer@z)
scalar_error=abs(actual-expected)/max(abs(actual),abs(expected),1e-30)
result=dict(pred_a=max(identity_error,scalar_error)<=1e-8,identity_error=identity_error,scalar_contraction_error=scalar_error,operator_seconds=times,wall_seconds=time.perf_counter()-start,core_shape=list(h.shape),core_bytes=h.numel()*h.element_size(),scope='CPU FP64 exact first-output composed quartic operator, two actions only; not eigenvalue convergence, native GPU timing or circuit evidence.')
out=P/'QUARTIC_WEIGHTED_TRACE_NATIVE_V1_PRICE.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']
