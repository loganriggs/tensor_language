"""Exact fixed-direction bilinear-MLP response, with explicit native background input."""
import json,time,torch
from pathlib import Path
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(9231753);tic=time.perf_counter()
 out=P/'SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_RESULT.json';assert not out.exists()
 binding=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files']
 state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 L,R,D=[state['transformer.h.8.mlp.'+k+'.weight'].double() for k in ('Left','Right','Down')]
 d=torch.load(P/'SCALAR_PRODUCER_NATIVE_LIFT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['writers'][0].double();ld=L@d;rd=R@d
 J=(D*rd[None])@L+(D*ld[None])@R
 h=D@(ld*rd);square_error=float((J@d/2-h).norm()/h.norm())
 z=torch.randn(32,1152,dtype=torch.float64);a=torch.linspace(-.1,.1,32,dtype=torch.float64)[:,None]*z.norm(dim=-1,keepdim=True)/d.norm();eps=torch.finfo(torch.float32).eps
 def evaluate(dtype):
  zz=z.to(dtype);aa=a.to(dtype);dd=d.to(dtype);jj=J.to(dtype);ll,rr,down=[v.to(dtype) for v in (L,R,D)]
  rho=zz.square().mean(-1,keepdim=True)+eps;zm=zz-aa*dd;rhom=zm.square().mean(-1,keepdim=True)+eps
  u=((zz@ll.T)*(zz@rr.T))@down.T/rho
  prediction=-aa*dd+(rho/rhom-1)*u-aa/rhom*((zz-aa*dd/2)@jj.T)
  actual=-aa*dd+((zm@ll.T)*(zm@rr.T))@down.T/rhom-u
  return float((prediction-actual).norm()/actual.norm())
 error64=evaluate(torch.float64);error32=evaluate(torch.float32)
 program=dict(direction=d,mixed_map=J)
 torch.save(program,P/'SCALAR_PRODUCER_DIRECTIONAL_MLP_V1_PROGRAM.pt')
 result={'pred_a':square_error<=1e-10,'pred_b':error64<=1e-10,'pred_c':error32<=1e-4,'quadratic_from_mixed_identity_error':square_error,'fp64_response_error':error64,'fp32_response_error':error32,'program_scalars':sum(t.numel() for t in program.values()),'program_tensor_bytes':sum(t.numel()*t.element_size() for t in program.values()),'original_mlp_scalars':15926400,'seconds':time.perf_counter()-tic,'scope':'Exact conditional response for one fixed native writer direction. Inputs z, native biasfree MLP output u and producer amplitude a remain required; full native background/prefix is not eliminated. Algebra probes, no end-task adoption, no rank fitting.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
