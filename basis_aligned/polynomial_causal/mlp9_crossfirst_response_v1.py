"""Compile the existing exact fixed-writer response formula at MLP9."""
from pathlib import Path
import json,time,torch
from directional_mlp_bridge_v1 import execute
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);torch.manual_seed(241346);tic=time.perf_counter();b=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in b if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);w=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)['writers'][1].clone().double();L=sd['transformer.h.9.mlp.Left.weight'].double();R=sd['transformer.h.9.mlp.Right.weight'].double();D=sd['transformer.h.9.mlp.Down.weight'].double();J=D@((R@w)[:,None]*L+(L@w)[:,None]*R);program=dict(direction=w,mixed_map=J);z=torch.randn(32,1152,dtype=torch.float64);a=torch.linspace(-1,1,32,dtype=torch.float64)[:,None];bamp=.37*torch.sin(torch.arange(32,dtype=torch.float64))[:,None];eps=torch.finfo(torch.float32).eps
 def mlp(z):
  x=z/(z.square().mean(-1,keepdim=True)+eps).sqrt();return ((x@L.T)*(x@R.T))@D.T
 base=mlp(z);actual=z-a*w+mlp(z-a*w)-z-base;pred=execute(z,base,a,program);mix=execute(z,base,a+bamp,program)-execute(z,base,a,program)-execute(z,base,bamp,program);directmix=(z-(a+bamp)*w+mlp(z-(a+bamp)*w))-(z-a*w+mlp(z-a*w))-(z-bamp*w+mlp(z-bamp*w))+z+base;rel=lambda x,y:float((x-y).norm()/y.norm().clamp_min(1e-30));response=rel(pred,actual);mixed=rel(mix,directmix);rho=z.square().mean(-1,keepdim=True)+eps
 # Writer term cancels from the mixed response; projection checks the derived three-vector span.
 spans=torch.stack([base,z@J.T,(J@w).expand_as(z)],-1)
 def coeff(t):
  rr=(z-t*w).square().mean(-1,keepdim=True)+eps;return torch.cat([rho/rr-1,-t/rr,t.square()/(2*rr)],-1)
 weights=coeff(a+bamp)-coeff(a)-coeff(bamp);spanerr=rel((spans*weights[:,None,:]).sum(-1),mix)
 result=dict(response_error=response,mixed_response_error=mixed,three_vector_mixed_span_error=spanerr,program_scalars=w.numel()+J.numel(),seconds=time.perf_counter()-tic,scope='Reused exact RMS-aware directional response atMLP9 onactualweights andrandomFP64states; signedamplitudes. BackgroundbiasfreeMLP output andstate remainexternal. Three-vector span isper-context algebra, not three identifiedsemanticcircuits ornativeeffectprediction.')
 assert max(response,mixed,spanerr)<1e-10;torch.save(program,P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt');(P/'MLP9_CROSSFIRST_RESPONSE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
