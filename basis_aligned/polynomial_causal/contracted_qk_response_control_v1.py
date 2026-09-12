import json,torch,time
from pathlib import Path
from contracted_qk_response_v1 import compile_program,features,scalar,EPS
from compiled_directional_response_v1 import predict
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];low=torch.load(P/'DIRECTIONAL_ROUTING_PREDICTOR_V1_TOP64.pt',weights_only=True);native=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True);bindings=json.loads((P/'DIRECTIONAL_INTERACTION_LOGIT_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in bindings if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True);lam=sd['transformer.h.9.lambdas'].double();bias=sd['transformer.h.8.mlp.Down_bias'].double();p=compile_program(native,low,lam[0]);errs=[];base=[];norms=[]
 import importlib.util,torch.nn.functional as F
 spec=importlib.util.spec_from_file_location('contract_control_runtime',P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 for i,row in enumerate(rows):
  n=len(row['ids']);ids=torch.tensor([row['ids']]);z=cache['z8'][0,i,:n][None].double();h=cache['raw9'][0,i,:n][None].double();a=cache['amplitude8'][0,i,:n][None];x0=F.rms_norm(F.embedding(ids,sd['transformer.wte.weight']),(1152,)).double();u=(h-lam[1]*x0)/lam[0]-z-bias
  reads,rho2=features(z,h,u,a,p);out=scalar(reads,rho2);full=predict(z,h,u,a,low['direction'],lam[0],left=low['left'],right=low['right']);reference=runtime.scalar(F.rms_norm(full.float(),(1152,)),ids,native,1);errs.append(rel(out,reference));norms.append(rel(rho2,full.square().mean(-1,keepdim=True)+EPS));base.append(rel(scalar(h@p['readers'].T,h.square().mean(-1,keepdim=True)+EPS),cache['scalar'][0,i,:n][None]))
 result={'baseline_max_relative_error':max(base),'contracted_max_relative_error':max(errs),'gram_norm_max_relative_error':max(norms),'pred_a':max(base)<=1e-5,'pred_b':max(errs)<=1e-5,'pred_c':max(norms)<=1e-10,'seconds':time.perf_counter()-tic,'scope':'CPU control on prior cache, FP64 contraction versus frozen FP32 normalized-state scalar. 769 explicit query/key/inside/value readings; exact Gram norm. No new native behavior or autonomous prefix claim.'};(P/'CONTRACTED_QK_RESPONSE_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(p,P/'CONTRACTED_QK_RESPONSE_V1_PROGRAM.pt');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
