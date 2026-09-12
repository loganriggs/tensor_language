import json,torch,time,importlib.util
import torch.nn.functional as F
from pathlib import Path
from response_ports_v1 import compile_weights,project_ports,execute
from response_without_mlp_port_v1 import predict
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();p=torch.load(P/'CONTRACTED_QK_RESPONSE_V1_PROGRAM.pt',weights_only=True);w=compile_weights(p);native=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True);cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];spec=importlib.util.spec_from_file_location('port_control_runtime',P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime);rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));errors=[];norms=[]
 for i,row in enumerate(rows):
  n=len(row['ids']);z=cache['z8'][0,i,:n][None].double();h=cache['raw9'][0,i,:n][None].double();amp=cache['amplitude8'][0,i,:n][None];donor=cache['amplitude8'][0,row['donor_id'],:n][None];ports=project_ports(z,h,p);assert sum(t.numel() for t in ports.values())==901*n
  for a in [amp,-amp,amp-donor]:
   out,rho2=execute(ports,a,w);state=predict(z,h,a,p['direction'],p['gain'],p['left'],p['right']);ref=runtime.scalar(F.rms_norm(state.float(),(1152,)),torch.tensor([row['ids']]),native,1);errors.append(rel(out,ref));norms.append(rel(rho2,state.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps))
 result={'pred_a':max(errors)<=1e-5,'pred_b':max(norms)<=1e-10,'max_scalar_relative_error':max(errors),'max_squared_norm_relative_error':max(norms),'ports_per_token':901,'amplitudes_per_token':1,'executor_weight_scalars':sum(x.numel() for x in w.values()),'executor_tensor_bytes':sum(x.numel()*x.element_size() for x in w.values()),'projection_matrix_scalars':sum(p[k].numel() for k in ['readers','left','right','direction']),'seconds':time.perf_counter()-tic,'scope':'216 cached numerical cases, removal/addition/donor amplitudes. Closed declared-port API, not autonomous prefix generation. Projection matrices and generating native h/z still cost resources; runtime speed unmeasured.'};torch.save(w,P/'RESPONSE_PORTS_V1_WEIGHTS.pt');(P/'RESPONSE_PORTS_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
