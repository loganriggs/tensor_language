"""Build native factors and test explicit edited-normalizer generation."""
from pathlib import Path
import hashlib,json,os,time,torch
import torch.nn.functional as F
from mlp8_value_norm_closed_v1 import prepare,execute
from regional_endpoint_batching_v1 import group_rows
P=Path(__file__).resolve().parent;STEM='CITY_MLP8_NORM_CLOSED_V1'
@torch.no_grad()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();out=P/(STEM+'_CPU_RESULT.json');assert not out.exists()
 binding=json.loads((P/'PHI4_PROVENANCE_V1_BINDING.json').read_text())['files'];checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'));sd=torch.load(checkpoint,weights_only=True,mmap=True)
 rows=json.loads((P/'CITY_VALUE_MEDIATION_FRESH_V1_ROWS.json').read_text())['rows'];groups,_=group_rows(rows);tokens=torch.tensor(sorted({t for r in groups for t in r['ids']}))
 program={'left':sd['transformer.h.8.mlp.Left.weight'].clone(),'right':sd['transformer.h.8.mlp.Right.weight'].clone(),'down':sd['transformer.h.8.mlp.Down.weight'].clone(),'bias':sd['transformer.h.8.mlp.Down_bias'].clone(),'value_reader':sd['transformer.h.9.attn.c_v.weight'][1024:1152].clone(),'lambdas9':sd['transformer.h.9.lambdas'].clone(),'mixture':sd['transformer.h.9.attn.lamb'].clone(),'token_ids':tokens,'initial_table':F.rms_norm(sd['transformer.wte.weight'][tokens].float(),(1152,)).clone()}
 runtime=prepare(program);old=torch.load(P/'CITY_VALUE_MEDIATION_FRESH_V1_ARTIFACT.pt',weights_only=True)['fixtures'];fixtures=[];norm_errors=[];value_errors=[];outside=[]
 for f,row in zip(old,groups,strict=True):
  inputs={k:f['inputs'][k] for k in ['z','delta']};inputs['token_ids']=torch.tensor([row['ids']]);value,rho=execute(runtime,**inputs,return_rho=True)
  native_rho=(f['inputs']['mixed9_edited'].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
  norm_errors.append(float((rho-native_rho).norm()/native_rho.norm()));value_errors.append(float((value-f['native_value_delta']).norm()/f['native_value_delta'].norm()))
  mask=inputs['delta'].abs().sum(-1)>0;outside.append(float(value[~mask].abs().max()));fixtures.append({'inputs':inputs,'delta':value,'rho':rho,'native_value_delta':f['native_value_delta']})
 rejected=[];f=fixtures[0]['inputs'];bad=f['token_ids'].clone();bad[0,0]=-1
 for inputs in [{**f,'token_ids':bad},{k:v.expand(2,*v.shape[1:]) for k,v in f.items()}]:
  try:execute(runtime,**inputs);rejected.append(False)
  except ValueError:rejected.append(True)
 floats=sum(v.numel() for v in program.values() if v.is_floating_point());owned=all(v.untyped_storage().nbytes()==v.numel()*v.element_size() for v in program.values())
 r={'pred_a':len(fixtures)==40 and max(norm_errors)<=1e-5,'pred_b':max(value_errors)<=1e-4,'pred_c':max(outside)==0 and all(rejected) and owned and all(bool(torch.isfinite(f['delta']).all()) for f in fixtures),'max_rho_relative_error':max(norm_errors),'max_value_relative_error':max(value_errors),'max_outside':max(outside),'unsupported_tokens_and_batch_rejected':all(rejected),'tokens':len(tokens),'stored_float_scalars':floats,'stored_float_bytes':4*floats,'stored_integer_bytes':8*len(tokens),'derived_folded_down_scalars':runtime['folded_down'].numel(),'derived_folded_down_bytes':runtime['folded_down'].numel()*8,'runtime_float_bytes':sum(v.numel()*v.element_size() for v in runtime.values() if v.is_floating_point()),'native_input_scalars_T32':36864,'intervention_scalars_T32':36864,'native_scalar_norm_inputs':0,'checkpoint_sha256':binding[checkpoint],'full_model_forwards':0,'seconds':time.perf_counter()-start,'scope':'Opened native factor closure of editedRMS9; nativez8 and upstreamdelta remain external; larger storedweights, no independentcomposition claim.','source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_MLP8_NORM_CLOSED_V1_PREREGISTRATION.md','mlp8_value_norm_closed_v1.py','build_city_mlp8_norm_closed_v1.py']}}
 torch.save(program,P/(STEM+'_PROGRAM.pt'));torch.save({'fixtures':fixtures},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
