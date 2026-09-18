"""Physically trim the mediator factors and verify exact interface reduction."""
from pathlib import Path
import hashlib,json,os,time,torch
from mlp8_value_mediator_v1 import execute
P=Path(__file__).resolve().parent;STEM='CITY_MLP8_VALUE_EXTRACTED_V1'
@torch.no_grad()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();out=P/(STEM+'_CPU_RESULT.json');assert not out.exists()
 old=torch.load(P/'CITY_FINEWEB_MLP8_VALUE_FOLD_V1_PROGRAM.pt',weights_only=True)
 program={k:old[k].clone() for k in ['left','right','folded_down','lambda9','mixture']}
 fresh=torch.load(P/'CITY_VALUE_MEDIATION_FRESH_V1_ARTIFACT.pt',weights_only=True)['fixtures'];fixtures=[];errors=[];outside=[]
 for f in fresh:
  inputs={k:f['inputs'][k] for k in ['z','delta']};inputs['edited_rho9']=(f['inputs']['mixed9_edited'].double().square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
  got=execute(program,**inputs);expected=f['candidate_value_delta'];errors.append(float((got-expected).norm()/expected.norm()))
  mask=inputs['delta'].abs().sum(-1)>0;outside.append(float(got[~mask].abs().max()));fixtures.append({'inputs':inputs,'expected':expected,'delta':got})
 rejected=[];base=fixtures[0]['inputs']
 for change in [{'edited_rho9':torch.zeros_like(base['edited_rho9'])},{'edited_rho9':base['edited_rho9'][...,0]},{'z':base['z'].expand(2,-1,-1),'delta':base['delta'].expand(2,-1,-1),'edited_rho9':base['edited_rho9'].expand(2,-1,-1)}]:
  try:execute(program,**{**base,**change});rejected.append(False)
  except ValueError:rejected.append(True)
 count=sum(v.numel() for v in program.values());owned=all(v.untyped_storage().nbytes()==v.numel()*v.element_size() for v in program.values())
 r={'pred_a':max(errors)<=1e-10 and max(outside)==0 and all(rejected) and all(bool(torch.isfinite(f['delta']).all()) for f in fixtures),'pred_b':count==11206658 and owned,'fixtures':40,'max_relative_error':max(errors),'max_outside':max(outside),'invalid_inputs_rejected':all(rejected),'static_float_scalars':count,'weight_bytes':4*count,'native_input_scalars_T32':36896,'intervention_scalars_T32':36864,'previous_native_input_scalars_T32':110592,'owned_storage':owned,'full_model_forwards':0,'seconds':time.perf_counter()-start,'scope':'Opened exact interface reduction. Native edited RMS remains supplied. No new fresh/independent composition claim.','source_shas':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_MLP8_VALUE_EXTRACTED_V1_PREREGISTRATION.md','mlp8_value_mediator_v1.py','build_city_mlp8_value_extracted_v1.py']}}
 torch.save(program,P/(STEM+'_PROGRAM.pt'));torch.save({'fixtures':fixtures},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
