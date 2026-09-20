"""Keep the frozen feature dictionary; change only diagnostic-selected scalar1."""
import json
from pathlib import Path
import torch
from extract_scalar_modes import evaluate,build
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 original=torch.load(P/'EXTRACTED_SCALAR_INTERVENTIONS_V1.pt',weights_only=True)
 fit=torch.load(P/'PRIMITIVE_DECODER_PROGRAMS_V1.pt',weights_only=True)['programs']['fixed_0.0001']
 program={k:v.clone() for k,v in original['program'].items()}
 for k in ['A','B','root_left','root_right']:assert torch.equal(program[k],fit[k])
 for k in ['quartic_readout','quadratic_readout','constant']:program[k][1]=fit[k][1]
 for k in ['quartic_readout','quadratic_readout','constant']:assert torch.equal(program[k][[0,2,3]],original['program'][k][[0,2,3]])
 checks=[]
 for panel in torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels']:
  x=panel['rows'].double();ref=evaluate({k:v.double() for k,v in original['program'].items()},x);new=evaluate({k:v.double() for k,v in program.items()},x)
  delta=float((new[:,[0,2,3]]-ref[:,[0,2,3]]).abs().max());assert delta==0
  checks.append(dict(context=panel['context'],unchanged_scalar_max_abs=delta,changed_mode1_relative_norm=float((new[:,1]-ref[:,1]).norm()/ref[:,1].norm())))
 dag,out=build({k:v.double() for k,v in program.items()});cost=dag.cost(out);assert cost['products']==10 and cost['stored_coefficients']==13916
 artifact={**original,'program':program,'scope':'Diagnostic-selected refit of scalar1 only. Input features, other scalar functions, and residual writers are unchanged. Native joint behavior and fresh transfer untested.'}
 # Original source hash described the old extraction; retain it explicitly as lineage only.
 if 'source_program_sha256' in artifact:artifact['original_source_program_sha256']=artifact.pop('source_program_sha256')
 artifact['refit_source']='PRIMITIVE_DECODER_PROGRAMS_V1.pt:fixed_0.0001'
 torch.save(artifact,P/'SELECTIVE_SCALAR_READOUT_V1.pt')
 result=dict(checks=checks,cost=cost,changed_scalar=1,selection='After reused diagnostic native outcomes, not a preregistered candidate or fresh confirmation.',scope=artifact['scope'])
 (P/'SELECTIVE_SCALAR_READOUT_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
