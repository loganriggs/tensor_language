"""Independent CPU-only replay of the exported sparse conditional response DAG."""
import json
from pathlib import Path
import torch
from conditional_attention_mlp_runtime import execute
from projected_bilinear_response import readout_prepared

ROOT=Path(__file__).resolve().parents[2]
artifact=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_conditional_dag_v670.pt'
package=torch.load(artifact,map_location='cpu',weights_only=True)
torch.set_num_threads(2)
errors=[];control_errors=[];rows=0
for case in package['cases']:
    logits,z=execute(package['runtime'],case['attention'],case['initial'],case['contexts'],case['final_context'],case['positions'])
    controls=readout_prepared(case['control_fixed'],z,case['control_context'])
    errors.append(float((logits-case['reference_logits']).abs().max()))
    control_errors.append(float((controls-case['reference_controls']).abs().max()))
    rows+=len(logits)
assert max(errors+control_errors)<=1e-8
assert not torch.cuda.is_initialized()
def values(v):
    if isinstance(v,torch.Tensor):return v.numel()
    if isinstance(v,dict):return sum(values(x) for x in v.values())
    if isinstance(v,list):return sum(values(x) for x in v)
    return 0
result=dict(rows=rows,max_logit_error=max(errors),max_control_logit_error=max(control_errors),
    cuda_initialized=torch.cuda.is_initialized(),artifact_bytes=artifact.stat().st_size,
    runtime_values=values(package['runtime']),prepared_case_values=values(package['cases']),
    scope='Prepared-context CPU extraction only; native context and initial-response generators remain external')
(ROOT/'basis_aligned/polynomial_causal/CONDITIONAL_DAG_CPU_REPLAY_2026-09-20.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
