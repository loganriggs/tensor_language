"""Post-result native-reference, sign and prior-program comparison."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
r=json.loads((P/'COMPOSED_JOINT_NATIVE_FULL_V1_RESULT.json').read_text())
previous=json.loads((P/'COMPOSED_BACKGROUND_SHARED_V2_RESULT.json').read_text())
a=torch.load(P/'MLP9_TO_MLP10_RESIDUAL_FOLD_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['measures'].double()
native=torch.tensor([c['native_scores'] for c in r['cells']],dtype=torch.float64)
generated=torch.tensor([c['generated_scores'] for c in r['cells']],dtype=torch.float64)
old_corners=a[:,[0,1,3,2]]
reference_error=float((native-old_corners).abs().max())
assert reference_error==0, 'Cross-receipt comparison requires identical native corners'
effect=lambda x:x[:,3]-x[:,1]-x[:,2]+x[:,0]
truth=effect(native);prediction=effect(generated)
old_parent=a[:,5]+torch.tensor([c['background_drift'] for c in previous['cells']],dtype=torch.float64)+torch.tensor([c['transported_effects'] for c in previous['cells']],dtype=torch.float64)
old_prediction=old_parent-a[:,1]-a[:,3]+a[:,0]
groups=[]
for lo,hi,name in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'FineWeb'+str(k)) for k in range(4)]:
    n=truth[lo:hi];g=prediction[lo:hi];old=old_prediction[lo:hi]
    groups.append(dict(name=name,old_relative_errors=((old-n).norm(dim=0)/n.norm(dim=0)).tolist(),
                       new_relative_errors=((g-n).norm(dim=0)/n.norm(dim=0)).tolist(),
                       sign_reversals=(g*n<0).sum(dim=0).tolist(),
                       material_sign_reversals=((g*n<0)&(n.abs()>=1e-5)).sum(dim=0).tolist()))
flips=[dict(row=i,endpoint=j,native=float(truth[i,j]),generated=float(prediction[i,j]))
       for i,j in (prediction*truth<0).nonzero().tolist()]
result=dict(native_corners_maxabs_error=reference_error,groups=groups,all_sign_reversals=flips,
            maximum_state_error=max(max(c['state_errors']) for c in r['cells']),
            maximum_branch_output_error=max(c['absolute_output_error'] for c in r['cells']),
            scope='Post-result aligned historical-panel comparison. Same native corners exactly. Old model predicts parent using generated additive background plus local cross; new predicts all changed branches including inherited mixed response. Extra branch/dense suffix cost remains.')
(P/'COMPOSED_JOINT_NATIVE_FULL_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
