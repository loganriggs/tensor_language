from pathlib import Path
import torch,json,itertools
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');torch.set_num_threads(2);torch.set_grad_enabled(False);rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)['y'].double();S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'];result=json.loads((p/'MIDPOINT_STABILITY_V1.json').read_text());records=[]
def basis(ids):
 y=rows[ids].flatten(0,1)@S;y-=y.mean(0);_,W=torch.linalg.eigh(y.T@y/len(y));return W.flip(1)
for r in result['partitions']:
 a=basis(r['first_documents']);b=basis(r['second_documents']);C=a.T@b;record=dict(partition=r['partition'],subspace_min_cosines={str(k):float(torch.linalg.svdvals(C[:k,:k]).min()) for k in [3,4,5,8]},first4_capture_by_second8=(C[:4,:8].square().sum(1)).sqrt().tolist(),second4_capture_by_first8=(C[:8,:4].square().sum(0)).sqrt().tolist());records.append(record)
out=p/'MIDPOINT_STABILITY_DIAGNOSIS_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,scope='Post-hoc neighboring-mode diagnosis using originalcalibration cache. Higher-rank subspace agreement is not stable individual identification; no changed pass criteria.'),indent=2)+'\n');print(json.dumps(records,indent=2))
