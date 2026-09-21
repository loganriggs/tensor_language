"""Opened-state component and sensitivity checks for frozen weight-fitted programs."""
import json,time
from pathlib import Path
import torch
from sparse_quartic_bank import features
from paired_root_compiler import cast
from audit_root_feature_conditions import root_features
from audit_root_matched_reader import stats
P=Path(__file__).resolve().parent


def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double()
 labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=labels['target'].double();weight=labels['weight'].double()
 matched=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];pairs=torch.tensor(matched['pairs_flat']);rec,don=pairs.T
 reference=torch.tensor([r['reference'] for r in matched['pair_rows']],dtype=torch.float64);cached=target[don,1]-target[rec,1];replay=float((cached-reference).norm()/reference.norm());assert replay<1e-5,replay
 programs=[('empirical384',P/'EXPANDED_ROOT_EMPIRICAL_V1.pt')]
 for seed in [1001,1002]:
  programs += [(f'cp_coefficient_{seed}',P/f'QUARTIC_CP512_SEED{seed}_V2.pt'),(f'cp_gaussian_I_{seed}',P/f'GAUSSIAN_CP512_SEED{seed}_V1.pt'),(f'cp_shifted_{seed}',P/f'GAUSSIAN_CP_DATA_COVARIANCE_SHIFTED_SEED{seed}_V1.pt')]
 for seed in [1101,1102]:programs.append((f'sparse_bank_{seed}',P/f'SPARSE_QUARTIC_BANK_SEED{seed}_V1.pt'))
 rows=[]
 for name,path in programs:
  p=cast(torch.load(path,weights_only=True),torch.float64)
  if name=='empirical384':pred=root_features(p,x)
  elif name.startswith('sparse_bank'):pred=features(x,*p['factors'],p['pairs'])@p['coefficients'].T
  else:
   phi=torch.ones(len(x),512,dtype=torch.float64)
   for f in p['factors']:phi=phi*(x@f.T)
   pred=phi@p['coefficients'].T
  weighted=((weight*(pred-target).square()).sum(0)/(weight*target.square()).sum(0)).sqrt()
  row=dict(program=name,aggregate_value_error=float((pred-target).norm()/target.norm()),root1_sensitivity_error=float(weighted[1]),mean_root_sensitivity_error=float(weighted.mean()),all_root_sensitivity_errors=weighted.tolist(),same_token_root1_response=stats(pred[don,1]-pred[rec,1],reference))
  rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='all_root_sensitivity_errors'}),flush=True)
 result=dict(rows=rows,target_pair_replay=replay,seconds=time.monotonic()-start,scope='Frozen programs on opened2048states. Root1 fixed prior newline-associated writer; same-token30directed/20unordered pairs. Sensitivity metric is local final-logit geometry, not actual finite removal or untouched OOD.')
 (P/'GAUSSIAN_CP_COMPONENT_TRANSFER_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
