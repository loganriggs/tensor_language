"""Exact per-output-mode Gaussian gain versus reused text gain; no refitting."""
import json
from pathlib import Path
import torch
from frozen_program_evaluation import quartic
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(3);torch.set_grad_enabled(False)
 U=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True)['output_directions'].double();base={k:v.double() for k,v in torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True)['programs'][4].items()};panels=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];rows=[]
 for family in ['QUADRATIC','MIXED']:
  source=torch.load(P/f'{family}_SKIP_PROGRAM_V1.pt',weights_only=True);K=source['cross_covariance'].double();G=source['primitive_covariance'].double()
  for rank,s in source['programs'].items():
   s={k:v.double() for k,v in s.items()};W=s['skip_writer']@s['skip_reader'] if 'skip_reader' in s else s['skip_writer'];WG=U.T@W;KG=U.T@K
   gain=2*(WG*KG).sum(1)-((WG@G)*WG).sum(1);row=dict(family=family,rank=rank,exact_gaussian_mode_mse_reduction=gain.tolist(),text=[])
   for panel in panels:
    x=panel['rows'].double();y=panel['targets'].double();old=(quartic(base,x)-y)@U;new=(quartic(s,x)-y)@U;oldmse=old.square().mean(0);newmse=new.square().mean(0);row['text'].append(dict(context=panel['context'],mode_mse_reduction=(oldmse-newmse).tolist(),mode_mse_reduction_fraction=(1-newmse/oldmse).tolist()))
   rows.append(row)
 result=dict(records=rows,scope='Exact fixed-basis Gaussian per-mode gain versus reused text residual gain. No model fitting/selection; distinguishes aggregate objective from distribution mismatch. These polynomial metrics are not native intervention metrics.')
 (P/'SKIP_MODE_OBJECTIVE_AUDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
 for r in rows:print(r['family'],r['rank'],'Gaussian',r['exact_gaussian_mode_mse_reduction'],'text fractions',[p['mode_mse_reduction_fraction'] for p in r['text']])
if __name__=='__main__':main()
