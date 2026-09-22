"""Completed shared producer fit: opened-panel per-feature diagnostics."""
import hashlib,json
from pathlib import Path
import torch
from sparse_quartic_bank import features
from score_residual_fresh_panel import summarize
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 run=json.loads((P/'SHARED_MIXED_FEATURES_NATIVE_V1.json').read_text());assert run['predictions']['pred_a_integrity']
 x=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();y=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/19054614563.464127
 draws=torch.randint(32,(2000,32),generator=torch.Generator().manual_seed(220922));rows=[]
 for r in run['rows']:
  file=P/f"SHARED_MIXED_FEATURES_SEED{r['seed']}_V1.pt";assert hashlib.sha256(file.read_bytes()).hexdigest()==r['sha256'];p=torch.load(file,weights_only=True)
  pred=features(x,*[f.double() for f in p['factors']],p['pairs'])@(p['coefficients'].double()/19054614563.464127).T
  s=summarize(pred,y,draws);assert abs(s['pooled_relative_error']-r['final']['text_error'])<1e-5;s['seed']=r['seed'];rows.append(s)
 out=dict(rows=rows,scope='Opened2048state/32prefixpanel, per-output diagnosis of completed shared producers. Not included in frozen five-candidate freshpanel preregistration. Bootstrap conditional on openedpanel, not freshvalidation.')
 (P/'LEARNED_SHARED_RESIDUALS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 for r in rows:print(r['seed'],[(f['feature'],round(f['relative_error']*100,2)) for f in r['features']],r['worst10_error_share'])
if __name__=='__main__':main()
