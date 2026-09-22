"""All four terminal balanced-fit arms: opened-panel values and finite changes."""
import hashlib,json,time
import torch
from audit_conditional_residual_accounting import P,SCALE
from sparse_quartic_bank import features

def scores(pred,y,rec,don):
 er=pred-y;ref=y[don]-y[rec];de=pred[don]-pred[rec]-ref
 fv=(er.square().sum(0)/y.square().sum(0)).sqrt();fr=(de.square().sum(0)/ref.square().sum(0)).sqrt()
 return dict(value_error=float(er.norm()/y.norm()),feature_value_errors=fv.tolist(),small_value_rms=float(fv[4:].square().mean().sqrt()),response_error=float(de.norm()/ref.norm()),feature_response_errors=fr.tolist(),small_response_rms=float(fr[4:].square().mean().sqrt()))

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 run=json.loads((P/'BALANCED_SHARED_FEATURES_NATIVE_V1.json').read_text());assert run['predictions']['pred_a_integrity']
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();y=data['target'].double()/SCALE
 pairfile=P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json';matched=json.loads(pairfile.read_text());rec,don=torch.tensor(matched['pairs']).T
 oldx=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();oldy=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/SCALE;rows=[]
 for r in run['rows']:
  path=P/f"BALANCED_SHARED_FEATURES_{r['arm'].upper()}_SEED{r['seed']}_V1.pt";h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==r['sha256'];p=torch.load(path,weights_only=True);U,V=[f.double() for f in p['factors']];C=p['coefficients'].double()/SCALE
  def evaluate(xx):return features(xx,U,V,p['pairs'])@C.T
  oldpred=evaluate(oldx);old_error=float((oldpred-oldy).norm()/oldy.norm());assert abs(old_error-r['final']['text_error'])<1e-5
  pred=torch.cat([evaluate(xx) for xx in x.split(2048)]);assert torch.isfinite(pred).all()
  row=dict(seed=r['seed'],arm=r['arm'],sha256=h,old_panel_replay_error=abs(old_error-r['final']['text_error']),**scores(pred,y,rec,don));rows.append(row);print(row['seed'],row['arm'],row['value_error'],row['small_value_rms'],row['response_error'],row['small_response_rms'],flush=True)
 contrasts=[]
 for seed in [1101,1102]:
  u=next(r for r in rows if r['seed']==seed and r['arm']=='uniform');b=next(r for r in rows if r['seed']==seed and r['arm']=='balanced')
  contrasts.append(dict(seed=seed,**{k+'_ratio':b[k]/u[k] for k in ['value_error','small_value_rms','response_error','small_response_rms']}))
 result=dict(rows=rows,contrasts=contrasts,registered_predictions=run['predictions'],seconds=time.monotonic()-start,n_pairs=len(rec),pair_receipt_sha256=hashlib.sha256(pairfile.read_bytes()).hexdigest(),scope='Posthoc opened256document panel/all4frozenarms. Matchedtokenposition pairs fixed beforeterminalresults. No fitting or candidate selection. Registered bars evaluated separately onoriginal2048states; these extra diagnostics are not freshconfirmation or newpromotioncriteria.')
 (P/'BALANCED_SHARED_FOLLOWUP_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
