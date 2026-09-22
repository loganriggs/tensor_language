"""Frozen fresh-panel diagnostics; all bootstrap units are whole documents."""
import hashlib,json,math,time
from pathlib import Path
import torch
from sparse_quartic_bank import features
from paired_root_compiler import cast
from audit_root_feature_conditions import root_features
P=Path(__file__).resolve().parent;SCALE=19054614563.464127

def summarize(pred,y,draws):
 r=pred-y;e=r.square();energy=y.square();n=len(y);docs=draws.shape[1];length=n//docs
 assert n%docs==0
 de=e.reshape(docs,length,16).sum(1);dy=energy.reshape(docs,length,16).sum(1)
 be=de[draws].sum(1);by=dy[draws].sum(1)
 pooled=(e.sum()/energy.sum()).sqrt();bs=(be.sum(1)/by.sum(1)).sqrt();bf=(be/by).sqrt();q=lambda a:torch.quantile(a,torch.tensor([.025,.975],dtype=a.dtype),dim=0).tolist()
 rowe=e.sum(1);rowy=energy.sum(1);top=rowe.topk(math.ceil(.1*n)).indices;mean=r.mean(0);cov=(r-mean).T@(r-mean)/n;evals=torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)
 return dict(pooled_relative_error=float(pooled),pooled_document_bootstrap_95=q(bs),features=[dict(feature=g,relative_error=float((e[:,g].sum()/energy[:,g].sum()).sqrt()),document_bootstrap_95=q(bf[:,g]),residual_energy_share=float(e[:,g].sum()/e.sum()),target_energy_share=float(energy[:,g].sum()/energy.sum())) for g in range(16)],worst10_error_share=float(rowe[top].sum()/rowe.sum()),target_energy_in_worst10=float(rowy[top].sum()/rowy.sum()),common_rms_error_quantiles={str(v):float(torch.quantile(rowe.sqrt()/(rowy.mean().sqrt()),v)) for v in [.5,.9,.95,.99,1.]},row_relative_error_quantiles={str(v):float(torch.quantile((rowe/rowy.clamp_min(1e-30)).sqrt(),v)) for v in [.5,.9,.95,.99,1.]},prefix_errors=((de.sum(1)/dy.sum(1)).sqrt()).tolist(),bias_energy_share=float(n*mean.square().sum()/e.sum()),centered_residual_pc_shares=(evals/evals.sum()).tolist())

def main(dry=False):
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 if dry:
  g=torch.Generator().manual_seed(22);y=torch.randn(256*64,16,generator=g,dtype=torch.float64);draws=torch.randint(256,(2000,256),generator=g);s=summarize(1.1*y,y,draws)
  assert abs(s['pooled_relative_error']-.1)<1e-12
  assert max(abs(v-.1) for v in s['pooled_document_bootstrap_95'])<1e-12
  assert all(abs(f['relative_error']-.1)<1e-12 for f in s['features'])
  print('actual-shape bootstrap and proportional-residual control PASS');return
 start=time.monotonic();a=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);meta=json.loads((P/'RESIDUAL_FRESH_ROWS_V1.json').read_text());assert a['token_content_sha256']==meta['token_content_sha256'];x=a['rows'].double();y=a['target'].double()/SCALE;assert x.shape==(16384,1152) and y.shape==(16384,16)
 draws=torch.randint(256,(2000,256),generator=torch.Generator().manual_seed(220922));rows=[]
 for file,sha in a['candidate_sha256'].items():
  assert hashlib.sha256((P/file).read_bytes()).hexdigest()==sha
  prog=torch.load(P/file,weights_only=True);t=time.monotonic();preds=[]
  for xx in x.split(2048):
   if file.startswith('EXPANDED'):pred=root_features(cast(prog,torch.float64),xx)/SCALE
   elif file.startswith('MIXED_CP'):
    z=torch.ones(len(xx),512,dtype=torch.float64)
    for f in prog['factors']:z*=xx@f.double().T
    pred=z@(prog['coefficients'].double()/SCALE).T
   else:pred=features(xx,*[f.double() for f in prog['factors']],prog['pairs'])@(prog['coefficients'].double()/SCALE).T
   preds.append(pred)
  pred=torch.cat(preds);elapsed=time.monotonic()-t;s=summarize(pred,y,draws);s.update(artifact=file,prediction_seconds=elapsed)
  if file.startswith('MIXED_CP'):
   seed=prog['seed'];old={1001:.06322183414224321,1002:.06587528291254952}[seed]
   s['predictions']=dict(aggregate_transfer=s['pooled_relative_error']<=1.25*old and all(f['relative_error']<=.1 for f in s['features'][:3]),feature_weakness_replication=sum(f['relative_error']>.3 for f in s['features'][4:])>=8,concentration_replication=s['worst10_error_share']>=.3)
  rows.append(s);print(file,s['pooled_relative_error'],s.get('predictions'),flush=True)
 result=dict(rows=rows,seconds=time.monotonic()-start,documents=256,states=16384,outputs=16,bootstrap_draws=2000,bootstrap_seed=220922,scope='Frozen candidates on new document panel. Whole-document percentile bootstrap conditional on this sample; not full-model, OOD or selective-intervention proof. No fitting.')
 (P/'RESIDUAL_FRESH_SCORES_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 import sys
 main('--dry-run' in sys.argv)
