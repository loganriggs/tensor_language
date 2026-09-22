"""Evaluate all native local-residual arms on the larger opened panel."""
import argparse,hashlib,json,time,math
import torch
from audit_conditional_residual_accounting import P,SCALE,load
from audit_balanced_shared_followup import scores

def correction(p,x):
 nout,per=p['coefficients'].shape
 assert p['output_start']==4 and per==p['per_output'] and nout==12
 phi=torch.stack([x@a.T for a in p['factors']]).prod(0).reshape(len(x),nout,per)
 return (phi*p['coefficients']).sum(2)/SCALE

def controls():
 torch.manual_seed(39001);x=torch.randn(37,11,dtype=torch.float64)
 p=dict(factors=[torch.randn(96,11,dtype=torch.float64) for _ in range(4)],coefficients=torch.randn(12,8,dtype=torch.float64),output_start=4,per_output=8)
 dense=torch.zeros(12,96,dtype=torch.float64)
 for i in range(12):dense[i,i*8:(i+1)*8]=p['coefficients'][i]
 expected=torch.stack([x@a.T for a in p['factors']]).prod(0)@dense.T/SCALE
 actual=correction(p,x);replay=float((actual-expected).norm()/expected.norm());assert replay<1e-12
 parent=torch.randn(37,16,dtype=torch.float64);candidate=parent.clone();candidate[:,4:]+=actual
 assert torch.equal(parent[:,:4],candidate[:,:4])
 return dict(packed_dense_replay=replay,protected_outputs_equal=True)

def distribution(pred,y):
 """Residual concentration within the supplied output subspace."""
 assert pred.shape==y.shape and torch.isfinite(pred).all() and torch.isfinite(y).all()
 residual=pred-y; e=residual.square().sum(1); target=y.square().sum(1)
 tiny=torch.finfo(y.dtype).tiny
 quant=lambda z:{str(q):float(torch.quantile(z,q)) for q in [.5,.9,.95,.99,1.]}
 mean=residual.mean(0);centered=residual-mean
 eig=torch.linalg.eigvalsh(centered.T@centered).flip(0).clamp_min(0)
 return dict(row_relative_error=quant((e/target.clamp_min(tiny)).sqrt()),
  row_error_over_common_target_rms=quant((e/target.mean().clamp_min(tiny)).sqrt()),
  error_concentration={str(q):float(e.topk(max(1,math.ceil(len(e)*q))).values.sum()/e.sum().clamp_min(tiny)) for q in [.01,.05,.1]},
  target_energy_on_worst_error_rows={str(q):float(target[e.topk(max(1,math.ceil(len(e)*q))).indices].sum()/target.sum().clamp_min(tiny)) for q in [.01,.05,.1]},
  bias_energy_share=float(len(y)*mean.square().sum()/e.sum().clamp_min(tiny)),
  centered_residual_pc_energy_shares=(eig/eig.sum().clamp_min(tiny)).tolist())


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--lbfgs',action='store_true');args=parser.parse_args()
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();check=controls()
 prefix='NATIVE_LOCAL_LBFGS_FOLLOWUP' if args.lbfgs else 'LOCAL_QUARTIC_FOLLOWUP'
 optimizers=['lbfgs'] if args.lbfgs else ['adam','muon']
 if args.check:
  (P/(prefix+'_PREFLIGHT_V1.json')).write_text(json.dumps(check,indent=2)+'\n');print(check);return
 receipt=P/('NATIVE_LOCAL_LBFGS_V1.json' if args.lbfgs else 'LOCAL_QUARTIC_RESIDUAL_NATIVE_V1.json')
 if not receipt.exists():raise SystemExit('Pending native fit receipt; do not restart or modify queued learner.')
 run=json.loads(receipt.read_text());assert len(run['rows'])==2*len(optimizers)
 assert {(r['optimizer'],r['seed']) for r in run['rows']}=={(o,s) for o in optimizers for s in [25001,25002]}
 parent,ph=load('MIXED_CP_FEATURES_SEED1001_V1.pt')
 def pv(x):return torch.stack([x@a.T for a in parent['factors']]).prod(0)@parent['coefficients'].T/SCALE
 oldx=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();oldy=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/SCALE
 oldpair=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat']).T
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();y=data['target'].double()/SCALE
 pair=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T
 oldparent=pv(oldx);parentpred=torch.cat([pv(xx) for xx in x.split(1024)]);baseline=scores(parentpred,y,*pair);rows=[]
 for row in run['rows']:
  name=('NATIVE_LOCAL_' if args.lbfgs else 'LOCAL_QUARTIC_RESIDUAL_')+f"{row['optimizer'].upper()}_SEED{row['seed']}_V1.pt";p,sha=load(name)
  assert sha==row['sha256'] and p['parent_sha256']==ph
  oldpred=oldparent.clone();oldpred[:,4:]+=correction(p,oldx);oldscore=scores(oldpred,oldy,*oldpair)
  replay=max(abs(oldscore[k]-row['metrics'][k]) for k in ['small_value_rms','small_response_rms']);assert replay<1e-4
  pred=parentpred.clone();pred[:,4:]+=torch.cat([correction(p,xx) for xx in x.split(1024)]);assert torch.equal(pred[:,:4],parentpred[:,:4]);s=scores(pred,y,*pair)
  doc=[]
  for pp,yy in zip(pred.split(64),y.split(64)):
   e=((pp[:,4:]-yy[:,4:]).square().sum(0)/yy[:,4:].square().sum(0)).sqrt();doc.append(float(e.square().mean().sqrt()))
  rec=dict(optimizer=row['optimizer'],seed=row['seed'],sha256=sha,old_export_score_replay=replay,protected_outputs_equal=True,value_ratio=s['small_value_rms']/baseline['small_value_rms'],response_ratio=s['small_response_rms']/baseline['small_response_rms'],document_small_value_rms=doc,residual_distribution=distribution(pred,y),small_output_residual_distribution=distribution(pred[:,4:],y[:,4:]),**s)
  rows.append(rec);print(row['optimizer'],row['seed'],rec['value_ratio'],rec['response_ratio'],flush=True)
 result=dict(rows=rows,baseline=baseline,baseline_distribution=distribution(parentpred,y),baseline_small_distribution=distribution(parentpred[:,4:],y[:,4:]),controls=check,original_registered_predictions=run['predictions'],descriptive_transfer=any(all(r['value_ratio']<=.85 and r['response_ratio']<=.85 for r in rows if r['optimizer']==o) for o in optimizers),seconds=time.monotonic()-start,scope='All registered arms in this optimizer receipt on already opened 16384-state/2494-pair panel. No selection, refitting, new holdout or native causal adoption. Absolute per-output errors retained; learned additions target only outputs4–15.')
 (P/(prefix+'_V1.json')).write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
