"""Small arithmetic-graph refit: keep 32 products, learn only their weights.
Alternating least squares optimizes each source's 16 weights conditionally.
Primary ridge .01; 0 and 1 controls. Three starts, 100 full sweeps.
The eight evaluation documents are opened by the preceding functional study.
"""
import json,sys,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from source_interface import source_read
torch.set_num_threads(2);torch.set_grad_enabled(False)
data=torch.load(P/'FUNCTIONAL_QUARTIC_MODE3_INPUTS_V1.pt',weights_only=True)
c=data['common'];white=data['whitened_delta'];affine=data['affine']
train=24*64;scale=c['scale'];truth=c['true_phi'];start=time.perf_counter()
sign=[c['sources'][k]['sign'] for k in ['a','b']]
features=[]
for k in ['a','b']:
 U=c['sources'][k]['initial']
 features.append((white@U).square()-U.square().sum(0))
ga0=(c['t']-.5*affine[:,0])/scale-c['alpha']
gb0=affine[:,1]/scale-c['beta']
GA=-.5*features[0]/scale[:,None];GB=features[1]/scale[:,None]
variance=truth[:train].var(unbiased=False);normalizer=(train*variance).sqrt()
z=torch.load(P/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True)['z'].flatten(0,1).double()
def predict(a,b):return (ga0+GA@a)*(gb0+GB@b)
def objective(a,b,penalty):
 return ((predict(a,b)[:train]-truth[:train]).square().mean()/variance
   +penalty*((a-sign[0]).square().sum()+(b-sign[1]).square().sum())/32)
def update(fixed,which,penalty):
 if which=='a':mult=gb0+GB@fixed;design=mult[:,None]*GA;target=truth-mult*ga0;prior=sign[0]
 else:mult=ga0+GA@fixed;design=mult[:,None]*GB;target=truth-mult*gb0;prior=sign[1]
 X=design[:train]/normalizer;y=target[:train]/normalizer
 if penalty:
  ridge=(penalty/32)**.5;X=torch.cat([X,ridge*torch.eye(16,dtype=X.dtype)]);y=torch.cat([y,ridge*prior])
 return torch.linalg.lstsq(X,y,driver='gelsd').solution
def export(a,b):
 p={k:c[k].clone() for k in ['h_reader','residual_writer','alpha','beta']}
 for k,w in [('a',a),('b',b)]:
  source=c['sources'][k];U=source['initial'];proj=c['source_inverse_root']@U;pm=c['mu']@proj
  p[k+'_reader']=proj;p[k+'_eigenvalues']=w;p[k+'_linear']=source['linear']-2*proj@(w*pm)
  p[k+'_bias']=source['constant']-c['mu']@source['linear']+(w*(pm.square()-U.square().sum(0))).sum()
 return p
def score(a,b):
 pred=predict(a,b);out={}
 for name,sl in [('train',slice(0,train)),('evaluation',slice(train,None))]:
  true=truth[sl];out[name+'_variation_error']=float((pred[sl]-true).norm()/(true-true.mean()).norm())
 return out
records=[];programs={'initial':export(*sign)};baseline=score(*sign)
for penalty in [0.,.01,1.]:
 for seed in range(3):
  gen=torch.Generator().manual_seed(530+seed)
  a,b=[s.clone() if seed==0 else s+.1*torch.randn(s.shape,dtype=s.dtype,generator=gen) for s in sign]
  history=[float(objective(a,b,penalty))];best=(history[0],a.clone(),b.clone(),0)
  for step in range(1,101):
   a=update(b,'a',penalty);b=update(a,'b',penalty);value=float(objective(a,b,penalty))
   assert value<=history[-1]+1e-10
   history.append(value)
   if value<best[0]:best=(value,a.clone(),b.clone(),step)
  value,a,b,step=best;label=f'ridge{penalty}_seed{seed}';program=export(a,b)
  qa=source_read(z,program,'a');qb=source_read(z,program,'b')
  replay=((c['t']-.5*qa)/scale-c['alpha'])*(qb/scale-c['beta'])
  error=float((replay-predict(a,b)).norm()/predict(a,b).norm());assert error<1e-8
  records.append(dict(label=label,ridge=penalty,seed=seed,training_objective=value,best_step=step,**score(a,b),export_replay=error,max_weight_change=float(torch.cat([a-sign[0],b-sign[1]]).abs().max()),objective_history=history))
  programs[label]=program
selected={str(p):min([r for r in records if r['ridge']==p],key=lambda r:r['training_objective'])['label'] for p in [0.,.01,1.]}
primary=next(r for r in records if r['label']==selected['0.01'])
out=dict(predictions=dict(pred_a_replay=max(r['export_replay'] for r in records)<1e-8,pred_b_primary=primary['evaluation_variation_error']<=min(.15,.8*baseline['evaluation_variation_error'])),initial=baseline,selected_by_training_objective=selected,records=records,trainable_coefficients=32,source_products=32,seconds=time.perf_counter()-start,scope='24/8 existing documents; opened evaluation subset. Frozen source directions, exact centered affine branches; coefficient refit only. No fresh behavior or native intervention claim.')
(P/'FIXED_QUARTIC_PRODUCT_FIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
torch.save(programs,P/'FIXED_QUARTIC_PRODUCT_PROGRAMS_V1.pt')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
for row in records:
 if row['label'] in selected.values():print({k:v for k,v in row.items() if k!='objective_history'})
