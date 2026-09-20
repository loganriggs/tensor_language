"""Registered exact noncentral Gaussian bank metric correction, fixed widths."""
import itertools,json,math,time
from pathlib import Path
import torch
from audit_bank_function_metric import embedding
from gaussian_quartic_mean import quadratic_moments
from quartic_bank_refactor import evaluate,build
P=Path(__file__).resolve().parent

def features(a,b):return .5*(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:])

def objective(T,m,a,b,penalty=.001):
 Q=features(a,b);E=embedding(Q,m);scale=E.square().sum(1).clamp_min(1e-24).pow(.25);u,v=a/scale[:,None],b/scale[:,None];E=embedding(features(u,v),m);G=E@E.T;X=T@E.T;c=torch.linalg.solve(G+penalty*torch.eye(len(a),dtype=a.dtype),X.T).T;loss=((c@G)*c).sum()-2*(c*X).sum()+penalty*c.square().sum();return loss,(u,v,c)

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.perf_counter();core=torch.load(P/'QUARTIC_BANK_CORE_V1.pt',weights_only=True);prior=torch.load(P/'QUARTIC_BANK_REFACTOR_V1.pt',weights_only=True);source=torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True);s={k:v.double() for k,v in source['programs'][8].items()};panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];mu=panels[0]['mean'].double();M=panels[0]['covariance'].double();m=core['input_mapback']@mu;T=embedding(core['weighted_core'],m);norm=T.square().sum();records=[];best={};allfits={}
 for width,optimizer,seed in itertools.product([8,12],['adam','muon'],[0,1]):
  old=prior['allfits'][(width,optimizer,.03,seed)];a=torch.nn.Parameter(old['a'].clone());b=torch.nn.Parameter(old['b'].clone());initial=float((old['c']@embedding(features(a,b),m)-T).norm().detach()/norm.sqrt());opt=torch.optim.Adam([a,b],lr=.005) if optimizer=='adam' else torch.optim.Muon([a,b],lr=.005,weight_decay=0.,adjust_lr_fn='match_rms_adamw');bestloss=float('inf')
  for step in range(501):
   opt.zero_grad();loss,program=objective(T,m,a,b);value=float(loss.detach());assert math.isfinite(value)
   if value<bestloss:bestloss=value;saved=tuple(t.detach().clone() for t in program);selected=step
   if step==500:break
   (loss/norm).backward();opt.step()
   for g in opt.param_groups:g['lr']=.005*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/500)))
  u,v,c=saved;pred=c@embedding(features(u,v),m);err=float((pred-T).norm()/norm.sqrt());cancel=float(c.square().sum()/pred.square().sum());row=dict(width=width,optimizer=optimizer,seed=seed,initial_function_error=initial,function_error=err,penalized_objective=bestloss,selected_step=selected,function_component_energy_ratio=cancel);records.append(row);allfits[(width,optimizer,seed)]=dict(a=u,b=v,c=c)
  if width not in best or bestloss<best[width][0]['penalized_objective']:best[width]=(row,u,v,c)
  print(row,flush=True)
 targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];fresh=torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'];exports={};winners=[];i,j=torch.triu_indices(4,4)
 for width,(row,a,b,c) in best.items():
  A=a@core['input_mapback'];B=b@core['input_mapback'];C=torch.linalg.solve(core['sqrt_root_sensitivity'],c);mean,G=quadratic_moments(A,B,mu,M);mean=C@mean;G=C@G@C.T;constant=source['mean_teacher_gaussian'].double()-s['W']@s['Z']@(mean[i]*mean[j]+G[i,j]);program=dict(A=A,B=B,bank_writer=C,W=s['W'],Z=s['Z'],constant=constant);archive={k:v.float() for k,v in program.items()};loaded={k:v.double() for k,v in archive.items()};dag,out=build(loaded);x=panels[0]['rows'][:16].double();replay=float((dag.evaluate(out,x)-evaluate(loaded,x)).norm()/evaluate(loaded,x).norm());cost=dag.cost(out);assert cost['stored_coefficients']==sum(v.numel() for v in archive.values()) and cost['products']==width+10 and replay<1e-10
  diagnostics=[float((evaluate(loaded,p['rows'].double())-y).norm()/y.norm()) for p,y in zip(panels,targets)];fresherrors=[float((evaluate(loaded,p['rows'].double())-p['targets'].double()).norm()/p['targets'].double().norm()) for p in fresh];winners.append(dict(row,diagnostic_errors=diagnostics,fresh_errors=fresherrors,cost=cost,graph_replay=replay));exports[width]=archive;print('WINNER',width,diagnostics,fresherrors,cost,flush=True)
 eight=next(w for w in winners if w['width']==8);pred=dict(pred_a_function=eight['function_error']<=.5*eight['initial_function_error'],pred_b_composed=eight['fresh_errors'][0]<=.18440681,pred_c_export=eight['cost']['stored_coefficients']==28912 and eight['cost']['products']==18 and eight['graph_replay']<1e-10);torch.save(dict(programs=exports,allfits=allfits,teacher_scale=source['teacher_scale']),P/'NONCENTRAL_BANK_REFIT_V1.pt');(P/'NONCENTRAL_BANK_REFIT_V1.json').write_text(json.dumps(dict(records=records,winners=winners,predictions=pred,seconds=time.perf_counter()-start,scope='Exact noncentral Gaussian bank function metric with fixed local root sensitivity. Complete quartic scored only after weight-space selection; same reused diagnostic/fresh panels, no new unseen evaluation.'),indent=2)+'\n');print(pred)
if __name__=='__main__':main()
