#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_balance pred_c_retention
"""Matched uniform/balanced continuation of shared producers, two starts each.
Profile-gated deterministic10..100steps, exact mixedloss/readout, normalizedfeatures.
Replay/export<1e-4, normal<1e-8; bothgain>=1%/textimprove; response/sensitivity<=10%.
"""
import os,sys,time,math,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from shared_training_gradient import backward
 from shared_quadratic_bank import normalize_bank
 from sparse_quartic_bank import gram as cg,native_cross as cx,features,entries,support
 from shared_gaussian_moments import gram as gg,native_cross as gx
 from noncentral_gaussian_cp import project_shifted
 from quartic_cp import directional
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  torch.manual_seed(12600);dtype=torch.float64;pars=[torch.nn.Parameter(torch.randn(576,12,dtype=dtype)) for _ in range(2)];a,b=normalize_bank(*[p.reshape(144,4,12) for p in pars]);pairs=support(144,512,1100);x=torch.randn(9,12,dtype=dtype);C=torch.randn(16,512,dtype=dtype)
  loss=(features(x,a,b,pairs)@C.T).square().mean();loss.backward();opt=torch.optim.Muon(pars,lr=.01,weight_decay=0.,adjust_lr_fn='match_rms_adamw');opt.step()
  assert entries(a,b,pairs,torch.randint(12,(11,4))).shape==(11,512)
  rec,don=torch.tensor([[0,1],[2,3]]).T;pred=features(x,a,b,pairs)@C.T;assert (pred[don,1]-pred[rec,1]).shape==(2,)
  weights=torch.linspace(.01,2,16,dtype=dtype);sw=weights.sqrt();assert (C*sw[:,None]).shape==(16,512)
  projection=(torch.randn(16,dtype=dtype),torch.randn(16,12,dtype=dtype),torch.randn(16,12,12,dtype=dtype))
  assert [tuple((t*sw.reshape((-1,)+(1,)*(t.ndim-1))).shape) for t in projection]==[(16,),(16,12),(16,12,12)]
  report=[dict(seed=s,arm=a) for s in [1101,1102] for a in ['uniform','balanced']]
  assert len(report)==4 and all(len([r for r in report if r['seed']==s])==2 for s in [1101,1102])
  assert all(torch.isfinite(p.grad).all() for p in pars)
  from check_balanced_shared_gradient import main as check
  # The independent control writes only its existing deterministic CPU receipt.
  check();print(json.dumps(dict(actual144x4_support512=True,output16=True,optimizer_step=True)));return
 frozen=json.loads((P/'BALANCED_SHARED_INPUTS_V1.json').read_text())
 for file,sha in frozen.items():assert hashlib.sha256((P/file).read_bytes()).hexdigest()==sha
 profile=json.loads((P/'SHARED_GRADIENT_PROFILE_V1.json').read_text());assert all(profile['predictions'].values()),'Native profile must pass before fitting'
 estimated_step=profile['gradient_seconds']+(profile['seconds']-profile['gradient_seconds'])/4;steps=min(100,int(600/estimated_step));assert steps>=10,'Profile implies fewer than10steps in600s; improve contractions first'
 assert steps==11,'Registered matched continuation requires11steps'
 from audit_root_matched_reader import CK
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'BALANCED_SHARED_FEATURES_NATIVE_V1.json';assert not out.exists();start=time.monotonic();scale=19054614563.464127;ridge=1e-6
 saved=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);cache=saved['projections']['covariance'];S=cache['whitener'].cuda();mu=saved['mean'].cuda();loc=torch.linalg.solve(S,mu);zero=tuple(a.cuda() for a in cache['zero_projection'])
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].cuda().double();vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 def w(l,n):return state[f'transformer.h.{l}.mlp.{n}.weight'].cuda().double()
 teacher=[readers.T@w(17,'Down')/scale,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')];tr=teacher[:4]+[teacher[4]@S,teacher[5]@S]
 with torch.no_grad():
  projection=project_shifted(tr,loc,zero);indices=torch.randint(1152,(4096,4),generator=torch.Generator().manual_seed(951)).cuda();eyeinput=torch.eye(1152,device='cuda',dtype=torch.float64);query=torch.cat([directional(*teacher,[eyeinput[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
  isotropic=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();truth=torch.cat([directional(*teacher,[z]*4) for z in isotropic.split(128)])
  text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=labels['target'].cuda()/scale;weights=labels['weight'].cuda()
  matched=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(matched['pairs_flat'],device='cuda').T;reference=torch.tensor([r['reference'] for r in matched['pair_rows']],device='cuda',dtype=torch.float64)/scale
 archive=json.loads((P/'GAUSSIAN_SUPPORT_EXCHANGE_NATIVE_V1.json').read_text());lambdas={r['seed']:r['coefficient_weight'] for r in archive['rows']};eye=torch.eye(512,device='cuda',dtype=torch.float64)
 @torch.no_grad()
 def fit(a,b,pairs,lam):
  g=(gg(a@S,b@S,pairs,a@mu,b@mu)+lam*cg(a,b,pairs))/(1+lam)
  x=(gx(tr,loc,projection,a@S,b@S,pairs,a@mu,b@mu)+lam*torch.cat([cx(teacher,a,b,t) for t in pairs.split(64,dim=1)],1))/(1+lam)
  c=torch.linalg.solve(g+ridge*eye,x.T).T;normal=float((c@(g+ridge*eye)-x).norm()/x.norm());assert normal<1e-8
  return float(-(output_weights[:,None]*c*x).sum()),c,normal
 @torch.no_grad()
 def metrics(a,b,pairs,c):
  pred=features(text,a,b,pairs)@c.T;sens=((weights*(pred-target).square()).sum(0)/(weights*target.square()).sum(0)).sqrt()
  return dict(text_error=float((pred-target).norm()/target.norm()),gaussian_error=float((features(isotropic,a,b,pairs)@c.T-truth).norm()/truth.norm()),sampled_coefficient_error=float((entries(a,b,pairs,indices)@c.T-query).norm()/query.norm()),root1_sensitivity_error=float(sens[1]),root1_same_token_error=float(((pred[don,1]-pred[rec,1])-reference).norm()/reference.norm()),feature_errors=((pred-target).square().sum(0)/target.square().sum(0)).sqrt().tolist(),small_feature_rms=float(((pred-target).square().sum(0)[4:]/target.square().sum(0)[4:]).mean().sqrt()))
 rows=[]
 for seed,arm in [(s,a) for s in [1101,1102] for a in ['uniform','balanced']]:
  metric=json.loads((P/'OUTPUT_BALANCED_CALIBRATION_METRIC_V1.json').read_text());output_weights=torch.tensor(metric['weights'],device='cuda',dtype=torch.float64) if arm=='balanced' else torch.ones(16,device='cuda',dtype=torch.float64);sw=output_weights.sqrt();tw=[teacher[0]*sw[:,None]]+teacher[1:];trw=[tr[0]*sw[:,None]]+tr[1:];pw=tuple(t*sw.reshape((-1,)+(1,)*(t.ndim-1)) for t in projection)
  source=torch.load(P/f'SHARED_MIXED_FEATURES_SEED{seed}_V1.pt',weights_only=True);pairs=source['pairs'].cuda();lam=lambdas[seed];params=[torch.nn.Parameter(t.cuda().double().reshape(576,1152)) for t in source['factors']];rate=.1*math.sqrt(4/1152);opt=torch.optim.Muon(params,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=None;history=[];maxnormal=0.
  for step in range(steps+1):
   with torch.no_grad():a,b=normalize_bank(*[p.reshape(144,4,1152) for p in params])
   loss,c,normal=fit(a,b,pairs,lam);maxnormal=max(maxnormal,normal);assert math.isfinite(loss)
   if step==0:
    normalizer=abs(loss);initial_loss=loss;initial=metrics(a,b,pairs,c)
    with torch.no_grad():old=features(text[:128],*[t.cuda().double() for t in source['factors']],pairs)@(source['coefficients'].cuda().double()/scale).T;ref=features(text[:128],a,b,pairs)@c.T;replay=float((ref-old).norm()/old.norm());assert replay<1e-4
   if best is None or loss<best[0]:best=(loss,step,a.clone(),b.clone(),c.clone())
   history.append(dict(step=step,objective=loss,elapsed=time.monotonic()-start));print(json.dumps(dict(seed=seed,arm=arm,**history[-1])),flush=True)
   if step==steps:break
   opt.zero_grad();backward(params,144,4,pairs,c*sw[:,None],tw,trw,loc,pw,S,mu,lam,normalizer,chunk=8);assert all(torch.isfinite(p.grad).all() for p in params);opt.step()
   for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/steps)))
  with torch.no_grad():
   loss,selected,a,b,c=best;final=metrics(a,b,pairs,c);ref=features(text[:128],a,b,pairs)@c.T;fp=features(text[:128].float(),a.float(),b.float(),pairs)@(c*scale).float().T;drift=float((fp.double()/scale-ref).norm()/ref.norm())
   path=P/f'BALANCED_SHARED_FEATURES_{arm.upper()}_SEED{seed}_V1.pt';torch.save(dict(factors=[a.cpu().float(),b.cpu().float()],pairs=pairs.cpu(),coefficients=(c*scale).cpu().float(),writer=writer.cpu().float(),seed=seed,degree=4),path)
   rows.append(dict(seed=seed,arm=arm,output_weights=output_weights.cpu().tolist(),coefficient_weight=lam,initial=initial,final=final,initial_objective=initial_loss,selected_objective=loss,relative_gain=(initial_loss-loss)/abs(initial_loss),selected_step=selected,history=history,baseline_replay=replay,normal_residual=maxnormal,export_error=drift,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  del opt,params,best,a,b,c
 contrasts=[]
 for seed in [1101,1102]:
  u=next(r for r in rows if r['seed']==seed and r['arm']=='uniform');b=next(r for r in rows if r['seed']==seed and r['arm']=='balanced')
  contrasts.append(dict(seed=seed,small_error_ratio=b['final']['small_feature_rms']/u['final']['small_feature_rms'],pooled_ratio=b['final']['text_error']/u['final']['text_error'],response_ratio=b['final']['root1_same_token_error']/u['final']['root1_same_token_error'],sensitivity_ratio=b['final']['root1_sensitivity_error']/u['final']['root1_sensitivity_error']))
 pred=dict(pred_a_integrity=all(r['baseline_replay']<1e-4 and r['normal_residual']<1e-8 and r['export_error']<1e-4 for r in rows),pred_b_balance=all(r['small_error_ratio']<=.85 for r in contrasts),pred_c_retention=all(r['pooled_ratio']<=1.25 and r['response_ratio']<=1.10 and r['sensitivity_ratio']<=1.10 for r in contrasts))
 result=dict(predictions=pred,contrasts=contrasts,steps=steps,estimated_step_seconds=estimated_step,rows=rows,products=1088,coefficients=1353728,integer_indices=1024,seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Fixed512rootpairs, learn144normalizedrank<=8quadratics (4bilinear terms) andexact16readouts. FixedGaussian/coefficient mixture, not maintained coefficientconstraint. Matcheduniform/balancedcontinuation: two learnedwarmstarts, same11steps/resetMuon/schedule. Calibrationlabelsecondmoments used only for outputmetric, capped1000relativeweight. No evaluationlabelsfitted. Openedevaluationdiagnostic, no finite-removal/OOD/semanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
