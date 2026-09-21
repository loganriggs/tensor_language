#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_learning pred_c_component
"""Joint CP512 directions under fixed coefficient-plus-shiftedGaussian metric.
Two archived starts1001/1002,100Muonsteps. Baselinereplay<1e-5/export<1e-4.
Both objectivegain>=1%,text<=1.1initial. Bothroot1response<=10% andsensitivity
<=1.1*.145391145. Price1536products2385920coeff. No hard coefficient guard.
"""
import os,sys,time,math,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from quartic_cp import directional,cp_entries,cp_gram
 from quartic_cp_profile import normalize_factors
 from mixed_gaussian_cp import native_mixed_objective
 from noncentral_gaussian_cp import project_shifted
 from check_mixed_gaussian_cp import controls
 torch.set_num_threads(2)
 def values(f,c,x):
  z=x@f[0].T
  for a in f[1:]:z=z*(x@a.T)
  return z@c.T
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checked=controls();torch.manual_seed(11400);dtype=torch.float64;d=12
  teacher=[torch.randn(*s,dtype=dtype) for s in [(16,7),(7,d),(7,d),(d,7),(7,d),(7,d)]]
  S=torch.eye(d,dtype=dtype);mu=torch.randn(d,dtype=dtype);projection=project_shifted(teacher,mu)
  params=[torch.randn(512,d,dtype=dtype,requires_grad=True) for _ in range(4)];f=normalize_factors(params)
  loss,c=native_mixed_objective(teacher,teacher,mu,projection,S,mu,f,1000.);loss.backward()
  assert c.shape==(16,512) and all(torch.isfinite(a.grad).all() for a in params)
  assert values(f,c,torch.randn(5,d,dtype=dtype)).shape==(5,16)
  assert cp_entries(c,f,torch.randint(d,(9,4))).shape==(9,16)
  print(json.dumps(dict(control_cases=len(checked),actual512_backward=True,steps=100,seeds=[1001,1002])));return
 from audit_root_matched_reader import CK
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'MIXED_CP_FEATURES_NATIVE_V1.json';assert not out.exists();scale=19054614563.464127
 saved=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);cache=saved['projections']['covariance'];S=cache['whitener'].cuda();mu=saved['mean'].cuda();zero=tuple(a.cuda() for a in cache['zero_projection']);location=torch.linalg.solve(S,mu);writer=saved['writer'].cuda()
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 teacher=[readers.T@weight(17,'Down')/scale,weight(17,'Left'),weight(17,'Right'),weight(16,'Down')*state['transformer.h.17.lambdas'][0].item(),weight(16,'Left'),weight(16,'Right')]
 with torch.no_grad():
  transformed=teacher[:4]+[teacher[4]@S,teacher[5]@S];projection=project_shifted(transformed,location,zero)
  indices=torch.randint(1152,(4096,4),generator=torch.Generator().manual_seed(951)).cuda();eye=torch.eye(1152,device='cuda',dtype=torch.float64)
  query=torch.cat([directional(*teacher,[eye[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
  gaussian=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();truth=torch.cat([directional(*teacher,[x]*4) for x in gaussian.split(128)])
  text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=labels['target'].cuda()/scale;weights=labels['weight'].cuda()
  pairdata=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(pairdata['pairs_flat'],device='cuda').T;reference=torch.tensor([r['reference'] for r in pairdata['pair_rows']],device='cuda',dtype=torch.float64)/scale
 guarded=json.loads((P/'COEFFICIENT_GUARDED_NATIVE_V1.json').read_text());primary={r['seed']:r for r in guarded['rows'] if r['ratio']==1.}
 archive=torch.load(P/'COEFFICIENT_GUARDED_NATIVE_V1.pt',weights_only=True)['programs']
 def metrics(f,c):
  with torch.no_grad():
   pred=values(f,c,text);sens=((weights*(pred-target).square()).sum(0)/(weights*target.square()).sum(0)).sqrt()
   return dict(text_error=float((pred-target).norm()/target.norm()),gaussian_error=float((values(f,c,gaussian)-truth).norm()/truth.norm()),sampled_coefficient_error=float((cp_entries(c,f,indices)-query).norm()/query.norm()),root1_sensitivity_error=float(sens[1]),root1_same_token_error=float(((pred[don,1]-pred[rec,1])-reference).norm()/reference.norm()))
 def coefficient_score(f,c):
  with torch.no_grad():return float((c*(c@cp_gram(f,f))).sum()-2*(c*directional(*teacher,f).T).sum()+1e-6*c.square().sum())
 rows=[]
 for seed in [1001,1002]:
  source=torch.load(P/f'QUARTIC_CP512_SEED{seed}_V2.pt',weights_only=True)
  params=[torch.nn.Parameter(a.cuda().double()) for a in source['factors']]
  lam=primary[seed]['solver']['multiplier'];rate=.1*math.sqrt(4/1152)
  opt=torch.optim.Muon(params,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw');history=[];best=None
  for step in range(101):
   f=normalize_factors(params);loss,c=native_mixed_objective(teacher,transformed,location,projection,S,mu,f,lam)
   value=float(loss.detach());assert math.isfinite(value)
   if step==0:
    normalizer=abs(value);assert normalizer>0;initial=metrics(f,c);initial_value=value;initial_coefficient=coefficient_score(f,c)
    with torch.no_grad():
     old=archive[seed];ref=values([a.cuda().double() for a in old['factors']],old['readouts']['1.0'].cuda().double()/scale,text[:128]);replay=float((values(f,c,text[:128])-ref).norm()/ref.norm());assert replay<1e-5,replay
   if best is None or value<best[0]:best=(value,step,[a.detach().clone() for a in f],c.detach().clone())
   if step%10==0:
    row=dict(seed=seed,step=step,objective=value,elapsed=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated());history.append(row);print(json.dumps(row),flush=True)
   if step==100:break
   opt.zero_grad();(loss/normalizer).backward();assert all(torch.isfinite(a.grad).all() for a in params)
   if step==0:assert torch.cuda.max_memory_allocated()<26*1024**3,'Native backward exceeds declared memory budget'
   opt.step()
   for group in opt.param_groups:group['lr']=rate*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/100)))
  with torch.no_grad():
   f,c=best[2:];final=metrics(f,c);ref=values(f,c,text[:128]);fp=values([a.float() for a in f],(c*scale).float(),text[:128].float()).double()/scale;drift=float((fp-ref).norm()/ref.norm())
   artifact=dict(factors=[a.cpu().float() for a in f],coefficients=(c*scale).cpu().float(),writer=writer.cpu().float(),seed=seed,degree=4,coefficient_weight=lam)
   path=P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt';torch.save(artifact,path)
   rows.append(dict(seed=seed,coefficient_weight=lam,initial=initial,final=final,initial_objective=initial_value,selected_objective=best[0],relative_objective_improvement=(initial_value-best[0])/abs(initial_value),selected_step=best[1],coefficient_objective_change=coefficient_score(f,c)-initial_coefficient,baseline_replay=replay,export_error=drift,history=history,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  del opt,params,best,f,c,loss
 pred=dict(pred_a_integrity=all(r['baseline_replay']<1e-5 and r['export_error']<1e-4 for r in rows),pred_b_learning=all(r['relative_objective_improvement']>=.01 and r['final']['text_error']<=1.1*r['initial']['text_error'] for r in rows),pred_c_component=all(r['final']['root1_same_token_error']<=.1 and r['final']['root1_sensitivity_error']<=1.1*.145391145 for r in rows))
 result=dict(predictions=pred,rows=rows,products=1536,coefficients=2385920,seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Fixed weighted coefficient-plus-shiftedGaussian objective, joint nonconvex feature learning with profiled readout. Lambda derived from preceding weight/statistics-only budget1 solve and held fixed. NOT a maintained coefficient constraint. Opened evaluation only; no OOD/finite-removal/semantic claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
