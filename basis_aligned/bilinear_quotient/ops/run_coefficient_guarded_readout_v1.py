#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_tradeoff pred_c_component
"""Fixed CP512,exact shiftedGaussian fitundercoefficient lossbudget.
KKT/feasibility<1e-8,export<1e-4. Primarybudget1text<=.9base/query<=1.1base.
Root1response<=.1/sensitivity<=1.1*.145391145 bothstarts. Same1536/2385920price.
"""
import os,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from noncentral_gaussian_cp import project_shifted,cross,gram
 from coefficient_guarded_readout import GuardedReadout
 from check_coefficient_guarded_readout import controls
 from quartic_cp import directional,cp_gram,cp_entries
 from audit_root_matched_reader import CK
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checked=controls();torch.manual_seed(11200);a=torch.randn(520,512,dtype=torch.float64);b=torch.randn(520,512,dtype=torch.float64);G0=a.T@a+.1*torch.eye(512);G=b.T@b+.1*torch.eye(512);X0=torch.randn(16,512,dtype=torch.float64);X=torch.randn_like(X0);s=GuardedReadout(G0,X0,G,X)
  for budget in [0.,s.capture,float('inf')]:
   c,info=s.solve(budget);assert c.shape==(16,512) and torch.isfinite(c).all()
  print(json.dumps(dict(control_cases=len(checked),shape512=True,budget_ratios=[0,.01,.1,1,10,100,'unconstrained'])));return
 out=P/'COEFFICIENT_GUARDED_NATIVE_V1.json';assert not out.exists();start=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False;scale=19054614563.464127
 saved=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);cache=saved['projections']['covariance'];S=cache['whitener'].cuda();mu=saved['mean'].cuda();zero=tuple(a.cuda() for a in cache['zero_projection']);location=torch.linalg.solve(S,mu);writer=saved['writer'].cuda()
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 teacher=[readers.T@weight(17,'Down')/scale,weight(17,'Left'),weight(17,'Right'),weight(16,'Down')*state['transformer.h.17.lambdas'][0].item(),weight(16,'Left'),weight(16,'Right')]
 with torch.no_grad():
  t=teacher[:4]+[teacher[4]@S,teacher[5]@S];projection=project_shifted(t,location,zero)
  indices=torch.randint(1152,(4096,4),generator=torch.Generator().manual_seed(951)).cuda();eye=torch.eye(1152,device='cuda',dtype=torch.float64);query=torch.cat([directional(*teacher,[eye[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
  x=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();truth=torch.cat([directional(*teacher,[z,z,z,z]) for z in x.split(128)])
  text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=labels['target'].cuda()/scale;weights=labels['weight'].cuda()
  pairdata=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];pairs=torch.tensor(pairdata['pairs_flat'],device='cuda');rec,don=pairs.T;reference=torch.tensor([r['reference'] for r in pairdata['pair_rows']],device='cuda',dtype=torch.float64)/scale
  def features(f,x):
   z=x@f[0].T
   for a in f[1:]:z=z*(x@a.T)
   return z
  rows=[];exports={}
  for seed in [1001,1002]:
   source=torch.load(P/f'QUARTIC_CP512_SEED{seed}_V2.pt',weights_only=True);f=[a.cuda().double() for a in source['factors']];fw=[a@S for a in f];bias=[a@mu for a in f]
   rawG0=cp_gram(f,f);X0=directional(*teacher,f).T;rawG=gram(fw,bias,fw,bias);X=cross(t,location,projection,fw,bias);ridge=1e-6*torch.eye(512,device='cuda',dtype=rawG.dtype);solver=GuardedReadout(rawG0+ridge,X0,rawG+ridge,X)
   endpoint_error=float((solver.C0-torch.linalg.solve(rawG0+ridge,X0.T).T).norm()/solver.C0.norm());phi_text=features(f,text);phi_x=features(f,x);phi_query=cp_entries(torch.eye(512,device='cuda',dtype=rawG.dtype),f,indices);c0=solver.C0;raw0=float((c0*(c0@rawG0)).sum()-2*(c0*X0).sum());readouts={}
   for ratio in [0.,.01,.1,1.,10.,100.,None]:
    budget=float('inf') if ratio is None else ratio*solver.capture;c,info=solver.solve(budget);pred=phi_text@c.T
    weighted=((weights*(pred-target).square()).sum(0)/(weights*target.square()).sum(0)).sqrt()
    fp=features([a.float() for a in f],text[:128].float())@(c*scale).float().T;drift=float((fp.double()/scale-pred[:128]).norm()/pred[:128].norm())
    row=dict(seed=seed,ratio=ratio,endpoint_error=endpoint_error,budget=None if ratio is None else budget,captured_coefficient_score=solver.capture,solver=info,unregularized_coefficient_loss_increase=float((c*(c@rawG0)).sum()-2*(c*X0).sum())-raw0,text_error=float((pred-target).norm()/target.norm()),gaussian_error=float((phi_x@c.T-truth).norm()/truth.norm()),sampled_coefficient_error=float((phi_query@c.T-query).norm()/query.norm()),root1_sensitivity_error=float(weighted[1]),root1_same_token_error=float(((pred[don,1]-pred[rec,1])-reference).norm()/reference.norm()),export_error=drift)
    rows.append(row);readouts[str(ratio)]=(c*scale).cpu().float();print(json.dumps(row),flush=True)
   exports[seed]=dict(factors=source['factors'],readouts=readouts,writer=writer.cpu().float(),degree=4)
  primary=[r for r in rows if r['ratio']==1.];base={r['seed']:r for r in rows if r['ratio']==0.}
 pred=dict(pred_a_integrity=all(r['endpoint_error']<1e-8 and r['solver'].get('stationarity',0)<1e-8 and (r['budget'] is None or r['solver']['displacement']<=r['budget']+1e-8*(1+r['budget'])) and r['export_error']<1e-4 for r in rows),pred_b_tradeoff=all(r['text_error']<=.9*base[r['seed']]['text_error'] and r['sampled_coefficient_error']<=1.1*base[r['seed']]['sampled_coefficient_error'] for r in primary),pred_c_component=all(r['root1_same_token_error']<=.1 and r['root1_sensitivity_error']<=1.1*.145391145 for r in primary))
 torch.save(dict(programs=exports,scope='Shared research package; each deployedcandidate usesone readout and1536products2385920coeff.'),P/'COEFFICIENT_GUARDED_NATIVE_V1.pt')
 result=dict(predictions=pred,rows=rows,seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Exact fixed-feature convex frontier; budgets are regularized coefficient deterioration relativecapturedscore, notrelativefulltensorerror. Openedstates andlocal sensitivity, no OOD/nativefiniteintervention claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
