#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_values pred_c_component
"""Frozen shared144x4graph, exact shiftedGaussian readout with coefficientguard.
Two starts1101/1102, primarybudget1plusunconstrained. KKT<1e-8,replay/export<1e-4.
Bothprimarytext<=.8initial,query<=1.1initial. Bothroot1response/sensitivity<=10%.
Same1088products1353728coeff1024indices. No featureorgraph edits in thisfit.
"""
import os,sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P))
 from shared_gaussian_moments import gram as gaussian_gram,native_cross as gaussian_cross
 from sparse_quartic_bank import gram,native_cross,features,entries,support
 from shared_quadratic_bank import normalize_bank
 from coefficient_guarded_readout import GuardedReadout
 from noncentral_gaussian_cp import project_shifted
 from check_shared_gaussian_moments import controls
 from quartic_cp import directional
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  with torch.enable_grad():checked=controls()
  torch.manual_seed(11900);dtype=torch.float64;d=12
  U,V=normalize_bank(torch.randn(144,4,d,dtype=dtype),torch.randn(144,4,d,dtype=dtype));pairs=support(144,512,1100);bu=torch.randn(144,4,dtype=dtype);bv=torch.randn_like(bu)
  teacher=[torch.randn(*s,dtype=dtype) for s in [(16,7),(7,d),(7,d),(d,7),(7,d),(7,d)]];location=torch.randn(d,dtype=dtype);projection=project_shifted(teacher,location)
  G0=gram(U,V,pairs);X0=native_cross(teacher,U,V,pairs);G1=gaussian_gram(U,V,pairs,bu,bv);X1=gaussian_cross(teacher,location,projection,U,V,pairs,bu,bv)
  ridge=1e-6*torch.eye(512,dtype=dtype);solver=GuardedReadout(G0+ridge,X0,G1+ridge,X1)
  for ratio in [0,1,float('inf')]:
   C,info=solver.solve(ratio*solver.capture);assert C.shape==(16,512) and torch.isfinite(C).all()
   assert (features(torch.randn(7,d,dtype=dtype),U,V,pairs)@C.T).shape==(7,16)
  print(json.dumps(dict(control_cases=len(checked),actual144x4x512=True,budgets=[0,1,'unconstrained'])));return
 from audit_root_matched_reader import CK
 torch.backends.cuda.matmul.allow_tf32=False;out=P/'SHARED_GAUSSIAN_READOUT_NATIVE_V1.json';assert not out.exists();start=time.monotonic();scale=19054614563.464127
 saved=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);cache=saved['projections']['covariance'];S=cache['whitener'].cuda();mu=saved['mean'].cuda();zero=tuple(a.cuda() for a in cache['zero_projection']);location=torch.linalg.solve(S,mu);writer=saved['writer'].cuda()
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu')
 def weight(layer,name):return state[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 vocab=state['lm_head.weight'].cuda().double();uw=vocab@writer;readers=vocab.T@uw/uw.square().sum(0);del vocab,uw
 teacher=[readers.T@weight(17,'Down')/scale,weight(17,'Left'),weight(17,'Right'),weight(16,'Down')*state['transformer.h.17.lambdas'][0].item(),weight(16,'Left'),weight(16,'Right')]
 transformed=teacher[:4]+[teacher[4]@S,teacher[5]@S];projection=project_shifted(transformed,location,zero)
 indices=torch.randint(1152,(4096,4),generator=torch.Generator().manual_seed(951)).cuda();eye=torch.eye(1152,device='cuda',dtype=torch.float64);query=torch.cat([directional(*teacher,[eye[ii[:,s]] for s in range(4)]) for ii in indices.split(256)])
 x=torch.randn(1024,1152,generator=torch.Generator().manual_seed(939),dtype=torch.float64).cuda();truth=torch.cat([directional(*teacher,[z]*4) for z in x.split(128)])
 text=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].cuda().double();labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=labels['target'].cuda()/scale;weights=labels['weight'].cuda()
 pairdata=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(pairdata['pairs_flat'],device='cuda').T;reference=torch.tensor([r['reference'] for r in pairdata['pair_rows']],device='cuda',dtype=torch.float64)/scale
 rows=[];exports={};cache_exports={}
 for seed in [1101,1102]:
  source=torch.load(P/f'SPARSE_SUPPORT_EXCHANGE_SEED{seed}_V1.pt',weights_only=True);U,V=[a.cuda().double() for a in source['factors']];pairs=source['pairs'].cuda()
  G0=gram(U,V,pairs);X0=torch.cat([native_cross(teacher,U,V,p) for p in pairs.split(64,dim=1)],1)
  Uw,Vw=U@S,V@S;bu,bv=U@mu,V@mu
  G1=gaussian_gram(Uw,Vw,pairs,bu,bv);X1=gaussian_cross(transformed,location,projection,Uw,Vw,pairs,bu,bv)
  ridge=1e-6*torch.eye(512,device='cuda',dtype=G0.dtype);solver=GuardedReadout(G0+ridge,X0,G1+ridge,X1)
  phi=features(text,U,V,pairs);phi_x=features(x,U,V,pairs);phi_query=entries(U,V,pairs,indices)
  archive_prediction=phi@(source['coefficients'].cuda().double()/scale).T;replay=float((phi@solver.C0.T-archive_prediction).norm()/archive_prediction.norm());assert replay<1e-4
  readouts={}
  for ratio in [0.,1.,None]:
   budget=float('inf') if ratio is None else ratio*solver.capture;C,info=solver.solve(budget);pred=phi@C.T
   sens=((weights*(pred-target).square()).sum(0)/(weights*target.square()).sum(0)).sqrt()
   fp=features(text[:128].float(),U.float(),V.float(),pairs)@(C*scale).float().T;drift=float((fp.double()/scale-pred[:128]).norm()/pred[:128].norm())
   row=dict(seed=seed,ratio=ratio,budget=None if ratio is None else budget,solver=info,baseline_replay=replay,export_error=drift,text_error=float((pred-target).norm()/target.norm()),gaussian_error=float((phi_x@C.T-truth).norm()/truth.norm()),sampled_coefficient_error=float((phi_query@C.T-query).norm()/query.norm()),root1_sensitivity_error=float(sens[1]),root1_same_token_error=float(((pred[don,1]-pred[rec,1])-reference).norm()/reference.norm()))
   rows.append(row);readouts[str(ratio)]=(C*scale).cpu().float();print(json.dumps(row),flush=True)
  exports[seed]=dict(factors=source['factors'],pairs=source['pairs'],readouts=readouts,writer=writer.cpu().float(),degree=4)
  cache_exports[seed]=dict(coefficient_gram=G0.cpu(),coefficient_cross=X0.cpu(),gaussian_gram=G1.cpu(),gaussian_cross=X1.cpu(),scale=scale)
 base={r['seed']:r for r in rows if r['ratio']==0.};primary=[r for r in rows if r['ratio']==1.]
 pred=dict(pred_a_integrity=all(r['baseline_replay']<1e-4 and r['export_error']<1e-4 and r['solver'].get('stationarity',0)<1e-8 and (r['budget'] is None or r['solver']['displacement']<=r['budget']+1e-8*(1+r['budget'])) for r in rows),pred_b_values=all(r['text_error']<=.8*base[r['seed']]['text_error'] and r['sampled_coefficient_error']<=1.1*base[r['seed']]['sampled_coefficient_error'] for r in primary),pred_c_component=all(r['root1_same_token_error']<=.1 and r['root1_sensitivity_error']<=.1 for r in primary))
 torch.save(dict(programs=exports,scope='Eachcandidate usesone readout andsame1088products1353728coeff1024indices.'),P/'SHARED_GAUSSIAN_READOUT_NATIVE_V1.pt');torch.save(cache_exports,P/'SHARED_GAUSSIAN_READOUT_GRAMS_V1.pt')
 result=dict(predictions=pred,rows=rows,seconds=time.monotonic()-start,peak_memory_bytes=torch.cuda.max_memory_allocated(),scope='Exact fixed-feature shiftedGaussian readout, coefficientbudgetrelativecapturedscore notfullnorm. Fixed144quadratics and learned512pairs. Inputstatistics/readers data-informed; no empiricaloutput fit, newfeatures, OOD or nativefinite-removal claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
