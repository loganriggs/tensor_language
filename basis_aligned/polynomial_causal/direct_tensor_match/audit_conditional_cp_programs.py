"""Constructive low-input-rank conditional programs; weights and covariance only."""
import hashlib,json,time
from pathlib import Path
import torch
from conditional_quartic_cp import construct,evaluate,price
from gaussian_cp_derivative_gram import gram
from mixed_gaussian_cp import gram_dynamic
P=Path(__file__).resolve().parent;SCALE=19054614563.464127

def cp_value(C,f,x):return torch.stack([x@a.T for a in f]).prod(0)@C.T

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double()
 opened=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=opened['rows'].double();y=opened['target'].double()/SCALE
 old=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();lab=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1];target=lab['target'].double()/SCALE;weight=lab['weight'].double()
 matched=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(matched['pairs_flat']).T;reference=torch.tensor([r['reference'] for r in matched['pair_rows']],dtype=torch.float64)/SCALE
 z=torch.randn(4096,1152,generator=torch.Generator().manual_seed(16020),dtype=torch.float64);probes=z@S.T+mu;rows=[]
 for seed in [1001,1002]:
  a=torch.load(P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',weights_only=True);f=[v.double() for v in a['factors']];C=a['coefficients'].double()/SCALE;fw=[a@S for a in f];bias=[a@mu for a in f]
  K=gram(fw,bias,input_coefficients=C);K=(K+K.T)/2;ev,Q=torch.linalg.eigh(K);assert ev.min()>-1e-9*ev.max();Q=Q.flip(1);ev=ev.clamp_min(0).flip(0)
  energy=((C.T@C)*gram_dynamic(fw,bias,fw,bias)).sum()
  parent_probe=cp_value(C,f,probes)
  for rank in [64,128,256]:
   program=construct(f,C,S,mu,Q[:,:rank]);pred=torch.cat([evaluate(program,xx) for xx in x.split(2048)]);po=evaluate(program,old);pg=evaluate(program,probes);plug=evaluate(program,probes,False)
   cost=price(program);cost['stored_floats_with_fixed_writer']=cost['stored_floats']+1152*16
   fe=((pred-y).square().sum(0)/y.square().sum(0)).sqrt();sensitivity=((weight[:,1]*(po[:,1]-target[:,1]).square()).sum()/(weight[:,1]*target[:,1].square()).sum()).sqrt()
   r=dict(seed=seed,input_rank=rank,cost=cost,parent_gaussian_probe_error=float((pg-parent_probe).norm()/parent_probe.norm()),plug_in_without_corrections_error=float((plug-parent_probe).norm()/parent_probe.norm()),parent_gaussian_error_upper_bound=float((ev[rank:].sum()/energy).sqrt()),opened256document_value_error=float((pred-y).norm()/y.norm()),opened_feature_errors=fe.tolist(),old_root1_response_error=float(((po[don,1]-po[rec,1])-reference).norm()/reference.norm()),old_root1_sensitivity_error=float(sensitivity))
   archive={k:([v.float() for v in value] if isinstance(value,list) else value.float()) for k,value in program.items()}
   archive['coefficients']=archive['coefficients']*SCALE;archive['constant']=archive['constant']*SCALE;archive['writer']=a['writer'].float();archive['seed']=seed;archive['input_rank']=rank
   fp=torch.cat([evaluate(archive,xx.float()).double()/SCALE for xx in x.split(2048)])
   drift=float((fp-pred).norm()/pred.norm());assert drift<1e-4,drift
   path=P/f'CONDITIONAL_CP_SEED{seed}_RANK{rank}_V1.pt';torch.save(archive,path)
   r['export_error']=drift;r['artifact_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
   rows.append(r);print(json.dumps(r),flush=True)
 result=dict(rows=rows,seconds=time.monotonic()-start,scope='Explicit conditional mean of two fittedCP512parents, Gaussianmean/covariance andexactgradient-subspace only; no textlabelsfit. Rank64/128/256fixed beforeconstruction. Previouslyopened256documents andoldresponsepanel diagnostic, not freshvalidation.4096independentartificialGaussianprobes estimateparenterror, exactderivativebound separately. Newdegree<=4program addsquadratic/constantcorrections;3584variableproducts, reducedlinearstorage. No fullmodel/OOD/manipulationadoption.')
 (P/'CONDITIONAL_CP_PROGRAMS_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
