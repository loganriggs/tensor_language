"""Compare frozen graph edit errors under Gaussian and normalized probes."""
import json,time
import torch
from audit_conditional_residual_accounting import P,SCALE,load

def evaluate(x,fs,C):
 return torch.stack([x@f.T for f in fs]).prod(0)@C.T

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double()
 radius=(S.square().sum()+mu.square().sum()).sqrt()
 source=json.loads((P/'CP_QUADRATIC_REUSE_V1.json').read_text());exact=json.loads((P/'QUADRATIC_EDIT_INTERFERENCE_V1.json').read_text())
 text=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True)['rows'].double();norms=text.norm(dim=1)
 rows=[]
 for proposal in source['proposals']:
  seed=proposal['seed'];p,sha=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');assert sha==proposal['parent_sha256'];fs=p['factors'];C=p['coefficients']/SCALE
  programs=[]
  for count in [64,128]:
   nf=[f.clone() for f in fs];nc=C.clone()
   for i,j,beta in proposal['selected'][:count]:
    for slot in [0,1]:nf[2*(i//512)+slot][i%512]=fs[2*(j//512)+slot][j%512]
    nc[:,i%512]*=beta
   programs.append((count,nf,nc))
  batches={64:[],128:[]};norm_records=[];replay=0.
  for batch in range(4):
   gen=torch.Generator().manual_seed(37100+batch);x=torch.randn(8192,1152,dtype=torch.float64,generator=gen)@S.T+mu
   scale=radius/x.norm(dim=1);weight=scale.pow(8);parent=evaluate(x,fs,C);parent_energy=parent.square().sum(1)
   norm_records.append(dict(mean=float(x.norm(dim=1).mean()),std=float(x.norm(dim=1).std()),min=float(x.norm(dim=1).min()),max=float(x.norm(dim=1).max())))
   for count,nf,nc in programs:
    pred=evaluate(x,nf,nc);error=(pred-parent).square().sum(1)
    direct=evaluate(x[:16]*scale[:16,None],nf,nc);via=pred[:16]*scale[:16,None].pow(4)
    replay=max(replay,float((direct-via).norm()/direct.norm()));assert replay<1e-10
    batches[count].append(dict(raw_error_energy=float(error.sum()),raw_parent_energy=float(parent_energy.sum()),normalized_error_energy=float((error*weight).sum()),normalized_parent_energy=float((parent_energy*weight).sum()),raw_relative=float((error.sum()/parent_energy.sum()).sqrt()),normalized_relative=float(((error*weight).sum()/(parent_energy*weight).sum()).sqrt())))
  for count,bs in batches.items():
   raw=(sum(b['raw_error_energy'] for b in bs)/sum(b['raw_parent_energy'] for b in bs))**.5
   normalized=(sum(b['normalized_error_energy'] for b in bs)/sum(b['normalized_parent_energy'] for b in bs))**.5
   ref=next(r for r in exact['rows'] if r['seed']==seed and r['count']==count)
   row=dict(seed=seed,count=count,raw_relative=raw,normalized_relative=normalized,normalized_over_raw=normalized/raw,exact_gaussian_relative=ref['gaussian_relative_edit'],mc_over_exact=raw/ref['gaussian_relative_edit'],text_relative=ref['text_relative_edit'],batches=bs,homogeneity_replay=replay)
   rows.append(row);print({k:v for k,v in row.items() if k!='batches'},flush=True)
 result=dict(rows=rows,radius=float(radius),text_norm=dict(mean=float(norms.mean()),std=float(norms.std()),min=float(norms.min()),max=float(norms.max())),gaussian_norm_batches=norm_records,seconds=time.monotonic()-start,scope='Artificial probes with mean/covariance derived from calibration inputs. Four independent batches; frozen programs. Radial normalization also changes mean/covariance; not evidence of on-text causal fidelity.')
 (P/'GAUSSIAN_RADIAL_EDIT_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
