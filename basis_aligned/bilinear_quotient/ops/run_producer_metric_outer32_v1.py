#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;native4608Gram,1152spectralproblems,128cachedvectors.
"""pred_a full metric rank, old replay/H-orthogonality/paired formula<=1e-8;
pred_b each paired error improves>=10%; pred_c all family swaps<=.1/sign>=.9/live>=4;
pred_d all family removal CE<=.02 and write error<=.05.
Equal32terms/output,76096fittedvalues plus15925248nativeparentvalues. No data fit.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from producer_metric_spectral_v1 import factor
from quartic_frozen_native_score_v2 import score
STEM='PRODUCER_METRIC_OUTER32_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,cached_vectors=128,gram_dimension=4608,outer_dimension=1152)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists();signal.alarm(600)
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
 p=torch.load(P/'QUARTIC_OUTER32_V1_PROGRAM.pt',weights_only=True);writer=p['output_writers'].cuda()
 u=state['lm_head.weight'].double().cuda();uw=u@writer;reader=u.T@uw-len(u)*u.mean(0)[:,None]*uw.mean(0)[None,:];del u
 l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
 scale=float(state['transformer.h.17.lambdas'][0]);assert scale==p['producer_scale']
 h=scale**2*(d0@atom_gram(l0,r0)@d0.T)
 he,hv=torch.linalg.eigh(h);hs=(hv*he.clamp_min(0).sqrt())@hv.T
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
 ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports'];den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
 producer=((x@l0.T)*(x@r0.T))@d0.T*scale
 modes=[];weights=[];reports=[];errors=[];baseline_scalar=[]
 for m in range(2):
  raw=l1.T@((d1.T@reader[:,m])[:,None]*r1);matrix=(raw+raw.T)/2
  a,mu,report=factor(matrix,h,32);modes.append(a);weights.append(mu)
  old_a=p['output_readers'][m].cuda();old_mu=p['outer_weights'][m].cuda()
  target=hs@matrix@hs;approx=hs@((a*mu)@a.T)@hs;old=hs@((old_a*old_mu)@old_a.T)@hs
  direct=float((target-approx).norm()/target.norm());baseline=float((target-old).norm()/target.norm())
  orth=float((a.T@h@a-torch.eye(32,device='cuda')).norm()/32**.5)
  errors.extend([abs(direct-report['paired_coefficient_relative_error']),orth])
  report.update(mode=m,baseline_paired_error=baseline,paired_error_improvement=1-direct/baseline,H_orthogonality_error=orth)
  reports.append(report);baseline_scalar.append(((producer@old_a).square()*old_mu).sum(-1))
 scalar=torch.stack([((producer@a).square()*mu).sum(-1) for a,mu in zip(modes,weights)],1)
 write=(scalar@writer.T/den[:,None]).cpu();oldwrite=(torch.stack(baseline_scalar,1)@writer.T/den[:,None]).cpu()
 ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0]
 oldreceipt=json.loads((P/'QUARTIC_OUTER32_NATIVE_EFFECTS_V1.json').read_text());errors.append(abs(float((oldwrite-ref).norm()/ref.norm())-oldreceipt['reports'][1]['physical_error']))
 rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
 prior=next(r for r in json.loads((P/'QUARTIC_GROUP_LIFTED_NATIVE_V1.json').read_text())['effects'] if r['name']=='lifted')
 effects=score([ref,oldwrite,write],ports['pre']+ports['native_output'],rows,state['lm_head.weight'].float(),prior)
 families=[]
 for family in sorted({r['family'] for r in rows}):
  ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
  families.append(dict(family=family,write_relative_error=float((write[ep]-ref[ep]).norm()/ref[ep].norm())))
 torch.save(dict(seed=p['seed'],output_readers=torch.stack(modes).cpu(),outer_weights=torch.stack(weights).cpu(),output_writers=writer.cpu(),producer_scale=scale,native_write=write),ap)
 result={'pred_a':max(errors)<=1e-8 and all(r['metric_rank']==1152 for r in reports) and effects['pred_a'],
  'pred_b':all(r['paired_error_improvement']>=.1 for r in reports),
  'pred_c':effects['pred_b'],'pred_d':effects['pred_c'] and all(f['write_relative_error']<=.05 for f in families)}
 result.update(dict(coefficient_reports=reports,identity_errors=errors,effects=effects,families=families,artifact_sha256=digest(ap),fitted_floats=76096,native_parent_floats=15925248,wall_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope='Paired coefficient metric truncation, exact native producer; original developmental effects only. Not full-symmetric quartic optimality, fresh validation, or circuit promotion.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='effects'},indent=2),flush=True);assert result['pred_a']
if __name__=='__main__':main()
