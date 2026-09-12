#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;4full-U spectra,128cachednativevectors,no fitting.
"""pred_a prior full-U spectrum/metric/projector replay<=1e-8;
pred_b centered top32 output capture improves>=10% relatively after producer folding;
pred_c complete-path developmental swaps<=.1/sign>=.9/live>=4, removals<=.02, writes<=.05.
Output-rank structural/sufficiency probe, not semantic circuits or full-model adoption.
Primary centered producer metric; centered identity-metric baseline.600sec limit.
"""
import os,sys,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quadratic_producer_projection_v1 import atom_gram
from audit_fullu_output_functions_v1 import spectrum
from quartic_frozen_native_score_v2 import score
STEM='FULLU_PAIRED_PRODUCER_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,spectra=4,output_rank=32,cached_vectors=128)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAMS.pt');assert not out.exists() and not ap.exists();signal.alarm(600)
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),mmap=True,weights_only=True)
 u=state['lm_head.weight'].double().cuda()
 l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
 scale=float(state['transformer.h.17.lambdas'][0]);h=scale**2*(d0@atom_gram(l0,r0)@d0.T);he,hv=torch.linalg.eigh(h);hs=(hv*he.clamp_min(0).sqrt())@hv.T
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
 ports=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports'];den=ports['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
 producer=((x@l0.T)*(x@r0.T))@d0.T*scale;products=(producer@l1.T)*(producer@r1.T);ref=(products@d1.T/den[:,None]).cpu()
 prior=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())
 reports=[];programs=[];writes=[];errors=[]
 for centered in (False,True):
  output=u-u.mean(0) if centered else u
  root=torch.linalg.cholesky(output.T@output)
  for folded in (False,True):
   left,right=(l1@hs,r1@hs) if folded else (l1,r1)
   values,vectors,weighted,cov=spectrum(output,left,right,d1)
   total=float(values.sum());captured={str(k):float(values[:k].sum()/total) for k in (1,2,8,16,32,64,128,256,512,1152)}
   name=('centered' if centered else 'full')+('_producer' if folded else '_identity')
   errors.append(float((cov-(vectors*values)@vectors.T).norm()/cov.norm()))
   if not centered and not folded:
    errors.extend(abs(captured[str(k)]-prior['captured_energy'][str(k)]) for k in (1,32,128,1152))
   report=dict(name=name,total_paired_coefficient_energy=total,captured_energy=captured)
   if centered:
    basis=torch.linalg.solve_triangular(root.T,vectors[:,:32],upper=True);down=vectors[:,:32].T@weighted
    projected=products@down.T@basis.T/den[:,None]
    direct=((products@d1.T)@root@vectors[:,:32])@basis.T/den[:,None]
    errors.append(float((projected-direct).norm()/direct.norm()))
    errors.append(float(((output@basis).T@(output@basis)-torch.eye(32,device='cuda')).norm()/32**.5))
    writes.append(projected.cpu());programs.append(dict(name=name,output_writers=basis.cpu(),latent_down=down.cpu(),producer_scale=scale,native_write=projected.cpu()))
   reports.append(report)
 # Use the existing native float32 tail with a new complete-path reference.
 torch.set_default_dtype(torch.float32)
 rows=json.loads((P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json').read_text())['rows']
 effects=score([ref,*writes],ports['pre']+ports['native_output'],rows,state['lm_head.weight'].float())
 families=[]
 for family in sorted({r['family'] for r in rows}):
  ids=torch.tensor([i for i,r in enumerate(rows) if r['family']==family]);ep=(2*ids[:,None]+torch.tensor([0,1])).flatten()
  families.append(dict(family=family,write_relative_error=float((writes[1][ep]-ref[ep]).norm()/ref[ep].norm())))
 improvement=reports[3]['captured_energy']['32']/reports[2]['captured_energy']['32']-1
 torch.save(dict(programs=programs,reference_write=ref),ap)
 result={'pred_a':max(errors)<=1e-8 and effects['pred_a'],'pred_b':improvement>=.1,'pred_c':effects['pred_b'] and effects['pred_c'] and all(f['write_relative_error']<=.05 for f in families)}
 result.update(dict(reports=reports,centered_top32_relative_capture_gain=improvement,identity_errors=errors,effects=effects,families=families,
  folded_decoder_floats=184320,remaining_native_producer_and_outer_reader_floats=26542080,full_native_two_layer_path_floats=31850496,
  artifact_sha256=digest(ap),wall_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope='Full and centered unembedding output-mode probe for complete bias-free two-MLP interaction path, paired coefficient metric. Native teacher background retained; no sparse inner factorization, corpus OOD, semantic circuit or deployed model claim.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='effects'},indent=2),flush=True);assert result['pred_a']
if __name__=='__main__':main()
