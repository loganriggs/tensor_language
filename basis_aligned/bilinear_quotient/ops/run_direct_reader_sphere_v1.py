#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_missing pred_c_effect
"""Equal-reader sphere probes; READER_SPHERE_PLAN_V1.md.
pred_a_instrument: invariance<1e-5, nativegradientprecision<1e-4, FD<.02.
pred_b_missing: invisible tangent gradient energy fraction>.1 bothpanels.
pred_c_effect: .05angle native paired RMS>.01scalarstd bothpanels.
Null: relevant local dependence captured by fixed readers; or instrument failure.
Price512states, 2directionfamilies x3angles, foldedweightcontractions only;
student unchanged13916scalarcoeff10products, no new native model replacement.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(rows=256,batch=64,angles=[.001,.01,.05],families=['gradient','random'],seed=260954)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from reader_sphere_probe import scalar_value_gradient,invisible_tangent,spherical_pair,toy_check
 from frozen_program_evaluation import load_teacher
 from extract_scalar_modes import evaluate
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'READER_SPHERE_V1.json';assert not out.exists();start=time.perf_counter();toy=toy_check()
 artifact=torch.load(P/'SELECTIVE_SCALAR_READOUT_V1.pt',weights_only=True);s={k:v.cuda().double() for k,v in artifact['program'].items()};reader=torch.cat([s['A'],s['B']]);frame=torch.linalg.qr(reader.T).Q
 teacher=load_teacher('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',artifact['teacher_scale']);u=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True)['output_directions'][:,1].cuda().double();factors=[(u@teacher[0].double()).float(),*teacher[1:]];del teacher
 cal=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'][::8];held=next(p for p in torch.load(P/'FROZEN_FRESH_VALIDATION_V1.pt',weights_only=True)['panels'] if int(p['context'])==256)['rows'][::32];panels={'calibration':cal,'diagnostic256':held};records=[];checks=[];precision=[];fdchecks=[];gradients={};rng=torch.Generator(device='cuda').manual_seed(PLAN['seed'])
 for panel,rows in panels.items():
  assert len(rows)==256;x=rows.cuda().double();values=[];grads=[]
  for chunk in x.split(64):
   v,g=scalar_value_gradient(factors,chunk.float());values.append(v.double());grads.append(g.double())
  value=torch.cat(values);gradient=torch.cat(grads);ref=scalar_value_gradient([t.double() for t in factors],x[:8])[1];precision.append(float((gradient[:8]-ref).norm()/ref.norm()));invisible,xnull=invisible_tangent(x,gradient,frame);tangent=gradient-x*((gradient*x).sum(1,keepdim=True)/x.square().sum(1,keepdim=True));fraction=float(invisible.square().sum()/tangent.square().sum());assert fraction<=1+1e-8
  # Fixed ambient missing-gradient subspace, distinct from x-dependent tangent projection.
  gradients[panel]=gradient-(gradient@frame)@frame.T
  random=torch.randn(x.shape,generator=rng,device=x.device,dtype=x.dtype);perturb=[]
  for family in PLAN['families']:
   direction=gradient if family=='gradient' else random
   for angle in PLAN['angles']:
    plus,minus,unit,radius=spherical_pair(x,direction,frame,angle);checks.extend([float(((plus-minus)@reader.T).norm()/(x@reader.T).norm()),float((plus.square().sum(1)-x.square().sum(1)).norm()/x.square().sum(1).norm()),float((minus.square().sum(1)-x.square().sum(1)).norm()/x.square().sum(1).norm()),float((evaluate(s,plus)-evaluate(s,minus)).norm()/evaluate(s,x).norm())]);vp=[];vm=[]
    for a,b in zip(plus.split(64),minus.split(64)):
     vp.append(scalar_value_gradient(factors,a.float())[0].double());vm.append(scalar_value_gradient(factors,b.float())[0].double())
    diff=torch.cat(vp)-torch.cat(vm);linear=2*torch.sin(x.new_tensor(angle))*radius[:,0]*(gradient*unit).sum(1);fd=float((diff-linear).norm()/linear.norm());rms=float(diff.square().mean().sqrt());perturb.append(dict(family=family,angle=angle,pair_difference_rms=rms,pair_difference_over_native_std=rms/float(value.std(unbiased=False)),first_order_relative_error=fd,relative_input_displacement=float((plus-x).norm()/x.norm())))
    if family=='gradient' and angle==.001:fdchecks.append(fd)
  records.append(dict(panel=panel,native_scalar_std=float(value.std(unbiased=False)),invisible_tangent_gradient_energy_fraction=fraction,perturbations=perturb))
 _,sv,vh=torch.linalg.svd(gradients['calibration'],full_matrices=False);coverage=[]
 for rank in [1,4,8,16]:coverage.append(dict(rank=rank,calibration=float((gradients['calibration']@vh[:rank].T).square().sum()/gradients['calibration'].square().sum()),diagnostic256=float((gradients['diagnostic256']@vh[:rank].T).square().sum()/gradients['diagnostic256'].square().sum())))
 pred=dict(pred_a_instrument=toy['gradient_relative_error']<1e-12 and max(checks)<1e-5 and max(precision)<1e-4 and max(fdchecks)<.02,pred_b_missing=all(r['invisible_tangent_gradient_energy_fraction']>.1 for r in records),pred_c_effect=all(next(q for q in r['perturbations'] if q['family']=='gradient' and q['angle']==.05)['pair_difference_over_native_std']>.01 for r in records))
 result=dict(plan=PLAN,records=records,gradient_subspace_coverage=coverage,predictions=pred,invariance_max=max(checks),gradient_fp32_fp64=max(precision),small_angle_fd=max(fdchecks),toy=toy,seconds=time.perf_counter()-start,scope='Artificial norm-preserving states with identical student readers. Gradient-selected directions are adaptive counterexamples, not natural text or certified population error floors. Subspace coverage is a held-panel derivative diagnostic, not a circuit.')
 guard_torch_save(dict(directions=vh[:16].cpu(),singular_values=sv.cpu(),scope=result['scope']),str(P/'READER_SPHERE_DIRECTIONS_V1.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
