#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_original pred_c_compressed
"""Gaussian information floor; CONDITIONAL_READER_PLAN_V1.md.
pred_a_instrument: same-reader/student pair relative disagreement<1e-5, toy<1e-12.
pred_b_original: calibration original-reader normalized floor<.10.
pred_c_compressed: calibration compressed-reader normalized floor<.15.
Null: dictionaries discard necessary inputs. Estimates, not certified bounds.
Price: frozen26/10product dictionaries; no fitted/exported replacement.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(laws=['calibration','isotropic'],dictionaries=['original','compressed'],pairs=2048,batch=64,seeds=list(range(260922,260926)))
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch,numpy as np
 sys.path.insert(0,str(P))
 from conditional_reader_bound import reader_frame,paired_inputs,toy_oracle
 from frozen_program_evaluation import quartic,native,load_teacher
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'CONDITIONAL_READER_V1.json';assert not out.exists();start=time.perf_counter();toy=toy_oracle()
 source=torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True);scale=source['teacher_scale']
 students={'original':source['programs'][8],'compressed':torch.load(P/'FUSED_ROOT_PROGRAM_V1.pt',weights_only=True)['programs'][4]};students={k:{n:t.cuda().double() for n,t in s.items()} for k,s in students.items()}
 readers={'original':torch.cat([students['original'][k].flatten(0,1) for k in ['U','V']]),'compressed':torch.cat([students['compressed'][k] for k in ['A','B']])}
 panel=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];mu=panel['mean'].cuda().double();cov=panel['covariance'].cuda().double();ev,evec=torch.linalg.eigh((cov+cov.T)/2);transform=evec*ev.clamp_min(0).sqrt()
 view=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True);U=view['output_directions'].cuda().double();offset=view['output_mean'].cuda().double()
 teacher=load_teacher('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',scale)
 records=[];checks=[];precision=[];arm=0
 for law in PLAN['laws']:
  m=mu if law=='calibration' else torch.zeros_like(mu);L=transform if law=='calibration' else torch.eye(len(mu),device='cuda',dtype=torch.float64)
  for name in PLAN['dictionaries']:
   reader=readers[name];Q,sv=reader_frame(reader,L);rng=torch.Generator(device='cuda').manual_seed(PLAN['seeds'][arm]);arm+=1;blocks=[];sumy=torch.zeros_like(mu);sums=torch.zeros(4,device='cuda',dtype=torch.float64)
   for i in range(0,PLAN['pairs'],PLAN['batch']):
    a,b=paired_inputs(m,L,Q,PLAN['batch'],rng);sa,sb=quartic(students[name],a),quartic(students[name],b)
    checks.extend([float(((a-b)@reader.T).norm()/(a@reader.T).norm()),float((sa-sb).norm()/sa.norm())])
    ya,yb=native(teacher,a.float()).double(),native(teacher,b.float()).double()
    if i==0:
     ref=native([t.double() for t in teacher],a[:4]);precision.append(float((ref-ya[:4]).norm()/ref.norm()))
    qa,qb=(ya-offset)@U,(yb-offset)@U;pa,pb=(sa-offset)@U,(sb-offset)@U
    numerator=torch.cat([((ya-yb).square().sum(1)/2).mean()[None],((qa-qb).square()/2).mean(0)])
    energy=torch.cat([((ya.square().sum(1)+yb.square().sum(1))/2).mean()[None],((qa.square()+qb.square())/2).mean(0)])
    error=torch.cat([(((sa-ya).square().sum(1)+(sb-yb).square().sum(1))/2).mean()[None],(((pa-qa).square()+(pb-qb).square())/2).mean(0)])
    blocks.append(dict(irreducible_mse=numerator.cpu().tolist(),target_second_moment=energy.cpu().tolist(),student_mse=error.cpu().tolist()));sumy+=(ya+yb).sum(0);sums+=(qa+qb).sum(0)
   n=np.array([x['irreducible_mse'] for x in blocks]);e=np.array([x['target_second_moment'] for x in blocks]);err=np.array([x['student_mse'] for x in blocks]);mean=sumy/(2*PLAN['pairs']);means=sums/(2*PLAN['pairs']);meanenergy=np.r_[float(mean.square().sum()),means.square().cpu().numpy()];variance=e.mean(0)-meanenergy
   index=np.random.default_rng(260927+arm).integers(0,len(blocks),(5000,len(blocks)));boot=np.sqrt(n[index].mean(1)/e[index].mean(1))
   row=dict(law=law,dictionary=name,rank=Q.shape[1],irreducible_mse=n.mean(0).tolist(),target_second_moment=e.mean(0).tolist(),target_variance=variance.tolist(),normalized_floor=np.sqrt(n.mean(0)/e.mean(0)).tolist(),variation_normalized_floor=np.sqrt(n.mean(0)/variance).tolist(),student_relative_error=np.sqrt(err.mean(0)/e.mean(0)).tolist(),floor_ci95=np.quantile(boot,[.025,.975],axis=0).T.tolist(),blocks=blocks);records.append(row);print({k:v for k,v in row.items() if k!='blocks'},flush=True)
 get=lambda name:next(r for r in records if r['law']=='calibration' and r['dictionary']==name)['normalized_floor'][0]
 pred=dict(pred_a_instrument=max(checks)<1e-5 and abs(toy['paired_difference_half']-3)<1e-12,pred_b_original=get('original')<.1,pred_c_compressed=get('compressed')<.15)
 result=dict(plan=PLAN,metric_order=['full','mode0','mode1','mode2','mode3'],records=records,predictions=pred,coordinate_student_pair_disagreement=max(checks),fp32_teacher_vs_fp64=max(precision),covariance_minimum_eigenvalue=float(ev.min()),toy=toy,seconds=time.perf_counter()-start,scope='Gaussian conditional variance estimates for fixed readers; artificial probes, not certified bounds or actual text floors. Student errors are frozen diagnostics.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
