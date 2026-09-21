#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_metric pred_c_transfer
"""Exact Gaussian functional readout of fixed32-feature learned quartic bank.
Integrity: crossFP32/64<1e-3,solve<1e-8,export<1e-4,regularizedobjective improves.
WeightedGaussian beats21.4388%secondpanel; strongestbar<=.8*15.0498%andcal<=5%.
Null: betterGaussianfit failsnaturaltransfer. Same656products903168coefficients.
No fullmodel, freshOOD, selectiveintervention orsemanticadoption claim.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,width=32,metrics=2,channel_batch=16,pair_batch=8,span_width=256)));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from gaussian_bank_metric import gram_by_degree
 from projected_quartic_gaussian import functional_cross
 from empirical_quartic_dictionary import evaluate
 start=time.monotonic();out=P/'WIDE_QUARTIC_GAUSSIAN_READOUT_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];ys=[y.cuda().double() for y in data['targets']];xs=[r['rows'].cuda().double() for r in torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']]
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 transform=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['metric_transforms']['second_floor01'].cuda();old={k:v.cuda().double() for k,v in torch.load(P/'WIDE_NATIVE_QUARTIC_32_INHERITED_LONG_V1.pt',weights_only=True).items()};U,V=old['U'],old['V'];baseline_writer=ru.double()@old['C']/scale;records=[]
 for metric,tr in [('isotropic',None),('second_floor01',transform)]:
  arm_start=time.monotonic();t=teacher if tr is None else [*teacher[:-2],teacher[-2]@tr,teacher[-1]@tr];a,b=(U,V) if tr is None else (U@tr.double(),V@tr.double());parts,meta=gram_by_degree(a,b);G=sum(parts.values());G=(G+G.T)/2;s=G.diag().sqrt();K=G/s[:,None]/s[None,:];eigen=torch.linalg.eigvalsh(K);negative=float(eigen.min()/eigen.max());assert negative>-1e-9
  meta32={k:(v.float() if torch.is_tensor(v) else v) for k,v in meta.items()};X,components=functional_cross(t,a.float(),b.float(),meta32,8,16);X=X.double()
  _,refmeta=gram_by_degree(a[:1],b[:1]);reference,_=functional_cross([z.double() for z in t],a[:1],b[:1],refmeta,1,16);cross_error=float((X[:,:1]-reference).norm()/reference.norm())
  rhs=X/s;system=K+1e-6*torch.eye(len(K),device='cuda',dtype=torch.float64);c=torch.linalg.solve(system,rhs.T).T;normal=float((c@system-rhs).norm()/rhs.norm());writer=c/s
  objective=lambda c:float(((c@G)*c).sum()-2*(c*X).sum()+1e-6*(c*s).square().sum())
  ours,base=objective(writer),objective(baseline_writer);improves=ours<=base+1e-6*max(abs(base),1e-30)
  energies={name:float(((baseline_writer@g)*baseline_writer).sum()) for name,g in parts.items()};total=sum(energies.values());fractions={k:v/total for k,v in energies.items()}
  program=dict(U=U,V=V,C=writer);errors=[float((evaluate(program,x)-y).norm()/y.norm()) for x,y in zip(xs,ys)];archive=dict(U=U.cpu().float(),V=V.cpu().float(),C=torch.linalg.solve(ru.double(),writer*scale).cpu().float());physical={k:v.cuda() for k,v in archive.items()};ref=evaluate(program,xs[0][:128]);actual=evaluate(physical,xs[0][:128].float()).double()@ru.double().T/scale;drift=float((actual-ref).norm()/ref.norm());path=P/f'WIDE_QUARTIC_GAUSSIAN_{metric.upper()}_V1.pt';torch.save(archive,path)
  row=dict(metric=metric,errors=errors,gaussian_regularized_objective_without_constant=ours,empirical_writer_gaussian_objective_without_constant=base,objective_improved=improves,empirical_writer_gaussian_energy_fractions=fractions,minimum_relative_gram_eigenvalue=negative,cross_fp32_fp64_error=cross_error,normal_residual=normal,export_replay=drift,stored_coefficients=sum(v.numel() for v in archive.values()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-arm_start);records.append(row);print(json.dumps(row),flush=True)
 weighted=records[1];pred=dict(pred_a_integrity=all(r['cross_fp32_fp64_error']<1e-3 and r['normal_residual']<1e-8 and r['export_replay']<1e-4 and r['objective_improved'] for r in records),pred_b_metric=weighted['errors'][1]<.2143877281,pred_c_transfer=weighted['errors'][1]<=.8*.1504982857 and weighted['errors'][0]<=.05)
 result=dict(predictions=pred,records=records,seconds=time.monotonic()-start,scope='Exact Gaussian functional moments with FP32native contractions and FP64Gram/solve. Data-informed fixed dictionary, two Gaussian laws. Old natural panels are diagnostic, no freshOOD or nativebehavior adoption. Student energy fractions are not teacher-error fractions.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
