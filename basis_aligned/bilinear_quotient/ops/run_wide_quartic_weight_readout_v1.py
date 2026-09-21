#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_weighted pred_c_transfer
"""Frozen32-feature learned/random dictionaries; isotropic/second-moment weights.
Cross FP32/64<1e-3,solve<1e-8,export<1e-4. Weighted beatsiso onbotholdsecond
panels; weightedlearned error<=.8*.1504982871 andcal<=.05. Four656product
903168coefficient programs. No semantic/OOD/fullmodeladoption claim.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,width=32,dictionaries=2,metrics=2,pair_batch=8)));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from batched_quartic_cross import cross
 from shared_quadratic_bank import bank_gram,normalize_bank
 from empirical_quartic_dictionary import features,evaluate,readout
 start=time.monotonic();out=P/'WIDE_QUARTIC_WEIGHT_READOUT_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];ys=[y.cuda().double() for y in data['targets']];xs=[r['rows'].cuda().double() for r in torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']]
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 transform=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['metric_transforms']['second_floor01'].cuda();old={k:v.cuda().double() for k,v in torch.load(P/'WIDE_NATIVE_QUARTIC_32_INHERITED_LONG_V1.pt',weights_only=True).items()};torch.manual_seed(3180);u,v=normalize_bank(torch.randn(32,4,1152,device='cuda',dtype=torch.float64)/math.sqrt(1152),torch.randn(32,4,1152,device='cuda',dtype=torch.float64)/math.sqrt(1152));records=[];baselines=[]
 for name,U,V in [('learned',old['U'],old['V']),('random',u,v)]:
  phi=features(xs[0],U,V)
  if name=='learned':baseline_writer=ru.double()@old['C']/scale
  else:
   bc,bs,_=readout(phi,ys[0]);baseline_writer=(bc/bs[:,None]).T
  baseline=dict(U=U,V=V,C=baseline_writer);baselines.append(dict(dictionary=name,errors=[float((evaluate(baseline,x)-y).norm()/y.norm()) for x,y in zip(xs,ys)]))
  for metric,tr in [('isotropic',None),('second_floor01',transform)]:
   arm_start=time.monotonic();t=teacher if tr is None else [*teacher[:-2],teacher[-2]@tr,teacher[-1]@tr];a,b=(U,V) if tr is None else (U@tr.double(),V@tr.double());G=bank_gram(a,b);G=(G+G.T)/2;s=G.diag().clamp_min(1e-30).sqrt();K=G/s[:,None]/s[None,:];eigen=torch.linalg.eigvalsh(K);negative=float(eigen.min()/eigen.max());assert negative>-1e-9
   X=cross(t,a.float(),b.float(),8).double();reference=cross([z.double() for z in t],a[:2],b[:2],1);cross_error=float((X[:,[0,1,32]]-reference).norm()/reference.norm());rhs=X/s;system=K+1e-6*torch.eye(len(K),device='cuda',dtype=torch.float64);c=torch.linalg.solve(system,rhs.T).T;normal=float((c@system-rhs).norm()/rhs.norm());writer=c/s
   objective=lambda c:float(((c@G)*c).sum()-2*(c*X).sum())
   program=dict(U=U,V=V,C=writer);errors=[float((evaluate(program,x)-y).norm()/y.norm()) for x,y in zip(xs,ys)];archive=dict(U=U.cpu().float(),V=V.cpu().float(),C=torch.linalg.solve(ru.double(),writer*scale).cpu().float());physical={k:v.cuda() for k,v in archive.items()};reference=evaluate(program,xs[0][:128]);actual=evaluate(physical,xs[0][:128].float()).double()@ru.double().T/scale;drift=float((actual-reference).norm()/reference.norm());path=P/f'WIDE_QUARTIC_WEIGHT_{name.upper()}_{metric.upper()}_V1.pt';torch.save(archive,path)
   row=dict(dictionary=name,metric=metric,errors=errors,coefficient_objective_without_constant=objective(writer),baseline_coefficient_objective_without_constant=objective(baseline_writer),minimum_relative_gram_eigenvalue=negative,cross_fp32_fp64_error=cross_error,normal_residual=normal,export_replay=drift,stored_coefficients=sum(v.numel() for v in archive.values()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-arm_start);records.append(row);print(json.dumps(row),flush=True)
 weighted=[r for r in records if r['metric']=='second_floor01'];iso=[r for r in records if r['metric']=='isotropic'];learned=next(r for r in weighted if r['dictionary']=='learned')
 pred=dict(pred_a_integrity=all(r['minimum_relative_gram_eigenvalue']>-1e-9 and r['cross_fp32_fp64_error']<1e-3 and r['normal_residual']<1e-8 and r['export_replay']<1e-4 for r in records),pred_b_weighted=all(a['errors'][1]<b['errors'][1] for a,b in zip(weighted,iso)),pred_c_transfer=learned['errors'][1]<=.8*.1504982871 and learned['errors'][0]<=.05)
 result=dict(predictions=pred,records=records,empirical_baselines=baselines,seconds=time.monotonic()-start,scope='Exact algebraic teacher-feature coefficient contractions with FP32native cross and FP64Gram/solve; numerical checks retained. Fixedlearned/data-informed or random dictionaries; isotropic or savedsecondmoment coefficient geometry. Only random+isotropic is fully weight-only. Empirical panel outcomes not fitting criteria. No newOOD/nativebehavior/semanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
