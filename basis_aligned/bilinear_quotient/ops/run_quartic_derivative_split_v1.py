#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_inside pred_c_gaussian
"""Split quartic derivative mismatch into within/outside learnedreader span.
Native FP32/64<1e-3,partition<1e-8,studentoutside<1e-8teacher-relative.
Empiricalcenteredwithin>50%bothpanels; Gaussianreadoutbeats empiricalcenteredboth.
Fixedfourwriters,656products903168coefficients; no adoptionclaim.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,panels=2,anchors=16,writers=4,metrics=3)));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from quartic_reader_span import jacobian,reader_basis
 from quartic_bank_derivative import feature_jacobian,split_error
 start=time.monotonic();out=P/'QUARTIC_DERIVATIVE_SPLIT_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 files={'empirical':'WIDE_NATIVE_QUARTIC_32_INHERITED_LONG_V1.pt','coefficient_isotropic':'WIDE_QUARTIC_WEIGHT_LEARNED_ISOTROPIC_V1.pt','coefficient_second':'WIDE_QUARTIC_WEIGHT_LEARNED_SECOND_FLOOR01_V1.pt','gaussian_second':'WIDE_QUARTIC_GAUSSIAN_SECOND_FLOOR01_V1.pt'}
 archives={n:torch.load(P/f,weights_only=True) for n,f in files.items()};old=archives['empirical'];U,V=old['U'].cuda().double(),old['V'].cuda().double()
 assert all(torch.equal(a['U'],old['U']) and torch.equal(a['V'],old['V']) for a in archives.values())
 writers={n:ru.double()@a['C'].cuda().double()/scale for n,a in archives.items()}
 saved=torch.load(P/'NATIVE_WEIGHTED_BANK_V1.pt',weights_only=True)['metric_transforms'];metrics={'isotropic':torch.eye(1152,device='cuda',dtype=torch.float64),'centered':saved['centered_floor01'].cuda().double(),'second':saved['second_floor01'].cuda().double()};bases={n:reader_basis(U@L,V@L) for n,L in metrics.items()};records=[];precision=None
 for panel_id,panel in enumerate(panels):
  indices=torch.linspace(0,len(panel['rows'])-1,16).long();xs=panel['rows'][indices].cuda().float();sums={(name,metric):dict(total=0.,within=0.,outside=0.,teacher=0.,student_outside=0.) for name in writers for metric in metrics}
  for x in xs:
   jt=jacobian(teacher,x).double();jp=feature_jacobian(U,V,x.double());js={n:w@jp for n,w in writers.items()}
   if precision is None:
    ref=jacobian([t.double() for t in teacher],x.double());precision=float((jt-ref).norm()/ref.norm())
   for metric,L in metrics.items():
    target=jt@L
    for name,j in js.items():
     terms=split_error(target,j@L,bases[metric])
     for key,value in terms.items():sums[(name,metric)][key]+=float(value)
  for (name,metric),s in sums.items():
   records.append(dict(panel=panel_id,writer=name,metric=metric,error=(s['total']/s['teacher'])**.5,within_error=(s['within']/s['teacher'])**.5,outside_error=(s['outside']/s['teacher'])**.5,within_squared_error_share=s['within']/s['total'],closure=abs(s['total']-s['within']-s['outside'])/s['total'],student_outside_fraction=(s['student_outside']/s['teacher'])**.5))
 empirical=[r for r in records if r['writer']=='empirical' and r['metric']=='centered'];gaussian=[r for r in records if r['writer']=='gaussian_second' and r['metric']=='centered'];pred=dict(pred_a_integrity=precision<1e-3 and all(r['closure']<1e-8 and r['student_outside_fraction']<1e-8 for r in records),pred_b_inside=all(r['within_squared_error_share']>.5 for r in empirical),pred_c_gaussian=all(a['error']<b['error'] for a,b in zip(gaussian,empirical)))
 result=dict(predictions=pred,records=records,native_fp32_fp64_error=precision,seconds=time.monotonic()-start,scope='Orthogonal decomposition of local polynomial derivative mismatch at32openedanchors, fourfrozenwriters. Within-span error notproof of realizable repair; no fullmodel orsemanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
