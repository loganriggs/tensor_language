#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_price pred_c_fidelity
"""Fixed expanded parent, rank16 outputsharing under coefficient/empirical Grams.
PairedFP32replay<1e-4,certificate<1e-8;384products/330240coeff target.
Primaryempiricalnativevalues/responses<=1.1parentbotholdpanels. No adoption.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def move(v,device):
 import torch
 if torch.is_tensor(v):return v.to(device)
 if isinstance(v,dict):return {k:move(t,device) for k,t in v.items()}
 if isinstance(v,list):return [move(t,device) for t in v]
 return v

def main():
 sys.path.insert(0,str(P));import torch
 from empirical_quartic_dictionary import features
 from quartic_finite_response import response_features
 from shared_quadratic_bank import bank_gram
 from weighted_output_projection import project
 from shared_root_block_compiler import compile_basis
 from paired_root_compiler import compile_program,cast,evaluate,price
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  torch.manual_seed(64);u=torch.randn(32,4,7,dtype=torch.float64);v=torch.randn_like(u);c=torch.randn(19,528,dtype=u.dtype);basis=torch.linalg.qr(torch.randn(19,16,dtype=u.dtype)).Q;source=compile_basis(u,v,c,torch.eye(19,dtype=u.dtype),basis);p,_=compile_program(source,allow_rotations=True);x=torch.randn(23,7,dtype=u.dtype);expected=features(x,u,v)@c.T@basis@basis.T;error=float((evaluate(p,x)-expected).norm()/expected.norm());assert error<1e-8;assert evaluate(cast(p,torch.float32),x.float()).shape==(23,19);print(json.dumps(dict(native_shape_replay=error,price=price(p),rank=16,metrics=['coefficient','empirical'])));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'EXPANDED_ROOT_COMPRESSION_V1.json';assert not out.exists()
 file=P/'QUARTIC_EXPANDED_STATE_V1.pt';parent={k:v.cuda().double() for k,v in torch.load(file,weights_only=True).items()};U,V=parent['U'],parent['V'];old=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];xs=[r['rows'].cuda().double() for r in old];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)['rows'].cuda().double();cal=torch.cat([xs[0],extra]);scale=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['teacher_scale']
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));ru=ru.double();A=ru@parent['C']/scale;C=(ru.float()@w('transformer.h.17.mlp.Down.weight'))/scale;L,R=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];left,right=w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')
 def teacher(x):
  x=x.float();h=((x@left.T)*(x@right.T))@D.T;return (((h@L.T)*(h@R.T))@C.T).double()
 phi=features(cal,U,V);pv=phi@A.T;G=phi.T@phi/pv.square().sum();response_gram=torch.zeros_like(G);response_energy=G.new_zeros(())
 for shift in [997,313,1427,179]:
  delta=.5*(cal.reshape(3,2048,1152).roll(shift,1).reshape_as(cal)-cal);dp=response_features(cal,delta,U,V);response_gram+=dp.T@dp;response_energy+=(dp@A.T).square().sum()
 G+=response_gram/response_energy;grams=dict(coefficient=bank_gram(U,V),empirical=G)
 panel_phi=[features(x,U,V) for x in xs];ys=[teacher(x) for x in xs];panel_dp=[[response_features(x,.5*(x.roll(s,0)-x),U,V) for s in [997,613,1379]] for x in xs];targets=[[teacher(x+.5*(x.roll(s,0)-x))-y for s in [997,613,1379]] for x,y in zip(xs,ys)]
 def assess(writer):
  return [dict(value_error=float((p@writer.T-y).norm()/y.norm()),response_errors=[float((dp@writer.T-t).norm()/t.norm()) for dp,t in zip(dps,ts)]) for p,y,dps,ts in zip(panel_phi,ys,panel_dp,targets)]
 baseline=assess(A);records=[]
 for name,gram in grams.items():
  basis,compressed,info=project(A,gram,16);source=compile_basis(U,V,parent['C'],ru,basis);p,diagnostics=compile_program(move(source,'cpu'),allow_rotations=True);archive=cast(p,torch.float32);physical=move(archive,'cuda');replays=[]
  for x,phi0 in zip(xs,panel_phi):
   actual=evaluate(physical,x.float()).double()@ru.T/scale;expected=phi0@compressed.T;replays.append(float((actual-expected).norm()/expected.norm()))
  metric_errors={}
  for metric,g in grams.items():
   residual=A-compressed;num=((residual@g)*residual).sum();den=((A@g)*A).sum();assert num>-1e-10*den;metric_errors[metric]=float((num.clamp_min(0)/den).sqrt())
  path=P/f'EXPANDED_ROOT_{name.upper()}_V1.pt';torch.save(archive,path);row=dict(metric=name,projection=info,parent_metric_errors=metric_errors,native_errors=assess(compressed),fp32_replay=replays,price=price(archive),pair_diagnostics=diagnostics,program_sha256=hashlib.sha256(path.read_bytes()).hexdigest());records.append(row);print(json.dumps({k:v for k,v in row.items() if k!='pair_diagnostics'}),flush=True)
 primary=next(r for r in records if r['metric']=='empirical');pred=dict(pred_a_integrity=all(max(r['fp32_replay'])<1e-4 and r['projection']['certificate']<1e-8 for r in records),pred_b_price=primary['price']['products']<=384 and primary['price']['stored_coefficients']<=330240,pred_c_fidelity=all(a['value_error']<=1.1*b['value_error'] and all(x<=1.1*y for x,y in zip(a['response_errors'],b['response_errors'])) for a,b in zip(primary['native_errors'],baseline)))
 result=dict(predictions=pred,parent_native_errors=baseline,records=records,parent_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),seconds=time.monotonic()-start,scope='Fixed data-informed parent; weight-onlycoefficient versuscalibrationfunctionalcompression; exactconditionaloutputprojectionandpairedrewrite. Oldpanels diagnostic, not nativeendpoint orsemanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
