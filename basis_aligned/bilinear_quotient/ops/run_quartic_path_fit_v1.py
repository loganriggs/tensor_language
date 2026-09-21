#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_path pred_c_preserve
"""One width32 path-integrated fit, matched300Adamsteps.
Evalpatherror<=.9halftrained; valuesandhalf/fullresponses<=1.1halftrained.
Replay/export<1e-4;656products903168coeff; no nonlinear endpoint/OOD claim.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 sys.path.insert(0,str(P));import torch
 from quartic_finite_response import fit,response_features
 from quartic_path_objective import design,rule,controls
 from empirical_quartic_dictionary import evaluate
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  cc=controls();torch.manual_seed(81);x=torch.randn(23,7,dtype=torch.float64);d=torch.randn_like(x);y=torch.randn(23,4,dtype=torch.float64);r=torch.randn(5,23,4,dtype=torch.float64);initial=(torch.randn(3,2,7,dtype=torch.float64),torch.randn(3,2,7,dtype=torch.float64));info,p=fit(x,y,d,r,initial,steps=2,design_builder=design);assert p['C'].shape==(4,6) and math.isfinite(info['objective']);print(json.dumps(dict(callback_shape_gradient_smoke='pass',steps=300,quadrature_nodes=5,controls=len(cc))));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'QUARTIC_PATH_FIT_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];xs=[r['rows'].cuda().double() for r in panels];ds=[x.roll(997,0)-x for x in xs];nodes,weights=rule(xs[0])
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight')/scale;L,R=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A,B=w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')
  def teacher(x):
   x=x.float();h=((x@A.T)*(x@B.T))@D.T;return (((h@L.T)*(h@R.T))@C.T).double()
  ys=[teacher(x) for x in xs];rs=[torch.stack([teacher(x+a*d)-y for a in nodes]) for x,d,y in zip(xs,ds,ys)];point=[torch.stack([teacher(x+a*d)-y for a in [.5,1.]]) for x,d,y in zip(xs,ds,ys)];replay=max(float((y-old.cuda().double()).norm()/old.cuda().double().norm()) for y,old in zip(ys,data['targets']))
 initial0=torch.load(P/'QUARTIC_JOINT_READOUT_LAMBDA_1_V1.pt',weights_only=True);initial=(initial0['U'].cuda().double(),initial0['V'].cuda().double())
 info,program=fit(xs[0],ys[0],ds[0],rs[0],initial,steps=300,rate=.03*math.sqrt(4/1152),design_builder=design)
 with torch.no_grad():
  def assess(p):
   result=[]
   for x,d,y,r,rr in zip(xs,ds,ys,rs,point):
    pred=torch.stack([response_features(x,a*d,p['U'],p['V'])@p['C'].T for a in nodes]);error=((pred-r).square().sum((1,2))*weights).sum();energy=(r.square().sum((1,2))*weights).sum();point_errors=[float((response_features(x,a*d,p['U'],p['V'])@p['C'].T-target).norm()/target.norm()) for a,target in zip([.5,1.],rr)];result.append(dict(value_error=float((evaluate(p,x)-y).norm()/y.norm()),integrated_response_error=float((error/energy).sqrt()),point_response_errors=point_errors))
   return result
  results={'path':assess(program)}
  for name,weight in [('half',1),('value',0)]:
   old={k:v.cuda().double() for k,v in torch.load(P/f'QUARTIC_RESPONSE_FEATURES_32_WEIGHT_{weight}_V1.pt',weights_only=True).items()};old['C']=ru.double()@old['C']/scale;results[name]=assess(old)
  archive={k:v.cpu().float() for k,v in program.items()};archive['C']=torch.linalg.solve(ru.double(),program['C']*scale).cpu().float();physical={k:v.cuda() for k,v in archive.items()};actual=evaluate(physical,xs[0][:128].float()).double()@ru.double().T/scale;expected=evaluate(program,xs[0][:128]);drift=float((actual-expected).norm()/expected.norm());path=P/'QUARTIC_PATH_FIT_V1.pt';torch.save(archive,path)
 a,b=results['path'][1],results['half'][1];pred=dict(pred_a_integrity=replay<1e-4 and drift<1e-4 and math.isfinite(info['objective']),pred_b_path=a['integrated_response_error']<=.9*b['integrated_response_error'],pred_c_preserve=a['value_error']<=1.1*b['value_error'] and all(x<=1.1*y for x,y in zip(a['point_response_errors'],b['point_response_errors'])))
 result=dict(predictions=pred,results=results,training=info,target_replay=replay,export_replay=drift,program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-start,scope='One matched inherited feature fit with exact polynomial path-integrated metric;openedpanel0fit/panel1evaluation,internaldonors. No native endpoint or semanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
