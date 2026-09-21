#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_response pred_c_values
"""Inherited4/32 featurelearn, responseweights0/1,300Adamsteps.
Primary32 responseeval<=.8control; valueeval<=1.1control; export/cache<1e-4.
Same26/656products,48384/903168coeff. Internalpaneldonors, no OOD adoption.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 sys.path.insert(0,str(P))
 import torch
 from quartic_finite_response import fit,response_features
 from empirical_quartic_dictionary import evaluate,controls
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checks=controls();torch.manual_seed(7);x=torch.randn(33,7,dtype=torch.float64);dx=x.roll(13,0)-x;y=torch.randn(33,5,dtype=torch.float64);r=torch.randn_like(y);init=(torch.randn(4,2,7,dtype=torch.float64),torch.randn(4,2,7,dtype=torch.float64));info,p=fit(x,y,dx,r,init,steps=2);assert p['C'].shape==(5,10) and math.isfinite(info['objective']);print(json.dumps(dict(shape_smoke='pass',envelope_gradient_max=max(r['gradient_error'] for r in checks),steps=300,arms=4)));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'QUARTIC_RESPONSE_FEATURES_V1.json';assert not out.exists();toy=json.loads((P/'QUARTIC_RESPONSE_FIT_TOYS_V1.json').read_text());assert toy['warm_all_pass']['adam'] and toy['selected_optimizer']=='adam'
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];xs=[r['rows'].cuda().double() for r in panels];deltas=[.5*(x.roll(997,0)-x) for x in xs]
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight')/scale;L,R=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A,B=w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')
  def teacher(x):
   x=x.float();h=((x@A.T)*(x@B.T))@D.T;return (((h@L.T)*(h@R.T))@C.T).double()
  ys=[teacher(x) for x in xs];rs=[teacher(x+d)-y for x,d,y in zip(xs,deltas,ys)];replay=max(float((y-old.cuda().double()).norm()/old.cuda().double().norm()) for y,old in zip(ys,data['targets']))
 records=[]
 for width,filename in [(4,'WIDE_NATIVE_QUARTIC_4_INHERITED_LONG_V1.pt'),(32,'QUARTIC_JOINT_READOUT_LAMBDA_1_V1.pt')]:
  old=torch.load(P/filename,weights_only=True);initial=(old['U'].cuda().double(),old['V'].cuda().double())
  for weight in [0.,1.]:
   info,p=fit(xs[0],ys[0],deltas[0],rs[0],initial,response_weight=weight,steps=300,rate=.03*math.sqrt(4/1152),optimizer='adam')
   with torch.no_grad():
    values=[float((evaluate(p,x)-y).norm()/y.norm()) for x,y in zip(xs,ys)];responses=[float((response_features(x,d,p['U'],p['V'])@p['C'].T-r).norm()/r.norm()) for x,d,r in zip(xs,deltas,rs)]
    archive={k:t.cpu().float() for k,t in p.items()};archive['C']=torch.linalg.solve(ru.double(),p['C']*scale).cpu().float();physical={k:t.cuda() for k,t in archive.items()};actual=evaluate(physical,xs[0][:128].float()).double()@ru.double().T/scale;expected=evaluate(p,xs[0][:128]);drift=float((actual-expected).norm()/expected.norm());path=P/f'QUARTIC_RESPONSE_FEATURES_{width}_WEIGHT_{int(weight)}_V1.pt';torch.save(archive,path)
   info.update(width=width,response_weight=weight,value_errors=values,response_errors=responses,export_replay=drift,products=4*width+width*(width+1)//2,stored_coefficients=sum(z.numel() for z in archive.values()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest());records.append(info);print(json.dumps(info),flush=True)
 primary=next(r for r in records if r['width']==32 and r['response_weight']==1);control=next(r for r in records if r['width']==32 and r['response_weight']==0)
 pred=dict(pred_a_integrity=replay<1e-4 and all(math.isfinite(r['objective']) and r['export_replay']<1e-4 for r in records),pred_b_response=primary['response_errors'][1]<=.8*control['response_errors'][1],pred_c_values=primary['value_errors'][1]<=1.1*control['value_errors'][1])
 result=dict(predictions=pred,records=records,target_replay=replay,seconds=time.monotonic()-start,scope='Matched inherited feature optimization, value/responsefit onpanel0; donorroll997internaltoeachexistingpanel. Openedevaluationpanel1, no finalOOD/nativeintervention orsemanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
