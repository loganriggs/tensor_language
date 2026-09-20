#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_law pred_b_order pred_c_precision
"""Frozen raw/normalized artificial-probe comparison; NORMALIZED_PROBE_PLAN_V1.
pred_a_law: normalized errors closer to text errors for both candidates, both seeds.
pred_b_order: quartic beats quadratic on both normalized seeds.
pred_c_precision: finite outputs and FP32/64 relative disagreement <1e-3.
Null: input radius constraint does not explain Gaussian/text discrepancy.
Price: quadratic34560 coefficients4products;quartic47312 coefficients26products.
No fitting, no text output targets. Calibration input moments only.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(seeds=[2033,2034],rows=2048,laws=['raw','normalized'],native_forwards=0,text_errors=dict(quadratic=.21995083789769065,quartic=.1837768810))
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_NORMALIZED_PROBE_V1.json';assert not out.exists();start=time.perf_counter()
 qs=torch.load(P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt',weights_only=True);hs=torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True);assert abs(float(qs['teacher_scale'])/float(hs['teacher_scale'])-1)<1e-6
 q={k:t.cuda() for k,t in qs['programs']['centered'].items()};h={k:t.cuda() for k,t in hs['programs'][8].items()};panel=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0];mu=panel['mean'].cuda().double();M=panel['covariance'].cuda().double();L=torch.linalg.cholesky(M)
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/qs['teacher_scale'],w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 del state,ru
 def evaluate(x,dtype):
  x=x.to(dtype);C,l,r,D,A,B=[t.to(dtype) for t in teacher];z=((x@A.T)*(x@B.T))@D.T;y=((z@l.T)*(z@r.T))@C.T;s={k:v.to(dtype) for k,v in q.items()};delta=x-s['mu'];yp=s['constant']+(delta@s['linear_reader'].T)@s['linear_writer'].T+((delta@s['quadratic_left'].T)*(delta@s['quadratic_right'].T))@s['quadratic_writer'].T;t={k:v.to(dtype) for k,v in h.items()};bank=((x@t['U'].flatten(0,1).T)*(x@t['V'].flatten(0,1).T)).reshape(len(x),4,4).sum(2);i,j=torch.triu_indices(4,4,device='cuda');yh=((bank[:,i]*bank[:,j])@t['Z'].T)@t['W'].T+t['constant'];return dict(teacher=y,quadratic=yp,quartic=yh)
 rows=[]
 with torch.no_grad():
  for seed in PLAN['seeds']:
   gen=torch.Generator(device='cuda').manual_seed(seed);raw=mu+torch.randn(PLAN['rows'],len(mu),generator=gen,device='cuda',dtype=torch.float64)@L.T
   for law in PLAN['laws']:
    x=raw if law=='raw' else raw/raw.norm(dim=1,keepdim=True)*len(mu)**.5;vals=evaluate(x,torch.float32);ref=evaluate(x[:32],torch.float64);precision={k:float((v[:32].double()-ref[k]).norm()/ref[k].norm()) for k,v in vals.items()};y=vals['teacher'].double();metrics={}
    for name in ['quadratic','quartic']:
     p=vals[name].double();res=p-y;metrics[name]=dict(error=float(res.norm()/y.norm()),variation=float(((p-p.mean(0))-(y-y.mean(0))).norm()/(y-y.mean(0)).norm()),mean_error_energy=float(len(x)*res.mean(0).square().sum()/y.square().sum()))
    empirical_mu=x.mean(0);center=x-empirical_mu;cov=center.T@center/len(x);row=dict(seed=seed,law=law,metrics=metrics,precision=precision,finite=all(bool(torch.isfinite(v).all()) for v in vals.values()),mean_shift=float((empirical_mu-mu).norm()/mu.norm()),covariance_shift=float((cov-M).norm()/M.norm()),relative_squared_radius_std=float(x.square().sum(1).std()/x.square().sum(1).mean()));rows.append(row);print(json.dumps(row),flush=True)
 pred=dict(pred_a_law=all(abs(next(r for r in rows if r['seed']==seed and r['law']=='normalized')['metrics'][n]['error']-PLAN['text_errors'][n])<abs(next(r for r in rows if r['seed']==seed and r['law']=='raw')['metrics'][n]['error']-PLAN['text_errors'][n]) for seed in PLAN['seeds'] for n in ['quadratic','quartic']),pred_b_order=all(r['metrics']['quartic']['error']<r['metrics']['quadratic']['error'] for r in rows if r['law']=='normalized'),pred_c_precision=all(r['finite'] and max(r['precision'].values())<1e-3 for r in rows));out.write_text(json.dumps(dict(plan=PLAN,records=rows,predictions=pred,seconds=time.perf_counter()-start,scope='Paired artificial probe diagnostic; normalization changes input law and moments. Same frozen candidates, no fitting.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
