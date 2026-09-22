"""Exact Taylor orders along opened same-token state changes; no fitting of models."""
import json,math,time
from pathlib import Path
import torch
from quartic_cp import directional
from audit_root_matched_reader import CK
from audit_root_feature_conditions import root_features
from paired_root_compiler import cast
from sparse_quartic_bank import features
P=Path(__file__).resolve().parent;SCALE=19054614563.464127

def expand(fn,x,d):
 nodes=torch.tensor([-1.,-.5,0.,.5,1.],dtype=x.dtype);vand=torch.stack([nodes**i for i in range(5)],1)
 values=torch.stack([fn(x+t*d) for t in nodes]);coefs=torch.linalg.solve(vand,values);checks=[]
 for t in [.25,1.25]:
  y=fn(x+t*d);pred=sum(coefs[k]*t**k for k in range(5));checks.append(float((pred-y).norm()/y.norm().clamp_min(1e-30)))
 assert max(checks)<1e-10
 return coefs,checks

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();torch.manual_seed(15000)
 # Algebraic control: known quartic in t, including sign cancellation.
 x=torch.randn(9,3,dtype=torch.float64);d=torch.randn_like(x);fn=lambda z:(z[:,0]+z[:,1])**4-z[:,0]**4
 coefs,checks=expand(fn,x,d)
 expected=torch.stack([math.comb(4,k)*((x[:,0]+x[:,1])**(4-k)*(d[:,0]+d[:,1])**k-x[:,0]**(4-k)*d[:,0]**k) for k in range(5)])
 control=float((coefs-expected).norm()/expected.norm());assert control<1e-12
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].double()[:,1];vocab=state['lm_head.weight'].double();uw=vocab@writer;reader=vocab.T@uw/uw.square().sum();del vocab,uw
 w=lambda l,n:state[f'transformer.h.{l}.mlp.{n}.weight'].double()
 T=[reader[None,:]@w(17,'Down')/SCALE,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left'),w(16,'Right')]
 def native(z):
  C,l,r,D,A,B=T;h=((z@A.T)*(z@B.T))@D.T;return (((h@l.T)*(h@r.T))@C.T)[:,0]
 cache=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1];matched=json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1];rec,don=torch.tensor(matched['pairs_flat']).T;x=cache['rows'][rec].double();d=cache['rows'][don].double()-x
 truth,tcheck=expand(native,x,d);polar=torch.stack([math.comb(4,k)*directional(*T,[x]*(4-k)+[d]*k)[:,0] for k in range(5)]);replay=float((polar-truth).norm()/truth.norm());assert replay<1e-10
 norm_linear=2*(x*d).sum(1)/x.square().sum(1);norm_quadratic=d.square().sum(1)/x.square().sum(1);norm_end=(1+norm_linear+norm_quadratic).square();normalized_native_delta=truth.sum(0)/norm_end-truth[0];normalized_native_first=truth[1]-2*norm_linear*truth[0]
 eps=1e-5
 xp=x+eps*d;xm=x-eps*d;radius=x.norm(dim=1,keepdim=True)
 normalized_fd=(native(xp*radius/xp.norm(dim=1,keepdim=True))-native(xm*radius/xm.norm(dim=1,keepdim=True)))/(2*eps)
 normalized_fd_error=float((normalized_fd-normalized_native_first).norm()/normalized_native_first.norm());assert normalized_fd_error<1e-7
 delta=truth[1:].sum(0);cached=torch.tensor([a['reference'] for a in matched['pair_rows']],dtype=torch.float64)/SCALE;native_replay=float((delta-cached).norm()/cached.norm());assert native_replay<1e-10
 paths=[('original384','EXPANDED_ROOT_EMPIRICAL_V1.pt','graph')]+[(f'CP512_{s}',f'MIXED_CP_FEATURES_SEED{s}_V1.pt','cp') for s in [1001,1002]]+[(f'CP256_{s}',f'CP_PARENT_REFIT_SEED{s}_V1.pt','cp') for s in [1001,1002]]+[(f'learned_shared_{s}',f'SHARED_MIXED_FEATURES_SEED{s}_V1.pt','shared') for s in [1101,1102]]
 rows=[]
 for name,file,kind in paths:
  p=torch.load(P/file,weights_only=True)
  if kind=='graph':p=cast(p,torch.float64);fn=lambda z:root_features(p,z)[:,1]/SCALE
  elif kind=='cp':
   fs=[f.double() for f in p['factors']];C=p['coefficients'].double()[1]/SCALE
   def fn(z):return torch.stack([z@f.T for f in fs]).prod(0)@C
  else:
   U,V=[f.double() for f in p['factors']];C=p['coefficients'].double()[1]/SCALE;fn=lambda z:features(z,U,V,p['pairs'])@C
  coef,checks=expand(fn,x,d);e=coef[1:]-truth[1:];total=e.sum(0);eg=e@e.T;tg=truth[1:]@truth[1:].T
  normalized_first=coef[1]-2*norm_linear*coef[0];normalized_delta=coef.sum(0)/norm_end-coef[0]
  ladder=[]
  for alpha in [.01,.05,.1,.25,.5,1.]:
   denominator=(1+norm_linear*alpha+norm_quadratic*alpha**2).square()
   native_effect=sum(truth[k]*alpha**k for k in range(5))/denominator-truth[0]
   candidate_effect=sum(coef[k]*alpha**k for k in range(5))/denominator-coef[0]
   ladder.append(dict(alpha=alpha,error=float((candidate_effect-native_effect).norm()/native_effect.norm()),native_effect_norm=float(native_effect.norm()),native_linearization_error=float((alpha*normalized_native_first-native_effect).norm()/native_effect.norm())))
  rows.append(dict(name=name,normalized_chord_ladder=ladder,normalized_first_derivative_error=float((normalized_first-normalized_native_first).norm()/normalized_native_first.norm()),normalized_first_order_response_error=float((normalized_first-normalized_native_delta).norm()/normalized_native_delta.norm()),normalized_endpoint_error=float((normalized_delta-normalized_native_delta).norm()/normalized_native_delta.norm()),response_error=float(total.norm()/delta.norm()),order_relative_errors=[float(e[k].norm()/truth[k+1].norm()) for k in range(4)],error_order_norm_over_net_native=[float(e[k].norm()/delta.norm()) for k in range(4)],signed_error_contributions=[float((e[k]*total).sum()/total.square().sum()) for k in range(4)],error_gram_over_net_error=(eg/total.square().sum()).tolist(),error_cancellation_ratio=float(eg.trace()/total.square().sum()),first_order_only_response_error=float((coef[1]-delta).norm()/delta.norm()),interpolation_replay=max(checks)))
 tg=truth[1:]@truth[1:].T
 result=dict(rows=rows,normalized_derivative_fd_error=normalized_fd_error,normalized_native_first_order_response_error=float((normalized_native_first-normalized_native_delta).norm()/normalized_native_delta.norm()),normalization_endpoint_difference=float((normalized_native_delta-delta).norm()/delta.norm()),control_error=control,polarization_replay=replay,native_reference_replay=native_replay,native_order_norms_over_total=[float(t.norm()/delta.norm()) for t in truth[1:]],native_order_gram_over_total=(tg/delta.square().sum()).tolist(),native_first_order_response_error=float((truth[1]-delta).norm()/delta.norm()),step_relative_to_input_median=float((d.norm(dim=1)/x.norm(dim=1)).median()),pairs=len(rec),unique_unordered_pairs=len({tuple(sorted([int(a),int(b)])) for a,b in zip(rec,don)}),seconds=time.monotonic()-start,scope='Root1 numerator, same30directed/20unordered opened newline-token pairs. Straightline between normalized endpoints passes through nonnormalized states; exact polynomial identity, not full-model or normalized-path intervention. Taylor coefficients depend on recipient anchor. Signed order contributions can be negative; not independent error percentages. No new fitting.')
 (P/'QUARTIC_RESPONSE_ORDERS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
