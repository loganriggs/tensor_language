"""Frozen CP readouts fitted to exact native Gaussian finite-change objectives."""
import json,time
import torch
from paired_gaussian_cp import gram_degrees,cross_degrees,combine
from noncentral_gaussian_cp import project_shifted,affine_moment
from mixed_gaussian_cp import gram_dynamic
from quartic_cp_profile import profile
from audit_root_matched_reader import CK
from audit_conditional_residual_accounting import P,SCALE,load
from audit_balanced_shared_followup import scores

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double();loc=torch.linalg.solve(S,mu)
 state=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');U=state['lm_head.weight'].double();UW=U@cache['writer'].double();readers=U.T@UW/UW.square().sum(0);del U,UW
 def w(l,n):return state[f'transformer.h.{l}.mlp.{n}.weight'].double()
 teacher=[readers.T@w(17,'Down')/SCALE,w(17,'Left'),w(17,'Right'),w(16,'Down')*state['transformer.h.17.lambdas'][0].item(),w(16,'Left')@S,w(16,'Right')@S]
 projection=project_shifted(teacher,loc,tuple(t.double() for t in cache['projections']['covariance']['zero_projection']))
 oldx=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'].double();oldy=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'].double()/SCALE;oldpairs=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat']).T
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);newpairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T;rows=[];integrity=[]
 for seed in [1001,1002]:
  p,h=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');fs=p['factors'];fw=[a@S for a in fs];bs=[a@mu for a in fs];G=gram_degrees(fw,bs);oldG=gram_dynamic(fw,bs,fw,bs);gerr=float((G.sum(0)-oldG).norm()/oldG.norm());assert gerr<1e-10;del oldG
  parts=[]
  for left in range(0,512,128):parts.append(cross_degrees(teacher,loc,projection,[f[left:left+128] for f in fw],[b[left:left+128] for b in bs]))
  X=torch.cat(parts,2);mean_phi=affine_moment(fw,bs);fits=[]
  for rho in [None,0.,.5,.9,1.]:
   g=combine(G,rho);cross=combine(X,rho);loss,C=profile(g,cross,ridge=1e-6);normal=float((C@(g+1e-6*torch.eye(len(g),dtype=g.dtype))-cross).norm()/cross.norm());assert normal<1e-8;mean_difference=C@mean_phi-projection[0];fits.append((rho,C,normal,mean_difference.tolist(),float(mean_difference.norm()/projection[0].norm())))
  for panel,x,y,pairs in [('original',oldx,oldy,oldpairs),('opened256',data['rows'].double(),data['target'].double()/SCALE,newpairs)]:
   phi=torch.cat([torch.stack([xx@a.T for a in fs]).prod(0) for xx in x.split(1024)])
   for rho,C,normal,mean_diff,mean_error in fits:
    pred=phi@C.T;s=scores(pred,y,*pairs);row=dict(seed=seed,panel=panel,rho=rho,normal_residual=normal,gaussian_mean_difference=mean_diff,gaussian_relative_mean_error=mean_error,text_mean_bias_over_target_rms=float((pred-y).mean(0).norm()/y.square().mean(0).sum().sqrt()),**s);rows.append(row);print(seed,panel,rho,s['value_error'],s['small_response_rms'],flush=True)
   rows.append(dict(seed=seed,panel=panel,rho='original_parent',**scores(phi@(p['coefficients']/SCALE).T,y,*pairs)))
  integrity.append(dict(seed=seed,parent_sha256=h,gram_degree_sum_replay=gerr))
 checks=[]
 for seed in [1001,1002]:
  base=next(r for r in rows if r['seed']==seed and r['panel']=='original' and r['rho'] is None);test=next(r for r in rows if r['seed']==seed and r['panel']=='original' and r['rho']==.5);rr=test['small_response_rms']/base['small_response_rms'];vr=test['value_error']/base['value_error'];checks.append(dict(seed=seed,response_ratio=rr,value_ratio=vr,pred_response=rr<=.85,pred_value_retention=vr<=1.1))
 result=dict(rows=rows,checks=checks,integrity=integrity,seconds=time.monotonic()-start,pred_primary=all(c['pred_response'] and c['pred_value_retention'] for c in checks),scope='Exact correlated Gaussian difference metric; calibration mu/covariance, no actual response labels fitted. Fixed512quartic features; all16output readouts refit. Both text panels opened. Difference-only objectives ignore constant offsets; values/mean drift explicitly tested. No causal adoption or export.')
 (P/'PAIRED_GAUSSIAN_CP_NATIVE_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
