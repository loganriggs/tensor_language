"""Fixed mixed-CP dictionaries, exact empirical calibration moment readouts."""
import hashlib,json,time
import torch
from audit_conditional_residual_accounting import P,SCALE,load
from audit_balanced_shared_followup import scores
from quartic_cp_profile import profile

def controls():
 torch.manual_seed(34000);rows=[]
 for kind in ['dense','duplicated','scaled','signed','sparse']:
  A=torch.randn(64,8,dtype=torch.float64);Y=torch.randn(64,3,dtype=torch.float64)
  if kind=='duplicated':A[:,1]=A[:,0]
  if kind=='scaled':A[:,0]*=100
  if kind=='signed':Y[:,1]=-Y[:,0]
  if kind=='sparse':A[A.abs()<1]=0
  ridge=.01;_,C=profile(A.T@A/64,Y.T@A/64,ridge=ridge);aa=torch.cat([A/(64**.5),ridge**.5*torch.eye(8,dtype=A.dtype)]);yy=torch.cat([Y/(64**.5),torch.zeros(8,3,dtype=A.dtype)]);ref=torch.linalg.lstsq(aa,yy,driver='gelsd').solution.T;err=float((C-ref).norm()/ref.norm());assert err<1e-10;rows.append(dict(kind=kind,coefficient_replay=err))
 return rows

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();checks=controls();cache=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True);labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True);rowcache=P.parents[1]/'bilinear_quotient/.rowcache';digest=lambda a:hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest()
 for key,file,n in [('calibration','fineweb_n96_skip1200.pt',96),('evaluation','fineweb_n192_skip7000.pt',32)]:
  ids=torch.load(rowcache/file,weights_only=True)[:n,:65];assert digest(ids[:,:64])==labels['token_sha256'][key]
  if key=='calibration':assert digest(ids[:32])==cache[0]['token_sha256'] and digest(ids[32:96])==extra['token_sha256']
  else:assert digest(ids)==cache[1]['token_sha256']
 calx=torch.cat([cache[0]['rows'],extra['rows']]).double();caly=labels['panels'][0]['target'].double()/SCALE;oldpairs=torch.tensor(json.loads((P/'ROOT_MATCHED_READER_V1.json').read_text())['rows'][1]['pairs_flat']).T;fresh=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);pairs=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T;gaussian=json.loads((P/'PAIRED_GAUSSIAN_CP_NATIVE_V1.json').read_text());rows=[]
 geometry=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True);mu=geometry['mean'].double();S=geometry['projections']['covariance']['whitener'].double();center=calx-calx.mean(0);cov=center.T@center/len(calx);mean_replay=float((mu-calx.mean(0)).norm()/mu.norm());covariance_replay=float((S@S.T-cov).norm()/cov.norm());assert max(mean_replay,covariance_replay)<1e-10
 for seed in [1001,1002]:
  p,h=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');fs=p['factors'];phi=lambda x:torch.cat([torch.stack([xx@f.T for f in fs]).prod(0) for xx in x.split(1024)]);a=phi(calx);G=a.T@a/len(a);X=caly.T@a/len(a);_,C=profile(G,X,ridge=1e-6);normal=float((C@(G+1e-6*torch.eye(512,dtype=G.dtype))-X).norm()/X.norm());assert normal<1e-8;cal_res=a@C.T-caly;cal_error=float(cal_res.norm()/caly.norm());cal_features=(cal_res.square().sum(0)/caly.square().sum(0)).sqrt()
  for panel,x,y,pair in [('original',cache[1]['rows'].double(),labels['panels'][1]['target'].double()/SCALE,oldpairs),('opened256',fresh['rows'].double(),fresh['target'].double()/SCALE,pairs)]:
   pred=phi(x)@C.T;s=scores(pred,y,*pair);base=next(r for r in gaussian['rows'] if r['seed']==seed and r['panel']==panel and r['rho'] is None);row=dict(seed=seed,panel=panel,parent_sha256=h,normal_residual=normal,calibration_error=cal_error,calibration_feature_errors=cal_features.tolist(),calibration_small_value_rms=float(cal_features[4:].square().mean().sqrt()),gaussian_baseline=base,small_response_ratio=s['small_response_rms']/base['small_response_rms'],value_ratio=s['value_error']/base['value_error'],**s);rows.append(row);print(seed,panel,s['value_error'],s['small_response_rms'],row['small_response_ratio'],flush=True)
 primary=[r for r in rows if r['panel']=='original'];out=dict(mean_replay=mean_replay,covariance_replay=covariance_replay,controls=checks,rows=rows,pred_response=all(r['small_response_ratio']<=.85 for r in primary),pred_value_retention=all(r['value_ratio']<=1.1 for r in primary),seconds=time.monotonic()-start,scope='Calibration-output regression/currentmixed512banks/all16outputs. Same raw ridge as Gaussian refit. Both evaluation panels opened, no evaluation fitting. Token ordering hashes checked; no export or semantic/causal adoption.')
 (P/'MIXED_CP_CALIBRATION_MOMENTS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
