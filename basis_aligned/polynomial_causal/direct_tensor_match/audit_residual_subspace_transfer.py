"""Diagnostic output subspaces only: no corrective predictor fitted or exported."""
import hashlib,json,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
SCALE=19054614563.464127

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
 cache=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True);labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)
 xs=[torch.cat([cache[0]['rows'],extra['rows']]).double(),cache[1]['rows'].double()];ys=[a['target'].double()/SCALE for a in labels['panels']]
 digest=lambda x:hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
 for k,file,n in [('calibration','fineweb_n96_skip1200.pt',96),('evaluation','fineweb_n192_skip7000.pt',32)]:
  ids=torch.load(P.parents[1]/'bilinear_quotient/.rowcache'/file,weights_only=True)[:n,:65]
  assert digest(ids[:,:64])==labels['token_sha256'][k]
  if k=='calibration':assert digest(ids[:32])==cache[0]['token_sha256'] and digest(ids[32:])==extra['token_sha256']
  else:assert digest(ids)==cache[1]['token_sha256']
 residuals=[];rows=[]
 for seed in [1001,1002]:
  a=torch.load(P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt',weights_only=True);rr=[]
  for x,y in zip(xs,ys):
   phi=torch.ones(len(x),512,dtype=torch.float64)
   for f in a['factors']:phi*=x@f.double().T
   rr.append(phi@(a['coefficients'].double()/SCALE).T-y)
  residuals.append(rr);cal,ev=rr;basis=torch.linalg.eigh(cal.T@cal).eigenvectors.flip(1)
  # Uncentered PCA includes mean error; column-space transfer, not a predictor.
  spectra=[]
  for rank in [1,2,4,8,12,16]:
   b=basis[:,:rank];proj=(ev@b)@b.T
   captured=float(proj.square().sum()/ev.square().sum());remainder=float((ev-proj).square().sum()/ev.square().sum())
   assert abs(captured+remainder-1)<1e-12
   spectra.append(dict(rank=rank,calibration_capture=float((cal@b).square().sum()/cal.square().sum()),evaluation_capture=captured,remaining_relative_error=float((ev-proj).norm()/ys[1].norm()),scope='Evaluation residual projection is an oracle, NOT a deployable correction. Basis learned on calibration only.'))
  mu=cal.mean(0);rows.append(dict(seed=seed,subspaces=spectra,mean_cosine=float(torch.nn.functional.cosine_similarity(mu,ev.mean(0),dim=0)),calibration_mean_norm=float(mu.norm()),evaluation_mean_norm=float(ev.mean(0).norm()),mean_only_diagnostic_error=float((ev-mu).norm()/ys[1].norm()),original_evaluation_error=float(ev.norm()/ys[1].norm())))
 align=[]
 for i,label in enumerate(['calibration','evaluation']):
  a,b=residuals[0][i],residuals[1][i]
  align.append(dict(panel=label,residual_cosine=float((a*b).sum()/(a.norm()*b.norm())),ensemble_error=float(((a+b)/2).norm()/ys[i].norm()),individual_errors=[float(r.norm()/ys[i].norm()) for r in [a,b]],scope='Averaging both programs roughly doubles arithmetic/storage; diagnostic, not a simpler candidate.'))
 result=dict(rows=rows,restart_alignment=align,seconds=time.perf_counter()-start,scope='Previously opened 6144 calibration and2048 evaluation token states. Two existing CP512 candidates. Diagnostic addresses stability and potential shared residual writers, not causal semantics, input attribution, fresh validation, or learned executable correction.')
 (P/'RESIDUAL_SUBSPACE_TRANSFER_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
