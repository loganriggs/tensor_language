"""Output-space diagnosis; oracle amplitudes are never candidate predictions."""
import hashlib,json,time
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
SCALE=19054614563.464127

def capture(a,b):
 projected=(a@b)@b.T
 c=float(projected.square().sum()/a.square().sum())
 remainder=float((a-projected).square().sum()/a.square().sum())
 assert abs(c+remainder-1)<1e-11
 return c

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 torch.manual_seed(43001);q=torch.linalg.qr(torch.randn(12,2,dtype=torch.float64)).Q
 control=torch.randn(50,2,dtype=torch.float64)@q.T
 assert abs(capture(control,q)-1)<1e-12
 cache=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
 extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)
 labels=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)
 ids=torch.load(P.parents[1]/'bilinear_quotient/.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:96,:65]
 digest=lambda x:hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
 assert digest(ids[:,:64])==labels['token_sha256']['calibration']
 assert digest(ids[:32])==cache[0]['token_sha256'] and digest(ids[32:])==extra['token_sha256']
 calx=torch.cat([cache[0]['rows'],extra['rows']]).double()
 caly=labels['panels'][0]['target'].double()/SCALE
 fresh=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True)
 x=fresh['rows'].double();y=fresh['target'].double()/SCALE
 rec,don=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T
 scale=caly[:,4:].square().mean(0).sqrt();assert (scale>0).all()
 rows=[]
 for seed in [1001,1002]:
  path=P/f'MIXED_CP_FEATURES_SEED{seed}_V1.pt';a=torch.load(path,weights_only=True)
  factors=[f.double() for f in a['factors']];C=a['coefficients'].double()/SCALE
  def pred(z):return torch.cat([torch.stack([xx@f.T for f in factors]).prod(0)@C.T for xx in z.split(1024)])
  cal=(pred(calx)-caly)[:,4:]/scale
  ev=(pred(x)-y)[:,4:]/scale;response=ev[don]-ev[rec]
  assert torch.isfinite(cal).all() and torch.isfinite(ev).all()
  basis=torch.linalg.eigh(cal.T@cal).eigenvectors.flip(1)
  assert float((basis.T@basis-torch.eye(12)).norm())<1e-11
  spectra=[]
  for rank in [1,2,4,8,12]:
   b=basis[:,:rank]
   spectra.append(dict(rank=rank,calibration_capture=capture(cal,b),value_capture=capture(ev,b),response_capture=capture(response,b)))
  ceilings={}
  for name,z in [('value',ev),('response',response)]:
   eigen=torch.linalg.eigvalsh(z.T@z).flip(0).clamp_min(0)
   ceilings[name]=float(eigen[:4].sum()/eigen.sum())
  assert all(abs(spectra[-1][key]-1)<1e-11 for key in ['calibration_capture','value_capture','response_capture'])
  row=dict(seed=seed,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),subspaces=spectra,evaluation_specific_rank4_oracle_capture=ceilings)
  rows.append(row);print(json.dumps(row),flush=True)
 result=dict(rows=rows,calibration_target_rms=scale.tolist(),prediction=all(s['value_capture']>=.9 and s['response_capture']>=.9 for row in rows for s in row['subspaces'] if s['rank']==4),seconds=time.monotonic()-start,controls=dict(planted_rank2=True,token_alignment=True,orthogonality=True,energy_accounting=True,full_rank_closure=True),scope='Calibration output basis, opened16384 states/2494 responses. Oracle residual amplitudes, no predictor, no export, no causal/OOD claim.')
 (P/'BALANCED_RESIDUAL_WRITERS_V1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
