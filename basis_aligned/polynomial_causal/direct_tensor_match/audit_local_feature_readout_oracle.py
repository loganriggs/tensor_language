"""Opened-label best-case readouts, separately for values and finite differences."""
import json,time,hashlib
from pathlib import Path
import torch
from audit_conditional_residual_accounting import P,SCALE,load

def solve(a,y):
 scale=a.square().sum(0).sqrt();assert (scale>0).all();a=a/scale
 u,s,v=torch.linalg.svd(a,full_matrices=False);rank=int((s>s[0]*1e-12).sum());basis=u[:,:rank]
 fit=basis@(basis.T@y);res=y-fit
 orth=float((a.T@res).norm()/(a.norm()*y.norm()).clamp_min(1e-30));assert orth<1e-10
 if rank==a.shape[1]:
  q=torch.linalg.qr(a,mode='reduced').Q;other=y-q@(q.T@y)
  agreement=float((other.square().sum(0)-res.square().sum(0)).abs().max()/y.square().sum(0).max().clamp_min(1e-30));assert agreement<1e-8
 else:agreement=None
 return res,dict(rank=rank,columns=a.shape[1],condition=float(s[0]/s[-1]),orthogonality=orth,qr_energy_agreement=agreement)

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double();y=data['target'].double()/SCALE
 rec,don=torch.tensor(json.loads((P/'ALL_FEATURE_MATCHED_RESPONSES_V1.json').read_text())['pairs']).T
 parent,ph=load('MIXED_CP_FEATURES_SEED1001_V1.pt')
 pp=torch.cat([torch.stack([z@f.T for f in parent['factors']]).prod(0)@parent['coefficients'].T/SCALE for z in x.split(1024)])
 residual=y[:,4:]-pp[:,4:];targets=y[:,4:];rows=[]
 for receipt,stem in [('LOCAL_QUARTIC_RESIDUAL_NATIVE_V1.json','LOCAL_QUARTIC_RESIDUAL_'),('NATIVE_LOCAL_LBFGS_V1.json','NATIVE_LOCAL_')]:
  run=json.loads((P/receipt).read_text())
  for row in run['rows']:
   name=stem+f"{row['optimizer'].upper()}_SEED{row['seed']}_V1.pt";a,h=load(name);assert h==row['sha256'] and a['parent_sha256']==ph
   phi=torch.cat([torch.stack([z@f.T for f in a['factors']]).prod(0) for z in x.split(1024)])
   modes={}
   for mode,design,wanted,ref in [('value',phi,residual,targets),('response',phi[don]-phi[rec],residual[don]-residual[rec],targets[don]-targets[rec])]:
    errors=[];checks=[]
    for g in range(12):
     err,check=solve(design[:,g*8:(g+1)*8],wanted[:,g:g+1]);errors.append(err);checks.append(check)
    local=torch.cat(errors,dim=1);shared,check=solve(design,wanted)
    def score(e):
     per=(e.square().sum(0)/ref.square().sum(0)).sqrt();return dict(per_output=per.tolist(),small_rms=float(per.square().mean().sqrt()))
    modes[mode]=dict(local=score(local),shared=score(shared),local_checks=checks,shared_check=check)
    assert (shared.square().sum(0)<=local.square().sum(0)+1e-9*wanted.square().sum(0)).all()
   result=dict(optimizer=row['optimizer'],seed=row['seed'],sha256=h,modes=modes);rows.append(result)
   print(row['optimizer'],row['seed'],{k:(v['local']['small_rms'],v['shared']['small_rms']) for k,v in modes.items()},flush=True)
 prediction=all(v['shared']['small_rms']<=.85*v['local']['small_rms'] for row in rows if row['optimizer']=='adam' for v in row['modes'].values())
 (P/'LOCAL_FEATURE_READOUT_ORACLE_V1.json').write_text(json.dumps(dict(rows=rows,prediction=prediction,seconds=time.monotonic()-start,scope='Opened-label oracle, fixed parent/new atom directions; separate value/response optima, not one predictor. No fitting/export for deployment or OOD/semantic claim.'),indent=2)+'\n')
if __name__=='__main__':main()
