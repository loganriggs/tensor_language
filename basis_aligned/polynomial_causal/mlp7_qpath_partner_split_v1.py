"""Exact fixed-recipient Q/R partner split into B and attention8 H."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
a=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);v=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);g=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);eig=g['eigenvalues'][:4];gain=float(g['lambdas'][0]);old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];cells=torch.zeros(96,2,2,dtype=torch.float64);total=torch.zeros(96,2,dtype=torch.float64);checks=[]
for i,row in enumerate(rows):
 j=row['donor_id'];n=len(row['ids']);c=int(v['cue_positions'][i]);q=a['parents'][i,:n,1];dB=a['parents'][j,:n,0]-a['parents'][i,:n,0];dH=a['parents'][j,:n,2]-a['parents'][i,:n,2];r=a['rho8_squared'][i,:n]
 parts=torch.stack([2*(q*dB*eig).sum(-1)/r,2*(q*dH*eig).sum(-1)/r],-1);s=2*(q*(dB+dH)*eig).sum(-1)/r;checks.append(float((parts.sum(-1)-s).norm()/s.norm().clamp_min(1e-30)));coef=v['gamma'][i,n-1,:n]*gain/v['rho9'][i,:n];sgn=-1 if i%2==0 else 1
 for src in range(2):
  ix=slice(c,c+1) if src==0 else slice(c+1,n);cells[i,src]=(coef[ix]@parts[ix])*sgn;total[i,src]=(coef[ix]@s[ix])*sgn
records=[]
for group in range(4):
 ix=slice(group*24,(group+1)*24);ss=[]
 for src in range(2):
  y=total[ix,src];x=cells[ix,src];ss.append(dict(source=['city','postcity'][src],B_H_aligned=(x.T@y/y.square().sum()).tolist(),B_H_mean=x.mean(0).tolist(),B_only_error=float((x[:,0]-y).norm()/y.norm()),H_only_error=float((x[:,1]-y).norm()/y.norm())))
 records.append(dict(group=group,cells=ss))
assert max(checks)<1e-10
result=dict(max_identity_error=max(checks),records=records,scope='Computational fixedQ/R partner swap split, not physicalsingle-parent intervention. B includesresidual/reentry/bias; H allattention8 heads, noindividualhead identification.')
(P/'MLP7_QPATH_PARTNER_SPLIT_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
