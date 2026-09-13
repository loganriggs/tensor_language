"""Three-input factorial of the Q-containing composite; no semantic Shapley claim."""
from pathlib import Path
from math import factorial
import json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 a=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);v=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);actual=torch.load(P/'MLP7_PHI_PATH_DONATION_V1_ARTIFACT.pt',weights_only=True)['donor_fields'];g=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);eig=g['eigenvalues'][:4];gain=float(g['lambdas'][0])
 old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh]
 cols=torch.zeros(96,2,3,dtype=torch.float64);single=torch.zeros_like(cols);target=torch.zeros(96,2,dtype=torch.float64);checks=[];account=[]
 for i,row in enumerate(rows):
  j=row['donor_id'];n=len(row['ids']);c=int(v['cue_positions'][i]);qs=[a['parents'][k,:n,1] for k in [i,j]];ss=[a['parents'][k,:n,0]+a['parents'][k,:n,2] for k in [i,j]];rr=[a['rho8_squared'][k,:n] for k in [i,j]]
  corners=[]
  for mask in range(8):
   q=qs[bool(mask&1)];s=ss[bool(mask&2)];r=rr[bool(mask&4)];corners.append(((q.square()+2*q*s)*eig).sum(-1)/r)
  pieces=[]
  for bit in range(3):
   term=torch.zeros(n,dtype=torch.float64)
   for mask in range(8):
    if mask&(1<<bit):continue
    k=mask.bit_count();term+=factorial(k)*factorial(2-k)/6*(corners[mask|(1<<bit)]-corners[mask])
   pieces.append(term)
  pieces=torch.stack(pieces,-1);dv=corners[7]-corners[0];account.append(float((pieces.sum(-1)-dv).norm()/dv.norm().clamp_min(1e-30)));factor=gain/v['rho9'][i,:n];gamma=v['gamma'][i,:n,:n]
  for source in range(2):
   mask=torch.zeros(n,dtype=torch.float64)
   if source==0:mask[c]=1
   else:mask[c+1:]=1
   field=gamma@(dv*factor*mask);ref=actual[i,2 if source==0 else 4,:n];checks.append(float((field-ref).norm()/ref.norm().clamp_min(1e-30)))
   sgn=-1 if i%2==0 else 1;target[i,source]=field[-1]*sgn
   cols[i,source]=(gamma[-1]@(pieces*(factor*mask)[:,None]))*sgn
   for bit in range(3):single[i,source,bit]=(gamma[-1]@((corners[1<<bit]-corners[0])*factor*mask))*sgn
 records=[]
 for group in range(4):
  ix=slice(group*24,(group+1)*24);cells=[]
  for source in range(2):
   y=target[ix,source];pred=cols[ix,source];one=single[ix,source];den=y.square().sum().clamp_min(1e-30)
   cells.append(dict(source=['city','postcity'][source],shapley_aligned=(pred.T@y/den).tolist(),single_port_relative_errors=[float((one[:,k]-y).norm()/y.norm().clamp_min(1e-30)) for k in range(3)],mean_full_write=float(y.mean())))
  records.append(dict(group=group,cells=cells))
 assert max(checks)<1e-4 and max(account)<1e-10
 result=dict(ports=['Q7_readings','B_plus_H_readings','RMS8_squared'],max_field_replay_error=max(checks),max_shapley_sum_error=max(account),records=records,Qonly_nearpost_20percent_pass=records[2]['cells'][1]['single_port_relative_errors'][0]<=.2,scope='Exact native/donor 8corner computational allocation. Shapley averages hybridinput marginal changes, not native causal responsibility. One-port swap keeps otherrecipientinputs; full pathdonation changesallthree. OriginalQgroupcausalscreen remains composite.')
 (P/'MLP7_QPATH_INPUT_ALLOCATION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
