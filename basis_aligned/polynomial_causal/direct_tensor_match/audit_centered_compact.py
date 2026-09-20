"""Audit the two training-selected centered programs on cached diagnostic panels."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def evaluate(s,x):
 delta=x-s['mu'].double();return s['constant'].double()+(delta@s['linear_reader'].double().T)@s['linear_writer'].double().T+((delta@s['quadratic_left'].double().T)*(delta@s['quadratic_right'].double().T))@s['quadratic_writer'].double().T

def main():
 torch.set_num_threads(4);source=torch.load(P/'NATIVE_CENTERED_COMPACT_V1.pt',weights_only=True);result=json.load(open(P/'NATIVE_CENTERED_COMPACT_V1.json'));panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets'];rows=[]
 for winner in result['winners']:
  key=tuple(winner[k] for k in ['metric','linear_rank','quadratic_width','optimizer','seed']);s=source['students'][key];record=dict(key=list(key),panels=[])
  for panel,target in zip(panels,targets):
   prediction=evaluate(s,panel['rows'].double());pc=prediction-prediction.mean(0);tc=target-target.mean(0);residual=prediction-target;mean=residual.mean(0);centered=pc-tc;den=target.square().sum();meanenergy=len(target)*mean.square().sum()/den;centeredenergy=centered.square().sum()/den;total=residual.square().sum()/den;check=float(abs(total-meanenergy-centeredenergy));assert check<1e-12
   record['panels'].append(dict(total_relative_error=float(total.sqrt()),centered_relative_error=float(centered.norm()/tc.norm()),centered_cosine=float((pc*tc).sum()/(pc.norm()*tc.norm())),mean_error_energy_fraction=float(meanenergy),centered_error_energy_fraction=float(centeredenergy),identity_error=check))
  rows.append(record)
 out=dict(records=rows,scope='Frozen training-selected programs; same reused diagnostic panels. Mean/variation separation, no further model selection and no OOD claim.');(P/'CENTERED_COMPACT_VARIATION_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
