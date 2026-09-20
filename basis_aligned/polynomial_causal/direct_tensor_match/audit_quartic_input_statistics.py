"""CPU covariance geometry audit; diagnostic only, no feature selection."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);capture=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];bases=torch.load(P/'NATIVE_INPUT_MODE_V1.pt',weights_only=True)['projection_bases'];rows=[]
 for n,p in enumerate(capture):
  mean=p['mean'].double();cov=p['covariance'].double();second=p['second_moment'].double();ev=torch.linalg.eigvalsh(cov);second_ev=torch.linalg.eigvalsh(second);normalized=ev/(ev.sum()/len(ev));row=dict(panel=n,mean_energy_fraction=float(mean.square().sum()/second.trace()),centered_condition=float(ev[-1]/ev[0]),normalized_minimum=float(normalized[0]),centered_floor01_count=int((normalized<.01).sum()),centered_floor10_count=int((normalized<.1).sum()),second_floor01_count=int((second_ev/(second_ev.sum()/len(ev))<.01).sum()),projections=[])
  for name,basis in bases.items():
   Q=torch.linalg.qr(basis.double())[0];row['projections'].append(dict(name=name,dimension=Q.shape[1],mean_fraction=float((Q.T@mean).square().sum()/mean.square().sum()),centered_energy_fraction=float((Q*(cov@Q)).sum()/cov.trace()),second_moment_fraction=float((Q*(second@Q)).sum()/second.trace())))
  rows.append(row)
 a,b=[p['mean'].double() for p in capture];cosine=float(a@b/a.norm()/b.norm());out=dict(panels=rows,mean_cosine=cosine,scope='Input second-moment coverage only, not quartic function or causal sufficiency. No features fit on panel2. Floorcounts flag when configured floor is inactive.')
 (P/'QUARTIC_INPUT_STATISTICS_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
