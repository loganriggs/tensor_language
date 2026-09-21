"""Describe calibration input geometry before choosing Gaussian metric fits."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def main():
 torch.set_num_threads(2)
 panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True,map_location='cpu')['panels']
 extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True,map_location='cpu')['rows']
 groups={'original_calibration':panels[0]['rows'].double(),'expanded_calibration':torch.cat([panels[0]['rows'].double(),extra.double()]),'opened_evaluation':panels[1]['rows'].double()}
 rows={};saved={}
 for name,x in groups.items():
  mean=x.mean(0);center=x-mean;cov=center.T@center/len(x);second=x.T@x/len(x)
  closure=float((cov+mean[:,None]*mean[None,:]-second).norm()/second.norm());assert closure<1e-12
  stats={}
  for metric,M in [('centered_covariance',cov),('second_moment',second)]:
   e=torch.linalg.eigvalsh((M+M.T)/2).flip(0);total=e.sum();floor=1e-6*total/len(e)
   stats[metric]=dict(trace=float(total),minimum_eigenvalue=float(e[-1]),maximum_eigenvalue=float(e[0]),top4_trace_fraction=float(e[:4].sum()/total),directions_for90percent=int(torch.searchsorted(e.cumsum(0),.9*total))+1,directions_for99percent=int(torch.searchsorted(e.cumsum(0),.99*total))+1,below_trace_scaled_floor=int((e<floor).sum()),proposed_absolute_floor=float(floor))
  rows[name]=dict(states=len(x),mean_squared_norm=float(mean.square().sum()),mean_energy_fraction=float(mean.square().sum()/second.trace()),closure=closure,metrics=stats)
  saved[name]=dict(mean=mean,covariance=cov,second_moment=second)
 reference=saved['expanded_calibration'];held=saved['opened_evaluation'];old=saved['original_calibration']
 shifts={key:dict(original_to_expanded=float((old[key]-reference[key]).norm()/reference[key].norm()),expanded_to_opened_evaluation=float((reference[key]-held[key]).norm()/held[key].norm())) for key in ['mean','covariance','second_moment']}
 # No held-out geometry is exported as a fitting metric.
 torch.save(dict(**reference,states=6144,scope='Calibration only. Centered covariance uses denominator N; secondmoment is E[xx^T]. Raw matrices, no flooring or isotropic blend.'),P/'EXPANDED_INPUT_GEOMETRY_V1.pt')
 result=dict(rows=rows,relative_statistic_shifts=shifts,scope='Input statistics only, not polynomial reconstruction or OOD validation. Gaussian law N(0,secondmoment) matches second moments but not nonzero mean; N(0,covariance) is centered surrogate, not actual activation law.')
 (P/'EXPANDED_INPUT_GEOMETRY_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
