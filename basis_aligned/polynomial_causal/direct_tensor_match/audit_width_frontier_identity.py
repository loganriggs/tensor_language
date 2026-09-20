"""Same gauge-aware centered feature comparison at smaller dictionary widths."""
import itertools,json
from pathlib import Path
import torch
from audit_shared_bank_identity import compare
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);fits=torch.load(P/'BANK_WIDTH_FRONTIER_V1.pt',weights_only=True)['allfits'];core=torch.load(P/'QUARTIC_BANK_CORE_V1.pt',weights_only=True);mu=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['mean'].double();m=core['input_mapback']@mu;groups=[]
 for width in [4,6]:
  rows=[];chosen={k:f for k,f in fits.items() if k[0]==width}
  for (ka,a),(kb,b) in itertools.combinations(chosen.items(),2):rows.append(dict(first=list(ka),second=list(kb),**compare(a,b,m)))
  groups.append(dict(width=width,pairs=rows,minimum_centered_feature_cosine=min(r['centered_feature_minimum_cosine'] for r in rows),median_centered_feature_cosine=float(torch.tensor([r['centered_feature_mean_cosine'] for r in rows]).quantile(.5)),minimum_centered_function_cosine=min(r['centered_bank_function_cosine'] for r in rows)))
 out=dict(groups=groups,scope='Descriptive random-start comparison includes all rates/optimizers, including underfit arms; do not treat low similarity alone as proof of nonidentifiability. Same warm-start width8 audit used different starts.');(P/'WIDTH_FRONTIER_IDENTITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print([{k:v for k,v in g.items() if k!='pairs'} for g in groups])
if __name__=='__main__':main()
