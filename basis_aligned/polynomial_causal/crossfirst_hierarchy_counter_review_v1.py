"""Diagnose failed linear-effect composition without rewriting the criterion."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);z=torch.load(P/'CROSSFIRST_HIERARCHY_V1_ARTIFACT.pt',weights_only=True)['measures'];records=[]
 for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
  for metric in ([0] if lo<96 else [0,1]):
   a=z[lo:hi,:,metric];child=a[:,1]-a[:,0];contextual=a[:,2]-a[:,3];gain=float((child*contextual).sum()/child.square().sum().clamp_min(1e-30));res=contextual-gain*child;records.append(dict(cell=label,metric='regional_margin' if lo<96 else ['CE','newline_margin'][metric],native_child_norm=float(child.norm()),oracle_gain=gain,residual_after_oracle_gain_over_child=float(res.norm()/child.norm().clamp_min(1e-30)),unscaled_error_over_child=float((contextual-child).norm()/child.norm().clamp_min(1e-30)),same_sign_count=int((child*contextual>0).sum()),count=hi-lo,max_absolute_discrepancy=float((contextual-child).abs().max())))
 result=dict(scope='Post-hoc exact outcome analysis; least-squares scalar gain is diagnostic only, not fitted/deployed predictor. Original child-relative10% criterion remains failed. CE andmargin are distinctoutputs, neither replacesregisteredmetric.',records=records);(P/'CROSSFIRST_HIERARCHY_V1_COUNTER_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
