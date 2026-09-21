"""Conditional linear-readout extrapolation diagnostic for frozen short-run features."""
import json,time
from pathlib import Path
import torch
from empirical_quartic_dictionary import features

def leverage(z,other,ridge):
 gram=z.T@z;chol=torch.linalg.cholesky(gram+ridge*torch.eye(z.shape[1],dtype=z.dtype));a=torch.linalg.solve_triangular(chol,z.T,upper=False).square().sum(0);b=torch.linalg.solve_triangular(chol,other.T,upper=False).square().sum(0)
 return a,b

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();p=Path(__file__).resolve().parent;identity=torch.eye(3,dtype=torch.float64);a,b=leverage(identity,2*identity,.1);assert torch.allclose(a,torch.full_like(a,1/1.1)) and torch.allclose(b,4*a)
 xs=[r['rows'].double() for r in torch.load(p/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']];rows=[]
 for width in [4,32]:
  for init in ['INHERITED','RANDOM']:
   program=torch.load(p/f'WIDE_NATIVE_QUARTIC_{width}_{init}_V1.pt',weights_only=True);u,v=program['U'].double(),program['V'].double();phi=features(xs[0],u,v);scales=phi.square().mean(0).sqrt().clamp_min(1e-12);z=phi/scales;other=features(xs[1],u,v)/scales;a,b=leverage(z,other,len(z)*1e-6)
   row=dict(width=width,initialization=init.lower(),readout_columns=z.shape[1],conditional_effective_degrees_of_freedom=float(a.sum()),training_leverage_mean=float(a.mean()),evaluation_leverage_mean=float(b.mean()),evaluation_over_training_mean=float(b.mean()/a.mean()),training_quantiles=torch.quantile(a,torch.tensor([.5,.9,.99,1.],dtype=a.dtype)).tolist(),evaluation_quantiles=torch.quantile(b,torch.tensor([.5,.9,.99,1.],dtype=b.dtype)).tolist());rows.append(row)
 pred=dict(pred_a_positive_control=True,pred_b_wide_extrapolation=all(r['evaluation_over_training_mean']>2 for r in rows if r['width']==32))
 result=dict(predictions=pred,records=rows,seconds=time.monotonic()-start,scope='Conditional ridge-readout leverage in training RMS-scaled root features, frozen100stepprograms. This measures feature-design extrapolation, not a causal explanation, generalization theorem, or total degrees of freedom of learned features. Same oldpanels; no fit or output-frame assumptions.')
 (p/'QUARTIC_FEATURE_LEVERAGE_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
