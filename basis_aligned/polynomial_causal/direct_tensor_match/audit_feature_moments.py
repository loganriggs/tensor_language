"""Input-only higher-moment diagnostic in frozen candidate feature directions."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);s={k:v.double() for k,v in torch.load(P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt',weights_only=True)['programs']['centered'].items()};panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];M=panels[0]['covariance'].double();a,b=s['quadratic_left'],s['quadratic_right'];A=torch.cat([s['linear_reader'],a,b]);variance=((A@M)*A).sum(1);qm=((a@M)*b).sum(1);aa=a@M@a.T;bb=b@M@b.T;ab=a@M@b.T;gaussian=aa*bb+ab*ab.T;ev,E=torch.linalg.eigh(gaussian);inv=(E*ev.clamp_min(1e-20).rsqrt())@E.T;records=[]
 for i,panel in enumerate(panels):
  x=panel['rows'].double()-s['mu'];z=(x@A.T)/variance.sqrt();q=(x@a.T)*(x@b.T);qc=q-q.mean(0);emp=qc.T@qc/len(q);spectrum=torch.linalg.eigvalsh(inv@emp@inv);records.append(dict(panel=i,projection_standardized_second_moments=z.square().mean(0).tolist(),projection_standardized_third_moments=z.pow(3).mean(0).tolist(),projection_standardized_fourth_moments=z.pow(4).mean(0).tolist(),gaussian_fourth_reference=3.,quadratic_mean_relative_mismatch=float((q.mean(0)-qm).norm()/qm.norm()),quadratic_covariance_relative_mismatch=float((emp-gaussian).norm()/gaussian.norm()),quadratic_covariance_generalized_eigenvalues=spectrum.tolist()))
 out=dict(direction_order='8 frozen linear readers,4 quadratic left readers,4 quadratic right readers',records=records,scope='Input moments only; calibration mu,M fixed. Gaussian fourth-moment prediction may fail even when second moments agree. Panel1 also includes input moment shift. No causal attribution or fitting.');(P/'FEATURE_MOMENT_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
