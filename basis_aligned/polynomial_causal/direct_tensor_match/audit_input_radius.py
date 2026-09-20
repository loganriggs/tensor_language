"""Compare observed normalized-input radius with the fitted Gaussian law."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];mu=panels[0]['mean'].double();M=panels[0]['covariance'].double();gaussian_mean=M.trace()+mu.square().sum();gaussian_var=2*M.square().sum()+4*mu@M@mu;records=[]
    for i,panel in enumerate(panels):
        x=panel['rows'].double();radius=x.square().sum(1);mean=radius.mean();var=radius.var(unbiased=False)
        records.append(dict(panel=i,radius_squared_mean=float(mean),radius_squared_std=float(var.sqrt()),relative_radius_squared_std=float(var.sqrt()/mean),gaussian_expected_squared_radius=float(gaussian_mean),gaussian_squared_radius_std=float(gaussian_var.sqrt()),gaussian_relative_squared_radius_std=float(gaussian_var.sqrt()/gaussian_mean),relative_gaussian_radius_variance_mismatch=float((gaussian_var-var)/mean.square()),max_relative_deviation=float((radius/mean-1).abs().max())))
    out=dict(records=records,scope='Input-only diagnostic, calibration Gaussian fixed from panel0. Radius mismatch does not by itself establish its effect on output reconstruction. No fit or model selection.')
    (P/'INPUT_RADIUS_GAUSSIAN_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
