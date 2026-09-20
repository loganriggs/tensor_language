"""Known polynomial identity illustrates Gaussian-vs-normalized-input metric gap."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.manual_seed(2032);d=6;x=torch.randn(131072,d,dtype=torch.float64);sphere=x/x.norm(dim=1,keepdim=True)*d**.5
 def residual(t):return (t.square().sum(1)-d)*t[:,0]*t[:,1]
 g=residual(x).square();s=residual(sphere).square();expected=2*d+24;stderr=float(g.std()/len(g)**.5)
 out=dict(d=d,gaussian_exact_squared_error=expected,gaussian_probe_squared_error=float(g.mean()),gaussian_probe_standard_error=stderr,sphere_probe_squared_error=float(s.mean()),scope='F(x)=||x||^2 x1 x2 and student=d x1 x2 agree on radius sqrt(d). Gaussian error 2d+24 by independent moments. This is a known-identity control, not a native simplification.')
 assert s.mean()<1e-24 and abs(float(g.mean())-expected)<6*stderr
 (P/'SPHERE_METRIC_CONTROL_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
