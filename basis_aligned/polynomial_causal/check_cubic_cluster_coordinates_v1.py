from pathlib import Path
import torch,json
from cubic_cluster_coordinates_v1 import encode,components,reconstruct
from cubic_secant_coordinates_v1 import features
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(9122032)
 base=torch.randn(3,5);a=base[None]+.01*torch.randn(3,3,5);theta,chart=encode(a,[0,1,2])
 def represented(x):
  cc,mm=components(x,chart);return mm@features(cc)
 def direct(x):return torch.linalg.solve(chart['loading'],features(reconstruct(x,chart)))
 ref=direct(theta);value=float((represented(theta)-ref).norm()/ref.norm())
 target=torch.randn(ref.shape);x=theta.requires_grad_(True);g=torch.autograd.grad((represented(x)*target).sum(),x)[0];g2=torch.autograd.grad((direct(x)*target).sum(),x)[0];gradient=float((g-g2).norm()/g2.norm())
 # Hessian direction equality protects the proposed matrix-free native solver.
 d=torch.randn_like(theta)
 h1=torch.autograd.functional.hvp(lambda z:represented(z).square().sum(),theta,d)[1]
 h2=torch.autograd.functional.hvp(lambda z:direct(z).square().sum(),theta,d)[1]
 hvp=float((h1-h2).norm()/h2.norm());result=dict(polynomial_relative_error=value,gradient_relative_error=gradient,hvp_relative_error=hvp,passed=max(value,gradient,hvp)<=1e-10,scope='Known non-singular three-product cluster; exact same cubic span and parameter count. No native improvement or global conditioning guarantee.')
 out=P/'CUBIC_CLUSTER_COORDINATES_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result);assert result['passed']
if __name__=='__main__':main()
