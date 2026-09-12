from pathlib import Path
import torch,json
from cubic_projected_function_compare_v1 import compare,solve
from cubic_secant_coordinates_v1 import encode,components,features
from shared_cubic_source_projection_v1 import cross_factors
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(9122131)
 w=(torch.randn(2,2,3),torch.randn(2,2,4),torch.randn(2,2,3),torch.randn(2,2,4),torch.randn(2,2,4),torch.randn(2,2,2))
 a=torch.randn(3,3,4);b=torch.randn(3,3,4);theta,t=encode(b,(0,2));bb,mb=components(theta,t);ma=torch.eye(3)
 def dense(atoms,mix):
  phi=mix@features(atoms);aa,bq,c=cross_factors(atoms,*w)
  coef=torch.einsum('hrti,hrtj,hrto->hroij',aa,bq,c);coef=(coef+coef.transpose(-1,-2))/2
  coef=torch.einsum('ra,haoij->hroij',mix,coef);g=phi@phi.T
  weights=torch.stack([solve(g,v.flatten(1)).reshape(v.shape) for v in coef])
  return torch.einsum('rs,hroij->hsoij',phi,weights)
 x,y=dense(a,ma),dense(bb,mb);result=compare((a,ma),(bb,mb),w)
 expected={'first_energy':x.square().flatten(1).sum(1),'second_energy':y.square().flatten(1).sum(1),'cross_inner_product':(x*y).flatten(1).sum(1),'squared_difference':(x-y).square().flatten(1).sum(1)}
 errors={k:float((result[k]-v).norm()/v.norm().clamp_min(1e-30)) for k,v in expected.items()}
 # Reordering/rescaling readers leaves the full projected function unchanged.
 transformed=a[[2,0,1]]*torch.tensor([2.,.5,-3.])[:,None,None]
 identity=compare((a,ma),(transformed,ma),w);errors['reparameterized_function']=float(abs(identity['squared_difference'].sum())/identity['first_energy'].sum())
 out=P/'CUBIC_PROJECTED_FUNCTION_COMPARE_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps({'errors':errors,'passed':max(errors.values())<=1e-10},indent=2)+'\n');print(errors);assert max(errors.values())<=1e-10
if __name__=='__main__':main()
