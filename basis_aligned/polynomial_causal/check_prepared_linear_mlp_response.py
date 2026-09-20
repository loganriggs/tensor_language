from pathlib import Path
import torch,json
import prepared_linear_mlp_response as reduced
import prepared_bilinear_response_tensor as full
torch.set_num_threads(2);torch.manual_seed(174)
b,d,m,r=3,11,19,4
left,right=[torch.randn(m,d,dtype=torch.float64) for _ in range(2)];down=torch.randn(d,m,dtype=torch.float64);h=torch.randn(b,d,dtype=torch.float64);w=torch.randn(b,d,r,dtype=torch.float64);eps=torch.finfo(torch.float32).eps
g=reduced.compile(left,right,down,h,w,eps);f=full.compile(left,right,down,h,w,eps);f['coefficients'][:,:,r:]=0
errors=[]
for scale in [0.,.01,.3,1.,-1.]:
 z=scale*torch.randn(b,r,dtype=torch.float64);errors.append(float((reduced.evaluate(g,z)-full.evaluate(f,z)).abs().max()))
assert max(errors)<1e-10
p=Path(__file__).resolve().parent;out=p/'PREPARED_LINEAR_MLP_RESPONSE_CPU_V1.json';assert not out.exists();out.write_text(json.dumps(dict(max_error=max(errors),scales=[0.,.01,.3,1.,-1.],native_27_writer_values=62614,scope='Direct reduced compiler versus masked exact compiler, random CPU inputs; no native sufficiency claim'),indent=2)+'\n');print(out.read_text())
