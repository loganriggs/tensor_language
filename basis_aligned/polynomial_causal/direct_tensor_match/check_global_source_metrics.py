from pathlib import Path
import json,torch
from global_mixed_source_graph import component_scalars,score
P=Path(__file__).parent;torch.set_num_threads(2)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
p=next(iter(torch.load(P/'MIXED_PRODUCT_NATIVE_PROGRAMS_V1.pt',weights_only=True).values()))
d['indices']=d['indices'][:4];ids=d['indices'];z=d['z'][ids].clone().requires_grad_();h=d['h'][ids];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
phi=component_scalars(z,h,p);qs=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);reads=torch.einsum('ni,oij,nj->no',z,qs,z)
truth=torch.stack([((h@pair['a']-.5*reads[:,2*j])/s-pair['alpha'])*(reads[:,2*j+1]/s-pair['beta']) for j,pair in enumerate(d['pairs'])],1)
errors=[]
for j in range(3):
 a=torch.autograd.grad(phi[:,j].sum(),z,retain_graph=True)[0];b=torch.autograd.grad(truth[:,j].sum(),z,retain_graph=True)[0];errors.append(float((a-b).norm()/b.norm()))
r=score(p,d);replay=max(abs(a-b) for a,b in zip(errors,r['euclidean_jacobian_errors']));assert replay<1e-10
out=dict(three_output_autograd_replay=replay,score=r,scope='Four cached rows, independent autograd verifies analytic fixed-h Jacobian metric for all three outputs.')
(P/'GLOBAL_SOURCE_METRICS_PREFLIGHT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
