"""Earlier-input derivative at fixed later native h. This is a partial derivative,
not upstream total causal response. Predicted bars registered on root board.
"""
from pathlib import Path
import torch,json
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
ps=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)
winner=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text())['winner']
z=d['z'][d['indices']];h=d['h'][d['indices']];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()
def derivative(Q,linear,bias):
 reads=torch.einsum('ni,oij,nj->no',z,Q,z)+z@linear+bias
 derivatives=2*torch.einsum('oij,nj->noi',Q,z)+linear.T[None]
 result=[]
 for j,pair in enumerate(d['pairs'][:2]):
  a=(h@pair['a']-.5*reads[:,2*j])/scale-pair['alpha']
  b=reads[:,2*j+1]/scale-pair['beta']
  result.append(-.5*(b/scale)[:,None]*derivatives[:,2*j]+(a/scale)[:,None]*derivatives[:,2*j+1])
 return torch.stack(result,1)
Q=torch.stack([q for p in d['pairs'][:2] for q in p['Qs']]);native=derivative(Q,torch.zeros(1152,4,dtype=z.dtype),torch.zeros(4,dtype=z.dtype))
results={};controls=[]
for key,p in ps.items():
 s=p['shared_mixed'];raw=torch.einsum('ir,ro,jr->oij',s['left_reader'],s['product_weights'],s['right_reader']);Qs=(raw+raw.transpose(-1,-2))/2
 results[key]=derivative(Qs,s['source_linear'],s['source_bias'])
 # Independent central difference of direct factor executor at one state.
 from shared_mixed_source_graph import component_scalars
 direction=torch.randn(z.shape[-1],generator=torch.Generator().manual_seed(749),dtype=z.dtype);direction/=direction.norm();eps=1e-4
 finite=(component_scalars(z[:1]+eps*direction,h[:1],p)-component_scalars(z[:1]-eps*direction,h[:1],p))[0,:2]/(2*eps)
 analytical=results[key][0]@direction
 controls.append(float((finite-analytical).norm()/analytical.norm()))
def tangent(J):
 return J-(J*z[:,None,:]).sum(-1,keepdim=True)/z.square().sum(-1)[:,None,None]*z[:,None,:]
records=[]
for key,J in results.items():
 def err(T):return [float((J[:,i]-T[:,i]).norm()/T[:,i].norm()) for i in range(2)]
 records.append(dict(candidate=key,native_relative_jacobian_error=err(native),winner_relative_jacobian_difference=err(results[winner]),tangent_native_relative_error=[float((tangent(J)[:,i]-tangent(native)[:,i]).norm()/tangent(native)[:,i].norm()) for i in range(2)]))
out=dict(records=records,finite_difference_errors=controls,predictions=dict(pred_a_instrument=max(controls)<1e-5,pred_b_native=all(max(r['native_relative_jacobian_error'])<=.15 for r in records),pred_c_crossfit=all(max(r['winner_relative_jacobian_difference'])<=.10 for r in records)),scope='448opened states. Derivative with respect to z at fixed h and its RMS denominator. Tangent errors additionally project gradients onto the sphere orthogonal to each z; neither measures full upstream causal response. No native model forwards.')
(P/'PROFILED_SENSITIVITY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
