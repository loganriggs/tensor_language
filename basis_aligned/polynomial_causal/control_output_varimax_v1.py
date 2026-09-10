"""Dense folded tensor, directional derivative, rotation and planted controls."""
import json
from pathlib import Path
import torch
from orthogonal_output_varimax_v1 import output_factors,criterion,fit,sparsity
from sparse_core_stiefel_v1 import project_tangent,retract


def main():
    torch.set_num_threads(2);torch.manual_seed(978)
    u,l,r,d=[torch.randn(*s,dtype=torch.float64) for s in [(11,4),(7,5),(7,5),(4,7)]]
    f=output_factors(u,l,r,d,3)
    native=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
    dense=torch.einsum('vk,kij->vij',(u-u.mean(0))@d,native).flatten(1)
    left,sv,right=torch.linalg.svd(dense,full_matrices=False)
    cores=torch.einsum('jk,kab->jab',f['core'],native).flatten(1)
    bridge=float((f['loadings']@cores-(left[:,:3]*sv[:3])@right[:3]).norm()/dense.norm())
    gram_error=float((cores@cores.T-torch.eye(3)).abs().max())
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));h=project_tangent(q,torch.randn_like(q));x=f['loadings']/f['loadings'].square().mean().sqrt()
    score,g=criterion(x,q,True);eps=1e-6
    finite=abs(float((criterion(x,retract(q,h,eps))-criterion(x,retract(q,h,-eps)))/(2*eps)-(g*h).sum()))
    rotation_bridge=float(((f['loadings']@q)@(q.T@cores)-f['loadings']@cores).norm()/dense.norm())
    # Independent sparse leptokurtic factors with overlapping row membership.
    truth=torch.randn(3000,4,dtype=torch.float64)*(torch.rand(3000,4)<.12)
    mix,_=torch.linalg.qr(torch.randn(4,4,dtype=torch.float64));observed=truth@mix
    result=fit(observed,seconds=30,max_steps=400)
    alignment=(mix@result['rotation']).abs().max(0).values.min()
    # Exact nonlinear state continuation uses the same deterministic arithmetic.
    short=fit(observed,seconds=30,max_steps=10);long=fit(observed,seconds=30,max_steps=20)
    resumed=fit(observed,seconds=30,max_steps=10,state=short)
    resume=float((long['rotation']-resumed['rotation']).abs().max())
    out=dict(dense_projection_relative_error=bridge,orthogonal_function_error=gram_error,rotation_replay_relative_error=rotation_bridge,finite_gradient_absolute_error=finite,planted_minimum_axis_alignment=float(alignment),planted_converged=result['converged'],resume_max_difference=resume,before=sparsity(observed),after=sparsity(observed@result['rotation']))
    out['passed']=max(bridge,gram_error,rotation_bridge,finite,resume)<=1e-7 and alignment>.99 and result['converged']
    Path(__file__).with_name('OUTPUT_VARIMAX_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['passed']

if __name__=='__main__':main()
