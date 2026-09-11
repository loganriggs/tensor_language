"""Dense envelope, finite-difference, gauge, and emitted-graph controls."""
import json
from pathlib import Path
import numpy as np
import torch
from shared_reader_variable_projection_v2 import Objective
from ll1_joint_parent_graph_v2 import build,execute,factors
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def planted(seed=2901):
    torch.manual_seed(seed);d,r,m,o=9,3,3,5
    frame=torch.linalg.qr(torch.randn(d,d)).Q
    readers=torch.stack((frame[:,0],.2*frame[:,0]+.98*frame[:,1]))
    readers=readers/readers.norm(dim=1,keepdim=True)
    nodes=[dict(consumers=[0,1,2]),dict(consumers=[0,2])]
    aa,ss,cc=[],[],[]
    for g in range(m):
        ids=[i for i,n in enumerate(nodes) if g in n['consumers']]
        private=frame[:,2+2*g:2+2*g+r-len(ids)].T
        b=torch.cat((readers[ids],private))
        h=torch.randn(r,r);h=(h+h.T)/2
        vals,vec=torch.linalg.eigh(h)
        aa.append(vec.T@b);ss.append(vals);cc.append(torch.randn(o))
    parts=(torch.stack(aa),torch.stack(ss),torch.stack(cc))
    graph=build(*parts,readers,nodes)
    return cp(*parts),graph


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    target,graph=planted();total=float(dense(*target).square().sum())
    objective=Objective(target,graph,total)
    rng=np.random.default_rng(2902)
    point=objective.initial+.015*rng.normal(size=len(objective.initial))
    actual,gradient=objective.evaluate(point)
    leaf=torch.tensor(point,requires_grad=True)
    bases,writers,_,_=objective.maps(leaf)
    core=objective.cores.detach()
    fitted=torch.einsum('go,gdi,gij,gej->ode',writers,bases,core,bases)
    loss=((fitted-dense(*target)).square().sum()+objective.penalty*core.square().sum())/total
    expected=torch.autograd.grad(loss,leaf)[0].numpy()
    gradient_error=float(np.linalg.norm(gradient-expected)/np.linalg.norm(expected))
    loss_error=abs(actual-float(loss.detach()))
    direction=rng.normal(size=len(point));direction/=np.linalg.norm(direction)
    h=1e-6
    fd=(objective.evaluate(point+h*direction)[0]-objective.evaluate(point-h*direction)[0])/(2*h)
    analytic=float(gradient@direction)
    fd_error=abs(fd-analytic)/max(abs(fd),abs(analytic),1e-8)
    objective.evaluate(point);emitted=objective.physical(point)
    x=torch.randn(17,9)
    expected_values=torch.einsum('ode,nd,ne->no',fitted.detach(),x,x)
    executor_error=float((execute(emitted,x)-expected_values).norm()/expected_values.norm())
    original_tensor=dense(*cp(*factors(emitted)))
    # Signed row scales change QR coordinates but must preserve the optimized
    # function, including the sign conventions of all emitted cores.
    scaled=point.copy();blocks=[];offset=0
    for shape,size in zip(objective.shapes,objective.sizes):
        block=scaled[offset:offset+size].reshape(shape)
        if len(block):block*=np.geomspace(.2,5.,len(block))[:,None]*np.where(np.arange(len(block))%2,-1,1)[:,None]
        offset+=size
    scaled_loss,_=objective.evaluate(scaled)
    scaled_tensor=dense(*cp(*factors(objective.physical(scaled))))
    gauge_error=float((scaled_tensor-original_tensor).norm()/original_tensor.norm())
    result=dict(loss_error=loss_error,dense_envelope_gradient_error=gradient_error,
                directional_finite_difference_error=fd_error,executor_error=executor_error,
                signed_scale_function_error=gauge_error,signed_scale_loss_error=abs(scaled_loss-actual),
                inner_normal_residual=objective.last['inner']['normal_residual'],
                reduced_identity_error=objective.last['reduced_identity_error'],
                held=gradient_error<=1e-8 and fd_error<=1e-6 and max(executor_error,gauge_error,loss_error)<=1e-9)
    (Path(__file__).parent/'SHARED_READER_VARIABLE_PROJECTION_V2_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);assert result['held'],result


if __name__=='__main__':main()
