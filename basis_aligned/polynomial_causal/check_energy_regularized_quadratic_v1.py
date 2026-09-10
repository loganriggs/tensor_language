"""Dense objective and envelope-gradient controls for the explicit energy bias."""
import json
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel
from energy_regularized_quadratic_v1 import EnergyRegularizedObjective
from joint_quadratic_fit_v1 import product_cross

def main():
    torch.set_num_threads(2);torch.manual_seed(9116401);dt=torch.float64
    l,r=torch.randn(4,5,dtype=dt),torch.randn(4,5,dtype=dt)
    d=torch.randn(3,4,dtype=dt);u=torch.randn(7,3,dtype=dt);metric=u.T@u
    h=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
    target=torch.einsum('vk,kij->vij',u@d,h)
    x=torch.randn(47,5,dtype=dt);y=((x@l.T)*(x@r.T))@d.T
    results=[]
    for kind in ['product','square','shared_reader','block']:
        for mode in ['weight','data']:
            model=QuadraticModel(kind,5,products=2,readers=3,groups=2,block_size=2)
            total=target.square().sum() if mode=='weight' else ((y@metric)*y).sum()
            kwargs=dict(l=l,r=r,d=d) if mode=='weight' else dict(x=x,y=y)
            obj=EnergyRegularizedObjective(metric,total,penalty=.01,**kwargs)
            loss,w,g=obj.loss(model);a,b,c=model.components()
            if mode=='weight':
                forms=(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:])/2
                feature=torch.einsum('kj,kab->jab',c,forms)
                terms=torch.einsum('vj,jab->jvab',u@w,feature)
                direct=((target-terms.sum(0)).square().sum()+.01*terms.square().sum())/total
            else:
                feature=((x@a.T)*(x@b.T))@c
                terms=torch.einsum('nj,vj->jnv',feature,u@w)
                direct=((y@u.T-terms.sum(0)).square().sum()+.01*terms.square().sum())/total
            error=abs(float((loss-direct).detach()));loss.backward()
            # All parameter blocks, including block coefficients and dictionary codes.
            gradient_errors=[]
            for p in model.parameters():
                analytic=float(p.grad.flatten()[0])
                with torch.no_grad():
                    old=p.flatten()[0].clone();p.flatten()[0]=old+1e-5;plus=float(obj.loss(model)[0]);p.flatten()[0]=old-1e-5;minus=float(obj.loss(model)[0]);p.flatten()[0]=old
                gradient_errors.append(abs(analytic-(plus-minus)/2e-5))
            assert error<=1e-10 and max(gradient_errors)<=1e-6
            results.append(dict(kind=kind,metric=mode,dense_error=error,max_block_gradient_error=max(gradient_errors)))
    out=Path(__file__).with_name('ENERGY_REGULARIZED_QUADRATIC_V1_CONTROL.json')
    with out.open('x') as f:json.dump(dict(controls=results,passed=True),f,indent=2);f.write('\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
