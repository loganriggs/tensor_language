"""Dense loss and finite-difference controls for representation and metric variants."""
import json,hashlib
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel,QuadraticObjective
from joint_quadratic_fit_v1 import product_cross

def main():
    torch.manual_seed(9116011);torch.set_num_threads(2);dt=torch.float64
    dim=5;native=4;out=3
    l,r=torch.randn(native,dim,dtype=dt),torch.randn(native,dim,dtype=dt)
    d=torch.randn(out,native,dtype=dt);u=torch.randn(7,out,dtype=dt);m=u.T@u
    h=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
    target=torch.einsum('vk,kij->vij',u@d,h)
    x=torch.randn(41,dim,dtype=dt);y=((x@l.T)*(x@r.T))@d.T
    controls={}
    for kind in ['product','square','shared_reader','block']:
        model=QuadraticModel(kind,dim,products=2,readers=3,groups=2,block_size=2)
        for mode in ['weight','data']:
            total=target.square().sum() if mode=='weight' else ((y@m)*y).sum()
            objective=QuadraticObjective(m,total,l=l,r=r,d=d) if mode=='weight' else QuadraticObjective(m,total,x=x,y=y)
            loss,w,g=objective.loss(model);a,b,c=model.components()
            forms=(a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:])/2
            if mode=='weight':
                approx=torch.einsum('vk,kij->vij',(u@w)@c.T,forms)
                direct=(target-approx).square().sum()/total
            else:
                approx=(((x@a.T)*(x@b.T))@c)@w.T
                direct=(((approx-y)@u.T).square().sum())/total
            err=abs(float(loss-direct))
            model.zero_grad(set_to_none=True);loss.backward();p=next(iter(model.parameters()));analytic=float(p.grad.flatten()[0])
            with torch.no_grad():
                old=p.flatten()[0].clone();p.flatten()[0]=old+1e-5;plus=float(objective.loss(model)[0]);p.flatten()[0]=old-1e-5;minus=float(objective.loss(model)[0]);p.flatten()[0]=old
            graderr=abs(analytic-(plus-minus)/2e-5)
            assert err<=1e-10 and graderr<=1e-6
            controls[kind+'_'+mode]=dict(dense_loss_absolute_error=err,gradient_absolute_error=graderr,feature_condition=float(torch.linalg.cond(g)))
    result=dict(controls=controls,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    p=Path(__file__).with_name('STRUCTURED_QUADRATIC_MODELS_V1_CONTROL.json')
    with p.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
