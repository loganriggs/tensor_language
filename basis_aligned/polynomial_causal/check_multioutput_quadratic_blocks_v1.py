"""Dense tensor, symmetry, core-norm and all-parameter-block gradient controls."""
import json
from pathlib import Path
import torch
from multioutput_quadratic_blocks_v1 import MultioutputQuadraticBlocks,MultioutputWeightObjective
from joint_quadratic_fit_v1 import product_cross
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(91162200)
    dim,outputs,hidden=6,5,9
    l,r=torch.randn(hidden,dim,dtype=torch.float64),torch.randn(hidden,dim,dtype=torch.float64);d=torch.randn(outputs,hidden,dtype=torch.float64)
    u=torch.randn(8,outputs,dtype=torch.float64);m=u.T@u
    t=torch.einsum('ok,ki,kj->oij',d,l,r);t=.5*(t+t.transpose(-1,-2));total=torch.einsum('oij,op,pij->',t,m,t)
    model=MultioutputQuadraticBlocks(dim,groups=2,rank=3,outputs=2);objective=MultioutputWeightObjective(m,total,l=l,r=r,d=d,penalty=.01)
    loss,w,gram=objective.loss(model);e,c=model.components();q=torch.einsum('gri,gmrs,gsj->gmij',e,c,e).reshape(4,dim,dim)
    dense_cross=torch.einsum('oij,kij->ok',t,q);dense_gram=torch.einsum('kij,lij->kl',q,q);cross,_=objective.cross_gram(model)
    cross_error=float((cross-dense_cross).abs().max());gram_error=float((gram-dense_gram).abs().max());assert cross_error<=1e-10 and gram_error<=1e-10
    prediction=torch.einsum('ok,kij->oij',w,q);delta=prediction-t;og=w.T@m@w
    dense_loss=(torch.einsum('oij,op,pij->',delta,m,delta)+.01*(og.diag()*dense_gram.diag()).sum())/total
    loss_error=abs(float((loss-dense_loss).detach()));assert loss_error<=1e-10
    loss.backward();errors=[]
    for p in model.parameters():
        v=torch.randn_like(p);v/=v.norm();analytic=float((v*p.grad).sum());h=1e-5
        with torch.no_grad():
            saved=p.clone();p.copy_(saved+h*v);plus=float(objective.loss(model)[0]);p.copy_(saved-h*v);minus=float(objective.loss(model)[0]);p.copy_(saved)
        errors.append(abs(analytic-(plus-minus)/(2*h)))
    assert max(errors)<=1e-6
    symmetry=float((c-c.transpose(-1,-2)).abs().max());norm_error=float((c.square().sum((-1,-2))-1).abs().max());assert symmetry==0 and norm_error<=1e-12
    result=dict(schema='multioutput.quadratic.blocks.control.v1',passed=True,cross_max_absolute_error=cross_error,gram_max_absolute_error=gram_error,dense_loss_absolute_error=loss_error,gradient_finite_difference_errors=errors,core_symmetry_max_error=symmetry,core_unit_norm_error=norm_error,scope='Signed overlapping input blocks, multiple quadratic features and output writers per block. No native model fit, orthogonality assumption, data or causal identification.')
    with (P/'MULTIOUTPUT_QUADRATIC_BLOCKS_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
