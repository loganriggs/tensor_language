"""Dense full-symmetry and gradient oracle for joint routing times value."""
from pathlib import Path
import itertools,json
import torch
from joint_routing_value_polynomial_v1 import contract
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73270);dt=torch.float64
    q1=torch.randn(2,2,2,dtype=dt,requires_grad=True);q2=torch.randn_like(q1,requires_grad=True)
    k1=torch.randn(2,2,3,dtype=dt,requires_grad=True);k2=torch.randn_like(k1,requires_grad=True)
    v=torch.randn_like(k1,requires_grad=True);o=torch.randn(3,2,2,dtype=dt,requires_grad=True)
    a=torch.einsum('hka,hkc->hac',q1,k1);b=torch.einsum('hkb,hkd->hbd',q2,k2);m=torch.einsum('ohk,hke->hoe',o,v)
    raw=torch.einsum('hac,hbd,hoe->oabcde',a,b,m)
    tensor=sum(raw.permute(0,*[1+i for i in qp],*[3+i for i in sp]) for qp in itertools.permutations(range(2)) for sp in itertools.permutations(range(3)))/12
    q=torch.randn(7,2,2,dtype=dt);s=torch.randn(7,3,3,dtype=dt)
    pred=contract(q,s,q1,k1,q2,k2,v,o)
    exact=torch.einsum('oabcde,na,nb,nc,nd,ne->no',tensor,q[:,0],q[:,1],s[:,0],s[:,1],s[:,2])
    perm=contract(q.flip(1),s.flip(1),q1,k1,q2,k2,v,o)
    qq=q[:,0];ss=s[:,0];g=torch.randn(7,2,dtype=dt)
    direct=torch.einsum('nh,nhe,hoe->no',torch.einsum('na,hac,nc->nh',qq,a,ss)*torch.einsum('nb,hbd,nd->nh',qq,b,ss)*g,ss[:,None].expand(-1,2,-1),m)
    diagonal=contract(qq[:,None].expand(-1,2,-1),ss[:,None].expand(-1,3,-1),q1,k1,q2,k2,v,o,g)
    params=(q1,k1,q2,k2,v,o);probe=torch.randn_like(pred)
    pg=torch.autograd.grad((pred*probe).sum(),params,retain_graph=True);eg=torch.autograd.grad((exact*probe).sum(),params)
    result=dict(dense_error=float(((pred-exact).norm()/exact.norm()).detach()),permutation_error=float(((pred-perm).norm()/pred.norm()).detach()),diagonal_error=float(((diagonal-direct).norm()/direct.norm()).detach()),gradient_errors=[float((x-y).norm()/y.norm()) for x,y in zip(pg,eg)])
    assert max([result[k] for k in ('dense_error','permutation_error','diagonal_error')]+result['gradient_errors'])<1e-10
    out=P/'JOINT_ROUTING_VALUE_POLYNOMIAL_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
