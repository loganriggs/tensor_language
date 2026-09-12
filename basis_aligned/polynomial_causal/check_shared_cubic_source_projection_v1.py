"""Dense projection/gradient/execution oracle for the shared cubic dictionary."""
from pathlib import Path
import itertools,json
import torch
from shared_cubic_source_projection_v1 import atom_gram,cross_factors,query_output_gram,capture,execute
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73310);dt=torch.float64
    q1=torch.randn(2,2,2,dtype=dt);q2=torch.randn_like(q1);k1=torch.randn(2,2,3,dtype=dt);k2=torch.randn_like(k1);v=torch.randn_like(k1);o=torch.randn(3,2,2,dtype=dt)
    atoms=torch.randn(3,3,3,dtype=dt,requires_grad=True)
    a=torch.einsum('hka,hkc->hac',q1,k1);b=torch.einsum('hkb,hkd->hbd',q2,k2);m=torch.einsum('ohk,hke->hoe',o,v)
    raw=torch.einsum('hac,hbd,hoe->hoabcde',a,b,m)
    tensor=sum(raw.permute(0,1,*[2+i for i in qp],*[4+i for i in sp]) for qp in itertools.permutations(range(2)) for sp in itertools.permutations(range(3)))/12
    rawh=torch.einsum('ra,rb,rc->rabc',atoms[:,0],atoms[:,1],atoms[:,2]);dictionary=sum(rawh.permute(0,*[1+i for i in perm]) for perm in itertools.permutations(range(3)))/6
    densegram=torch.einsum('rabc,tabc->rt',dictionary,dictionary)
    cross=torch.einsum('hoabcde,rcde->hroab',tensor,dictionary)
    densekernel=torch.einsum('hroab,htoab->hrt',cross,cross)
    factors=cross_factors(atoms,q1,k1,q2,k2,v,o)
    def rel(x,y):return float(((x-y).norm()/y.norm()).detach())
    ge=rel(atom_gram(atoms),densegram);ke=rel(query_output_gram(factors),densekernel)
    solved=torch.linalg.solve(densegram,cross.permute(1,0,2,3,4).flatten(1)).reshape(cross.shape[1],cross.shape[0],*cross.shape[2:]).permute(1,0,2,3,4)
    projected=torch.einsum('hroab,rcde->hoabcde',solved,dictionary)
    cap=capture(atoms,q1,k1,q2,k2,v,o);true=projected.square().sum();ce=rel(cap,true)
    gg=torch.autograd.grad(cap,atoms,retain_graph=True)[0];dg=torch.autograd.grad(true,atoms,retain_graph=True)[0];grad=rel(gg,dg)
    identity=rel((tensor-projected).square().sum(),tensor.square().sum()-cap)
    qq=torch.randn(5,2,dtype=dt);ss=torch.randn(5,3,dtype=dt)
    direct=torch.einsum('hoabcde,na,nb,nc,nd,ne->nho',projected,qq,qq,ss,ss,ss)
    ee=rel(execute(qq,ss,atoms,factors),direct)
    result=dict(atom_gram_error=ge,cross_gram_error=ke,capture_error=ce,gradient_error=grad,projection_identity_error=identity,execution_error=ee)
    assert max(result.values())<1e-10
    out=P/'SHARED_CUBIC_SOURCE_PROJECTION_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
