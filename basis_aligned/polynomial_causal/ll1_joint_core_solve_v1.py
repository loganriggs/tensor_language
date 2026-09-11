"""Matrix-free conditional optimum of all symmetric group cores together.

Input spans are orthonormal, output writers unit norm. Then whole-group tensor
energy is ||core||_F^2. Positive ridge makes the core problem strictly convex.
"""
import copy
import time
import torch
from ll1_joint_parent_graph_v2 import factors


def coordinates(graph):
    a,s,c=factors(graph);bases=[]
    for g in graph['groups']:
        rr=graph['readers'][g['parent_ids']]
        shared=torch.linalg.qr(rr.T,mode='reduced').Q
        bases.append(torch.cat((shared,g['private'].T),dim=1))
    bases=torch.stack(bases)
    overlap=a@bases
    norm=c.norm(dim=1)
    cores=((overlap.transpose(1,2)*s[:,None])@overlap)*norm[:,None,None]
    return bases,c/norm[:,None],cores


class System:
    def __init__(self,bases,writers,target,penalty=.01):
        self.penalty=penalty
        self.cross=torch.einsum('gdi,hdj->ghij',bases,bases)
        self.writer_gram=writers@writers.T
        m,d,r=bases.shape
        flat=bases.permute(1,0,2).reshape(d,m*r)
        left=(target[0]@flat).reshape(-1,m,r)
        right=(target[1]@flat).reshape(-1,m,r)
        weights=writers@target[2]
        rhs=[]
        for g in range(m):
            raw=(left[:,g].T*weights[g][None])@right[:,g]
            rhs.append((raw+raw.T)/2)
        self.rhs=torch.stack(rhs)
        self.condition_bound=(m+penalty)/penalty

    def apply(self,cores):
        first=torch.einsum('ghij,hjk->ghik',self.cross,cores)
        result=torch.einsum('ghik,ghlk,gh->gil',first,self.cross,self.writer_gram)
        return result+self.penalty*cores


def solve(system,initial,tolerance=1e-8,maxiter=2000,seconds=120):
    start=time.perf_counter();value=initial.clone()
    residual=system.rhs-system.apply(value);direction=residual.clone()
    squared=residual.square().sum();rhsnorm=system.rhs.norm()
    history=[];iterations=0
    for i in range(maxiter):
        if float(squared.sqrt()/rhsnorm)<=tolerance or time.perf_counter()-start>=seconds:break
        mapped=system.apply(direction);curvature=(direction*mapped).sum()
        if not bool(curvature>0):raise FloatingPointError('Nonpositive CG curvature')
        alpha=squared/curvature
        value+=alpha*direction
        residual-=alpha*mapped
        next_squared=residual.square().sum()
        direction=residual+(next_squared/squared)*direction
        squared=next_squared;iterations=i+1
        if iterations%25==0:history.append(dict(iteration=iterations,residual=float(squared.sqrt()/rhsnorm)))
    # True residual, not just the recursively updated stopping quantity.
    true=float((system.rhs-system.apply(value)).norm()/rhsnorm)
    return value,dict(iterations=iterations,seconds=time.perf_counter()-start,
                      normal_residual=true,converged=true<=tolerance,
                      condition_bound=system.condition_bound,history=history)


def install(graph,cores,writers):
    updated=copy.deepcopy(graph)
    for g,h,c in zip(updated['groups'],cores,writers):
        k=len(g['parent_ids']);rr=updated['readers'][g['parent_ids']]
        change=torch.linalg.qr(rr.T,mode='reduced').R
        inv=torch.linalg.inv(change)
        lam,rotation=torch.linalg.eigh(h[k:,k:])
        g['private']=rotation.T@g['private']
        shared=inv@h[:k,:k]@inv.T
        g['cross']=2*inv@h[:k,k:]@rotation
        coeff=[]
        for i in range(k):
            for j in range(i,k):coeff.append(shared[i,j]*(1 if i==j else 2))
        g['shared_coeff']=torch.stack(coeff) if coeff else h.new_zeros(0)
        g['lam']=lam;g['writer']=c
    return updated


def control():
    from chunked_bilinear_coefficient_v1 import dense
    from ll1_joint_parent_graph_v2 import build,execute
    from symmetric_ll1_objective_v1 import cp
    torch.manual_seed(2801)
    a=torch.randn(3,2,5);s=torch.randn(3,2);c=torch.randn(3,4)
    graph=build(a,s,c,torch.zeros(0,5),[])
    bases,writers,initial=coordinates(graph)
    target=tuple(torch.randn(*shape) for shape in ((7,5),(7,5),(4,7)))
    system=System(bases,writers,target)
    fitted,stats=solve(system,initial,tolerance=1e-12)
    eye=torch.eye(initial.numel()).reshape(-1,*initial.shape)
    matrix=torch.stack([system.apply(v).flatten() for v in eye],dim=1)
    exact=torch.linalg.solve(matrix,system.rhs.flatten()).reshape_as(fitted)
    dense_error=float((fitted-exact).norm()/exact.norm())
    updated=install(graph,fitted,writers)
    tensor=torch.einsum('go,gdi,gij,gej->ode',writers,bases,fitted,bases)
    from_graph=dense(*cp(*factors(updated)))
    coefficient_error=float((tensor-from_graph).norm()/tensor.norm())
    target_tensor=dense(*target)
    objective=(target_tensor-tensor).square().sum()+.01*fitted.square().sum()
    identity=target_tensor.square().sum()-2*(system.rhs*fitted).sum()+(fitted*system.apply(fitted)).sum()
    objective_error=float(abs(objective-identity)/objective)
    x=torch.randn(9,5);reference=torch.einsum('ode,nd,ne->no',tensor,x,x)
    replay=float((execute(updated,x)-reference).norm()/reference.norm())
    result=dict(**stats,dense_solve_error=dense_error,coefficient_error=coefficient_error,
                objective_error=objective_error,executor_error=replay,
                held=max(dense_error,coefficient_error,objective_error,replay,stats['normal_residual'])<=1e-9)
    assert result['held'],result
    return result
