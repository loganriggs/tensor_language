"""Joint DAG readers/writers with all interaction cores conditionally eliminated.

Only QR/normalization maps use autograd. Full tensor loss/gradients use implicit
CP contractions, and the core solve uses a matrix-free SPD normal operator.
"""
import copy
import numpy as np
import torch
from ll1_joint_core_solve_v1 import coordinates,System,solve,install
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import value_gradient


class Objective:
    def __init__(self,target,graph,total,penalty=.01,inner_tolerance=1e-11):
        self.target,self.total,self.penalty=target,total,penalty
        self.inner_tolerance=inner_tolerance
        self.template=graph;self.device=target[0].device
        parts=[graph['readers'],torch.cat([g['private'] for g in graph['groups']]),
               torch.stack([g['writer'] for g in graph['groups']])]
        parts=[v.detach().to(self.device,dtype=torch.float64) for v in parts]
        self.shapes=[v.shape for v in parts];self.sizes=[v.numel() for v in parts]
        self.scales=[max(float(v.norm()),1.) for v in parts]
        self.initial=torch.cat([(v/scale).flatten().cpu() for v,scale in zip(parts,self.scales)]).numpy()
        self.parents=[g['parent_ids'].to(self.device) for g in graph['groups']]
        self.private_sizes=[len(g['private']) for g in graph['groups']]
        self.core_start=coordinates(graph)[2].detach().to(self.device)
        self.last_point=None;self.last_value=None

    def unpack(self,point):
        blocks=point.split(self.sizes)
        return tuple(v.reshape(shape)*scale for v,shape,scale in zip(blocks,self.shapes,self.scales))

    def maps(self,point):
        shared,private,c=self.unpack(point)
        shared=shared/shared.norm(dim=1,keepdim=True)
        writers=c/c.norm(dim=1,keepdim=True)
        bases=[];privates=[]
        for ids,v in zip(self.parents,private.split(self.private_sizes)):
            u=torch.linalg.qr(shared[ids].T,mode='reduced').Q
            residual=v.T-u@(u.T@v.T)
            vv=torch.linalg.qr(residual,mode='reduced').Q
            bases.append(torch.cat((u,vv),dim=1));privates.append(vv.T)
        return torch.stack(bases),writers,shared,privates

    def evaluate(self,point):
        if self.last_point is not None and np.array_equal(point,self.last_point):
            return self.last_value
        leaf=torch.as_tensor(point,dtype=torch.float64,device=self.device).detach().requires_grad_()
        bases,writers,_,_=self.maps(leaf)
        with torch.no_grad():
            system=System(bases.detach(),writers.detach(),self.target,self.penalty)
            cores,stats=solve(system,self.core_start,tolerance=self.inner_tolerance)
            if not stats['converged']:
                raise FloatingPointError(f'Inner conditional solve unconverged: {stats}')
            values,vectors=torch.linalg.eigh(cores)
            a=(bases.detach()@vectors).transpose(1,2)
            loss,(ga,gb,gw),details=value_gradient(*self.target,*cp(a,values,writers.detach()),self.total)
            energy=cores.square().sum()/self.total
            loss+=self.penalty*energy
            grad_b=(ga+gb).reshape_as(a).transpose(1,2)@vectors.transpose(1,2)
            grad_c=(gw.reshape(writers.shape[1],len(writers),a.shape[1]).permute(1,0,2)*values[:,None]).sum(2)
            reduced=1-(cores*system.rhs).sum()/self.total
            details.update(group_energy=float(energy),inner=stats,reduced_identity_error=abs(float(loss-reduced)))
        # On orthonormal bases/unit writers the penalty is independent of these
        # coordinates when the optimal cores are held fixed (envelope theorem).
        gradient=torch.autograd.grad((bases,writers),leaf,(grad_b,grad_c))[0]
        self.last=details;self.cores=cores.detach();self.core_start=self.cores
        self.last_point=np.array(point,copy=True)
        self.last_value=(float(loss),gradient.detach().cpu().numpy())
        return self.last_value

    @torch.no_grad()
    def physical(self,point):
        # evaluate() must correspond to this exact point before using its cores.
        if self.last_point is None or not np.array_equal(point,self.last_point):self.evaluate(point)
        leaf=torch.as_tensor(point,dtype=torch.float64,device=self.device)
        _,writers,readers,privates=self.maps(leaf)
        graph=copy.deepcopy(self.template)
        graph['readers']=readers
        for g,v,c in zip(graph['groups'],privates,writers):g.update(private=v,writer=c)
        return install(graph,self.cores,writers)
