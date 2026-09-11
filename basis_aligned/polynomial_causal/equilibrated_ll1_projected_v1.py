"""Exact LL1 output elimination with group-Gram diagonal equilibration."""
import torch
from symmetric_ll1_projected_v1 import Objective as BaseObjective
from symmetric_ll1_objective_v1 import value_gradient


def output_solve(target,a,s,penalty=.01):
    m,r,d=a.shape;flat=a.reshape(m*r,d)
    gram=torch.einsum('irjs,ir,js->ij',(flat@flat.T).square().reshape(m,r,m,r),s,s)
    norms=gram.diag().sqrt()
    if not bool(torch.isfinite(norms).all() and (norms>0).all()):
        raise FloatingPointError('Zero or invalid group quadratic norm; no silent ridge repair.')
    left,right,writers=target
    projections=((left@flat.T)*(right@flat.T)).reshape(len(left),m,r)
    rhs=writers@(projections*s[None]).sum(2)
    system=gram/norms[:,None]/norms[None,:]+penalty*torch.eye(m,device=a.device,dtype=a.dtype)
    scaled_rhs=rhs.T/norms[:,None]
    scaled_c=torch.linalg.solve(system,scaled_rhs)
    c=scaled_c/norms[:,None]
    values=torch.linalg.eigvalsh(system)
    diagnostics=dict(output_normal_residual=float((system@scaled_c-scaled_rhs).norm()/scaled_rhs.norm()),
                     output_system_condition=float(values[-1]/values[0]),
                     output_condition_bound=(m+penalty)/penalty)
    return c,gram,rhs,diagnostics


class Objective(BaseObjective):
    def evaluate(self,point):
        a,s=self.unpack(point);an=a.norm(dim=-1,keepdim=True);sn=s.norm(dim=-1,keepdim=True)
        aa,ss=a/an,s/sn
        with torch.no_grad():
            c,gram,rhs,diagnostics=output_solve(self.target,aa,ss,self.penalty)
            loss,(ga,gs,gc),details=value_gradient(self.target,aa,ss,c,self.total,self.penalty)
            details.update(diagnostics)
            details['reduced_identity_error']=abs(float(loss-(1-(rhs.T*c).sum()/self.total)))
            ga=(ga-aa*(aa*ga).sum(-1,keepdim=True))/an
            gs=(gs-ss*(ss*gs).sum(-1,keepdim=True))/sn
            gradient=torch.cat([(g*scale).flatten().cpu() for g,scale in zip((ga,gs),self.scales)]).numpy()
        self.last=details;self.writer=c
        return float(loss),gradient
