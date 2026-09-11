"""Joint shared-parent/private-frame refinement in a fixed union input space."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from ll1_shared_parent_v1 import shared_parent, execute
from ll1_shared_parent_refit_v1 import core_refit
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


class Objective:
    def __init__(self,union,move,residuals,penalty=.01):
        self.union=union;self.residuals=residuals;self.penalty=penalty
        writers=torch.stack([g['writer'] for g in move['groups']])
        self.writer_gram=writers@writers.T
        initial=[union.T@move['parent'],torch.stack([union.T@g['private'] for g in move['groups']])]
        # Initial private frame already diagonalizes its private core.
        cores=[]
        for g in move['groups']:
            core=torch.zeros(16,16);core[0,0]=g['alpha'];core[0,1:]=g['beta'];core[1:,0]=g['beta'];core[1:,1:]=torch.diag(g['lam'])
            cores.append(core)
        initial.append(torch.stack(cores))
        self.shapes=[v.shape for v in initial];self.sizes=[v.numel() for v in initial]
        self.scales=[float(v.norm()) for v in initial]
        self.initial=torch.cat([(v/scale).flatten() for v,scale in zip(initial,self.scales)]).numpy()
        self.total=float(sum((c.square().sum()*self.writer_gram[i,i]) for i,c in enumerate(cores)))

    def physical(self,point):
        vals=point.split(self.sizes)
        parent,private,core=[v.reshape(shape)*scale for v,shape,scale in zip(vals,self.shapes,self.scales)]
        parent=parent/parent.norm()
        private=private-parent[None,:,None]*(parent[None,None,:]@private)
        private=torch.linalg.qr(private,mode='reduced').Q
        bases=torch.cat((parent.expand(2,-1)[:,:,None],private),dim=2)
        core=(core+core.transpose(1,2))/2
        matrices=bases@core@bases.transpose(1,2)
        return parent,bases,core,matrices

    def evaluate(self,point):
        packed=torch.tensor(point,requires_grad=True)
        _,_,_,matrices=self.physical(packed)
        gram=matrices.flatten(1)@matrices.flatten(1).T
        value=((gram*self.writer_gram).sum()-2*(matrices*self.residuals).sum()+self.penalty*(gram.diag()*self.writer_gram.diag()).sum())/self.total
        gradient=torch.autograd.grad(value,packed)[0]
        return float(value.detach()),gradient.detach().numpy()


def compact_residuals(target,parts,move,union):
    a,s,c=parts;indices=[i for i in range(len(a)) if i not in move['pair']]
    other=cp(a[indices],s[indices],c[indices]);results=[]
    l=target[0]@union;r=target[1]@union;ol=other[0]@union;orr=other[1]@union
    for g in move['groups']:
        e=(l.T*(g['writer']@target[2]))@r-(ol.T*(g['writer']@other[2]))@orr
        results.append((e+e.T)/2)
    return torch.stack(results)


def export(objective,point,move):
    with torch.no_grad():
        parent,bases,cores,_=objective.physical(torch.tensor(point))
        move['parent']=objective.union@parent
        aa=[];ww=[]
        for g,b,core in zip(move['groups'],bases,cores):
            fullbasis=objective.union@b
            lam,rot=torch.linalg.eigh(core[1:,1:])
            g.update(basis=fullbasis,core=core,alpha=core[0,0],beta=rot.T@core[1:,0],private=fullbasis[:,1:]@rot,lam=lam)
            val,vec=torch.linalg.eigh(core);aa.append((fullbasis@vec).T);ww.append(g['writer'][:,None]*val)
        a=torch.cat(aa);new=(a,a,torch.cat(ww,1));old=move['old_cp']
        delta=(torch.cat((new[0],old[0])),torch.cat((new[1],old[1])),torch.cat((new[2],-old[2]),1))
        return new,delta


def main():
    start=time.perf_counter();torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    pilot=json.loads((root/'MATCHED_SHARED_GROUPS_V1_RESULT.json').read_text())
    prior=json.loads((root/'LL1_SHARED_PARENT_V1_AUDIT.json').read_text())
    ck=next(k for k in pilot['binding'] if k.endswith('pytorch_model.bin'))
    sd=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    total=99245061353.47293;eta=.01;rows=[]
    for label in ('spectral','native'):
        saved=torch.load(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt',map_location='cpu',weights_only=True)
        parts=tuple(x.double() for x in saved['parts']);target=(l,r,saved['output_whitener'].double()@d)
        move=shared_parent(parts);union=torch.linalg.qr(parts[0][move['pair']].reshape(-1,1152).T,mode='reduced').Q
        _,initial_delta,_=core_refit(target,parts,move,eta)
        initial_energy=sum(g['writer'].square().sum()*g['core'].square().sum() for g in move['groups'])
        objective=Objective(union,move,compact_residuals(target,parts,move,union),eta)
        initial_value,gradient=objective.evaluate(objective.initial)
        rng=np.random.default_rng(2102);direction=rng.normal(size=len(gradient));direction/=np.linalg.norm(direction)
        eps=1e-6
        fd=(objective.evaluate(objective.initial+eps*direction)[0]-objective.evaluate(objective.initial-eps*direction)[0])/(2*eps)
        fd_error=abs(fd-gradient@direction)/max(1.,abs(fd),abs(gradient@direction))
        result=minimize(objective.evaluate,objective.initial,jac=True,method='L-BFGS-B',options=dict(maxiter=500,maxcor=20,maxls=30,ftol=0.,gtol=1e-9))
        new,delta=export(objective,result.x,move)
        def change(diff):
            return (inner(diff,diff)+2*inner(cp(*parts),diff)-2*inner(target,diff))/total
        initial_change=float(change(initial_delta));final_change=float(change(delta))
        final_energy=sum(g['writer'].square().sum()*g['core'].square().sum() for g in move['groups'])
        full_gain=initial_change-final_change+float(eta*(initial_energy-final_energy)/total)
        reduced_gain=(initial_value-result.fun)*objective.total/total
        replay=abs(full_gain-reduced_gain)
        torch.manual_seed(2101);x=torch.randn(17,1152)
        reference=((x@new[0].T)*(x@new[1].T))@new[2].T
        executor=float((execute(move,x)-reference).norm()/reference.norm())
        projection=next(r['capture_loss'] for r in prior['rows'] if r['label']==label)
        row=dict(label=label,pair=move['pair'],capture_loss=final_change,projection_capture_loss=projection,
                 fraction_projection_loss_removed=1-final_change/projection,penalized_gain=full_gain,
                 reduced_full_gain_replay=replay,finite_difference_error=float(fd_error),executor_replay=executor,
                 iterations=result.nit,evaluations=result.nfev,termination=result.message,gradient_inf=float(np.max(np.abs(result.jac))),
                 squared_relative_global_change=float(inner(delta,delta)/total),
                 pred_a=max(replay,float(fd_error),executor)<1e-6 and replay<1e-9 and executor<1e-9,
                 pred_b=bool(final_change<=projection*(.75 if label=='spectral' else 1)),
                 pred_c=True,floats_saved=1122,scope='Fixed-union shared-parent graph optimization; no global or whole-model convergence guarantee.')
        rows.append(row);print(json.dumps(row),flush=True)
        torch.save(dict(parent=move['parent'],groups=move['groups'],pair=move['pair']),f'/dev/shm/bilin18_ll1_shared_parent_optimize_v1_{label}.pt')
    (root/'LL1_SHARED_PARENT_OPTIMIZE_V1_AUDIT.json').write_text(json.dumps(dict(rows=rows,seconds=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':main()
