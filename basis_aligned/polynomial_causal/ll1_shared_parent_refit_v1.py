"""Exact conditional two-core refit after a shared-parent graph move (CPU)."""
import json
import time
from pathlib import Path
import torch
from ll1_shared_parent_v1 import shared_parent, execute
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


def symmetric_basis(rank):
    basis=[]
    for i in range(rank):
        for j in range(i,rank):
            matrix=torch.zeros(rank,rank)
            if i==j:matrix[i,j]=1
            else:matrix[i,j]=matrix[j,i]=2**-.5
            basis.append(matrix)
    return torch.stack(basis)


def core_refit(target, parts, move, penalty=.01):
    a,s,c=parts;rank=a.shape[1];basis=symmetric_basis(rank)
    indices=[i for i in range(len(a)) if i not in move['pair']]
    other=cp(a[indices],s[indices],c[indices])
    gram=[];rhs=[]
    for g in move['groups']:
        b=g['basis'];writer=g['writer']
        left=target[0]@b;right=target[1]@b
        residual=(left.T*(writer@target[2]))@right
        left=other[0]@b;right=other[1]@b
        residual-=(left.T*(writer@other[2]))@right
        rhs.append(torch.einsum('aij,ij->a',basis,residual))
        row=[]
        for h in move['groups']:
            cross=b.T@h['basis']
            mapped=cross[None]@basis@cross.T[None]
            row.append((writer@h['writer'])*(basis.flatten(1)@mapped.flatten(1).T))
        gram.append(torch.cat(row,dim=1))
    gram=torch.cat(gram,dim=0);rhs=torch.cat(rhs)
    energy_diag=torch.cat([g['writer'].square().sum().expand(len(basis)) for g in move['groups']])
    system=gram+penalty*torch.diag(energy_diag)
    coeff=torch.linalg.solve(system,rhs)
    normal=float((system@coeff-rhs).norm()/rhs.norm())
    for g,values in zip(move['groups'],coeff.reshape(2,-1)):
        core=torch.einsum('a,aij->ij',values,basis)
        lam,rot=torch.linalg.eigh(core[1:,1:])
        g.update(core=core,alpha=core[0,0],beta=rot.T@core[1:,0],
                 private=g['basis'][:,1:]@rot,lam=lam)
    # Isometric signed-square expansion of the updated cores for exact scoring.
    aa=[];ww=[]
    for g in move['groups']:
        lam,vec=torch.linalg.eigh(g['core'])
        aa.append((g['basis']@vec).T);ww.append(g['writer'][:,None]*lam)
    x=torch.cat(aa);new=(x,x,torch.cat(ww,1));old=move['old_cp']
    delta=(torch.cat((new[0],old[0])),torch.cat((new[1],old[1])),torch.cat((new[2],-old[2]),1))
    return new,delta,dict(normal_residual=normal,condition=float(torch.linalg.cond(system)))


def group_energy(move):
    return sum(g['writer'].square().sum()*g['core'].square().sum() for g in move['groups'])


@torch.no_grad()
def main():
    start=time.perf_counter();torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    original=json.loads((root/'MATCHED_SHARED_GROUPS_V1_RESULT.json').read_text())
    ck=next(k for k in original['binding'] if k.endswith('pytorch_model.bin'))
    sd=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    total=99245061353.47293;eta=.01;rows=[];parents=[]
    for label in ('spectral','native'):
        saved=torch.load(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt',map_location='cpu',weights_only=True)
        parts=tuple(x.double() for x in saved['parts']);target=(l,r,saved['output_whitener'].double()@d)
        move=shared_parent(parts);full=cp(*parts);parents.append(move['parent'])
        pre_energy=group_energy(move)
        def residual_change(delta):
            return (inner(delta,delta)+2*inner(full,delta)-2*inner(target,delta))/total
        pre_change=residual_change(move['delta'])
        new,delta,stats=core_refit(target,parts,move,eta)
        post_change=residual_change(delta)
        gain=pre_change-post_change+eta*(pre_energy-group_energy(move))/total
        torch.manual_seed(2101);x=torch.randn(17,parts[0].shape[-1])
        reference=((x@new[0].T)*(x@new[1].T))@new[2].T
        replay=float((execute(move,x)-reference).norm()/reference.norm())
        row=dict(label=label,pair=move['pair'],**stats,executor_replay=replay,
                 projection_capture_loss=float(pre_change),refitted_capture_loss=float(post_change),
                 penalized_objective_gain=float(gain),squared_relative_global_change=float(inner(delta,delta)/total),
                 pred_a=max(stats['normal_residual'],replay)<1e-9,
                 pred_b=float(gain)>=1e-6,pred_c=float(post_change)<=1e-4)
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(rows=rows,selected_parent_cross_start_abs_cosine=float(abs(parents[0]@parents[1])),
                seconds=time.perf_counter()-start,scope='Exact conditional joint core solve at fixed graph spans and writers. No convergence of graph topology/readers/whole fit or behavioral claim; original projection-only misses preserved.')
    (root/'LL1_SHARED_PARENT_REFIT_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
