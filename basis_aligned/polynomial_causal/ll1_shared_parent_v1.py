"""One explicit cross-LL1 shared-reader graph move, with exact coefficient scoring.

Select pair by largest principal cosine, then project into spans with a shared
parent. Diagonalize only the private core, leaving an arrowhead computation.
CPU only; frozen pilot candidates; not a general graph optimizer.
"""
import hashlib
import json
import time
from pathlib import Path
import torch
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


def shared_parent(parts):
    a, s, c = parts
    m, rank, width = a.shape
    bases = torch.linalg.qr(a.transpose(1, 2), mode='reduced').Q
    gram = (bases.permute(1,0,2).reshape(width,m*rank).T @
            bases.permute(1,0,2).reshape(width,m*rank)).reshape(m,rank,m,rank).permute(0,2,1,3)
    pairs = torch.triu_indices(m,m,1)
    cosines = torch.linalg.svdvals(gram[pairs[0],pairs[1]])[:,0]
    best = int(cosines.argmax())
    i,j = (int(pairs[k,best]) for k in (0,1))
    left, values, right = torch.linalg.svd(gram[i,j])
    frames = [bases[i]@left, bases[j]@right.T]
    parent = frames[0][:,0]+frames[1][:,0]
    parent /= parent.norm()
    groups=[]
    for group,frame in zip((i,j),frames):
        new_basis = torch.column_stack((parent,frame[:,1:]))
        contraction = new_basis.T@a[group].T
        core = (contraction*s[group])@contraction.T
        lam, rotation = torch.linalg.eigh(core[1:,1:])
        private = frame[:,1:]@rotation
        beta = rotation.T@core[1:,0]
        groups.append(dict(private=private,beta=beta,lam=lam,
                           alpha=core[0,0],writer=c[group],basis=new_basis,core=core))
    # CP is used for scoring, while the executable graph reuses reader values.
    aa=[parent]; bb=[parent]
    ww=[sum(g['alpha']*g['writer'] for g in groups)]
    for g in groups:
        aa.append(parent); bb.append(g['private']@g['beta']); ww.append(2*g['writer'])
        for k in range(rank-1):
            aa.append(g['private'][:,k]);bb.append(g['private'][:,k]);ww.append(g['lam'][k]*g['writer'])
    new_cp=(torch.stack(aa),torch.stack(bb),torch.stack(ww,1))
    old_cp=cp(a[[i,j]],s[[i,j]],c[[i,j]])
    delta=(torch.cat((new_cp[0],old_cp[0])),torch.cat((new_cp[1],old_cp[1])),
           torch.cat((new_cp[2],-old_cp[2]),1))
    return dict(pair=[i,j],cosine=float(values[0]),cosine_quantiles=torch.quantile(cosines,torch.tensor([0.,.5,.9,.99,1.])).tolist(),
                parent=parent,groups=groups,new_cp=new_cp,old_cp=old_cp,delta=delta)


def execute(move,x):
    u=x@move['parent']; result=x.new_zeros((len(x),move['groups'][0]['writer'].numel()))
    for g in move['groups']:
        z=x@g['private']
        scalar=g['alpha']*u.square()+2*u*(z@g['beta'])+z.square()@g['lam']
        result+=scalar[:,None]*g['writer']
    return result


@torch.no_grad()
def main():
    start=time.perf_counter();torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    receipt=json.loads((root/'MATCHED_SHARED_GROUPS_V1_RESULT.json').read_text())
    ck=next(k for k in receipt['binding'] if k.endswith('pytorch_model.bin'))
    sd=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
    native=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    rows=[]
    for label in ('spectral','native'):
        path=Path(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt')
        saved=torch.load(path,map_location='cpu',weights_only=True)
        parts=tuple(x.double() for x in saved['parts']);wh=saved['output_whitener'].double()
        target=(native[0],native[1],wh@native[2]);total=99245061353.47293
        move=shared_parent(parts)
        d2=float(inner(move['delta'],move['delta'])/total)
        change=float((inner(move['delta'],move['delta'])+2*inner(cp(*parts),move['delta'])-2*inner(target,move['delta']))/total)
        torch.manual_seed(2101);x=torch.randn(17,parts[0].shape[-1])
        aa,bb,ww=move['new_cp'];reference=((x@aa.T)*(x@bb.T))@ww.T
        replay=float((execute(move,x)-reference).norm()/reference.norm())
        compact_replay=[];orth=[]
        for g in move['groups']:
            new=g['basis']@g['core']@g['basis'].T
            parent=move['parent'];private=g['private'];partner=private@g['beta']
            arrow=g['alpha']*parent[:,None]*parent[None,:]+parent[:,None]*partner[None,:]+partner[:,None]*parent[None,:]+(private*g['lam'])@private.T
            compact_replay.append(float((new-arrow).norm()/new.norm()))
            orth.append(float((g['basis'].T@g['basis']-torch.eye(g['basis'].shape[1])).norm()))
        old_energy=float(inner(move['old_cp'],move['old_cp']))
        new_energy=float(inner(move['new_cp'],move['new_cp']))
        m,r,d=parts[0].shape;o=parts[2].shape[1]
        old_floats=2*(r*d+r+o)
        new_floats=(2*r-1)*d+2*(1+(r-1)+(r-1)+o)
        row=dict(label=label,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pair=move['pair'],
                 principal_cosine=move['cosine'],pairwise_cosine_quantiles=move['cosine_quantiles'],
                 squared_relative_global_change=d2,capture_loss=change,
                 executor_replay=replay,compact_replay=compact_replay,orthogonality=orth,
                 two_group_energy_ratio=new_energy/old_energy,
                 independent_group_floats=old_floats,shared_graph_floats=new_floats,
                 floats_saved=old_floats-new_floats,old_variable_products=2*r,new_variable_products=2*r+1)
        row['pred_a']=max([replay]+compact_replay+orth)<1e-9
        row['pred_b']=d2<=1e-5 and change<=1e-4
        row['pred_c']=old_floats>new_floats and row['new_variable_products']-row['old_variable_products']<=2
        rows.append(row);print(json.dumps(row),flush=True)
    result=dict(rows=rows,seconds=time.perf_counter()-start,scope='One principal-overlap-selected shared-parent move per unconverged native LL1 start. Weight-only; no joint refitting or semantic/behavioral claim. Coefficient constants, reader matrix and writers counted; shared whole-model background unchanged.')
    (root/'LL1_SHARED_PARENT_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
