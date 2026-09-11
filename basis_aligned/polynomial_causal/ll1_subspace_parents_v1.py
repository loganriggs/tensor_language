"""Propose shared mixed linear parents from LL1 input-subspace incidence."""
import json
from pathlib import Path
import torch
from shared_square_ll1_v1 import canonicalize
from structured_branch_amplitudes_v1 import inner


def describe(u,a,s,c,bases,total):
    membership=torch.einsum('gdr,d->gr',bases,u).square().sum(1)
    dots=torch.einsum('grd,d->gr',a,u)
    qu=torch.einsum('gr,grd->gd',s*dots,a)
    alpha=(qu*u).sum(1);private=qu-alpha[:,None]*u
    cnorm=c.square().sum(1)
    group_energy=cnorm*s.square().sum(1)  # canonical orthonormal eigenreaders
    mixed_individual=2*cnorm*private.square().sum(1)
    ids=((membership>=.95)&(mixed_individual>=.01*group_energy)).nonzero()[:,0]
    if len(ids)<2:return None
    writers=c[ids];partners=private[ids]
    mixed=2*((writers@writers.T)*(partners@partners.T)).sum()
    square=(alpha[ids,None]*writers).sum(0).square().sum()
    return dict(consumers=ids,mixed=mixed,square=square,private=partners,alpha=alpha[ids],
                writers=writers,membership=membership[ids],mixed_fractions=(mixed_individual/group_energy)[ids],
                valid=bool(mixed/(mixed+square)>=.1 and mixed/total>=1e-4))


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;rows=[];saved={};total=99245061353.47293
    for label in ('spectral','native'):
        raw=torch.load(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt',weights_only=True,map_location='cpu')['parts']
        a,s,c=canonicalize(*(v.double() for v in raw));m,r,d=a.shape;bases=a.transpose(1,2)
        cross=(a.flatten(0,1)@a.flatten(0,1).T).reshape(m,r,m,r).permute(0,2,1,3)
        pairs=torch.triu_indices(m,m,1);left,values,right=torch.linalg.svd(cross[pairs[0],pairs[1]])
        selected=(values>=.95).nonzero();candidates=[]
        for pair,k in selected.tolist():
            i,j=int(pairs[0,pair]),int(pairs[1,pair])
            u=bases[i]@left[pair,:,k]+bases[j]@right[pair,k]
            u/=u.norm();info=describe(u,a,s,c,bases,total)
            if info is not None and info['valid']:candidates.append((u,info))
        clusters=[]
        for index in sorted(range(len(candidates)),key=lambda i:float(candidates[i][1]['mixed']),reverse=True):
            u=candidates[index][0]
            cluster=next((cl for cl in clusters if all(abs(float(u@candidates[j][0]))>=.99 for j in cl)),None)
            if cluster is None:clusters.append([index])
            else:cluster.append(index)
        nodes=[];vectors=[];errors=[]
        for cluster in clusters:
            v=torch.stack([candidates[i][0] for i in cluster]);weights=torch.stack([candidates[i][1]['mixed'] for i in cluster])
            u=torch.linalg.svd(v*(weights/weights.sum()).sqrt()[:,None],full_matrices=False).Vh[0]
            info=describe(u,a,s,c,bases,total)
            if info is None or not info['valid']:continue
            ids=info['consumers'];partners=info['private'];writers=info['writers'];alpha=info['alpha']
            bank=(u.expand(len(ids),-1),2*partners,writers.T)
            replay=abs(float(inner(bank,bank)-info['mixed']))/float(info['mixed'])
            torch.manual_seed(2501);x=torch.randn(7,d);xp=x-(x@u)[:,None]*u
            original=(torch.einsum('nd,gkd->ngk',x,a[ids]).square()*s[ids][None]).sum(-1)@writers
            projected=(torch.einsum('nd,gkd->ngk',xp,a[ids]).square()*s[ids][None]).sum(-1)@writers
            scalar=x@u;component=(scalar[:,None].square()*alpha[None]+2*scalar[:,None]*(x@partners.T))@writers
            executor=float((original-projected-component).norm()/component.norm())
            errors.append(max(replay,executor));vectors.append(u)
            nodes.append(dict(consumers=ids.tolist(),mixed_energy_over_native=float(info['mixed']/total),
                              mixed_fraction_of_parent_energy=float(info['mixed']/(info['mixed']+info['square'])),
                              minimum_squared_membership=float(info['membership'].min()),
                              minimum_individual_mixed_group_fraction=float(info['mixed_fractions'].min()),
                              cluster_proposals=len(cluster),cp_energy_replay=replay,executor_replay=executor))
        degrees=[sum(g in n['consumers'] for n in nodes) for g in range(m)]
        row=dict(label=label,principal_pairs_directions_above_095=len(selected),eligible_pair_proposals=len(candidates),
                 retained_parents=len(nodes),parents_with_three_consumers=sum(len(n['consumers'])>=3 for n in nodes),
                 max_group_parent_degree=max(degrees),groups_with_multiple_parents=sum(n>=2 for n in degrees),
                 pred_a=max(errors,default=0.)<=1e-9,pred_b=sum(len(n['consumers'])>=3 for n in nodes)>=4,
                 pred_c=max(degrees)>=2,nodes=nodes)
        rows.append(row);saved[label]=dict(readers=torch.stack(vectors) if vectors else torch.zeros(0,d),nodes=nodes)
        print(json.dumps({k:v for k,v in row.items() if k!='nodes'}),flush=True)
    artifact=Path('/dev/shm/bilin18_ll1_subspace_parents_v1.pt');torch.save(saved,artifact)
    result=dict(rows=rows,cache=str(artifact),scope='Weight-only input-subspace incidence proposals in unconverged LL1 pilots. Candidate parent components overlap; summing them without inclusion/exclusion or a shared core would double-count interactions. Not a joint executable graph or behavioral circuit claim.')
    (root/'LL1_SUBSPACE_PARENTS_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
