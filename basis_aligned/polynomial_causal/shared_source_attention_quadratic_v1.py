"""Shared-source value constraints and the cross-source determinant channel.

Not a complete QK/V circuit: routing weights and normalized source states are
formal inputs here. The antisymmetric channel only vanishes for one source.
"""
import torch


def split_source_symmetry(tensor):
    # (..., head, source-feature, head, source-feature), symmetric under
    # exchanging whole (head,feature) slots. Swap only the two head indices.
    swapped=tensor.transpose(-4,-2)
    return (tensor+swapped)/2,(tensor-swapped)/2


def value_maps(o,v_current,v_base,mix,heads):
    width=o.shape[1]//heads
    assert v_current.shape==v_base.shape and o.shape[1]==heads*width
    return torch.stack([o[:,h*width:(h+1)*width]@torch.cat(((1-mix)*v_current[h*width:(h+1)*width],mix*v_base[h*width:(h+1)*width]),dim=1) for h in range(heads)])


def pullback(q,maps):
    return torch.einsum('hai,vab,kbj->vhikj',maps,q,maps)


def evaluate(t,z):return torch.einsum('nhi,vhikj,nkj->nv',z,t,z)


def symmetry_energies_low_rank(q,o,values,heads):
    """Exact norms without forming the (head×source)^2 lifted matrix.

    values stacks the per-head maps into the SAME source-feature coordinates.
    q is one residual quadratic; o maps concatenated value coordinates to it.
    """
    width=o.shape[1]//heads
    core=o.T@q@o;gram=values@values.T
    total=q.new_zeros(());partial_inner=q.new_zeros(())
    for h in range(heads):
        hs=slice(h*width,(h+1)*width);gh=gram[hs,hs]
        for k in range(h,heads):
            ks=slice(k*width,(k+1)*width);a=core[hs,ks];gk=gram[ks,ks]
            norm=((a@gk@a.T)*gh).sum()
            product=a@gram[ks,hs]
            trace_square=(product*product.T).sum()
            copies=1 if h==k else 2
            total+=copies*norm;partial_inner+=copies*trace_square
    return dict(total=total,symmetric=(total+partial_inner)/2,
                antisymmetric=(total-partial_inner)/2)


def controls():
    torch.manual_seed(184);dt=torch.float64
    heads,width,residual,source,outputs=3,2,6,4,5
    o=torch.randn(residual,heads*width,dtype=dt)
    vc=torch.randn(heads*width,source,dtype=dt);vb=torch.randn_like(vc)
    maps=value_maps(o,vc,vb,.37,heads)
    raw=torch.randn(outputs,residual,residual,dtype=dt);q=(raw+raw.transpose(-1,-2))/2
    t=pullback(q,maps);s,a=split_source_symmetry(t)
    gate=torch.randn(19,heads,dtype=dt);x=torch.randn(19,2*source,dtype=dt);z=gate[:,:,None]*x[:,None,:]
    physical=torch.einsum('hdi,nhi->nd',maps,z)
    direct=torch.einsum('ni,vij,nj->nv',physical,q,physical)
    scale=direct.norm()
    full_error=float((evaluate(t,z)-direct).norm()/scale)
    one_source_error=float((evaluate(s,z)-direct).norm()/scale)
    anti_zero=float(evaluate(a,z).norm()/scale)
    energy_split=abs(float((s.square().sum()+a.square().sum())/t.square().sum())-1)
    orthogonality=float(abs((s*a).sum())/t.square().sum())
    values=torch.cat((.63*vc,.37*vb),dim=1)
    lowrank=[symmetry_energies_low_rank(q[i],o,values,heads) for i in range(outputs)]
    lowrank_error=max(float(abs(v['total']/t[i].square().sum()-1)) for i,v in enumerate(lowrank))
    lowrank_error=max(lowrank_error,max(float(abs(v['antisymmetric']-a[i].square().sum())/t[i].square().sum()) for i,v in enumerate(lowrank)))
    # Multiple-source routing/value products sum to a matrix of rank>1.
    gate2=torch.randn_like(gate);x2=torch.randn_like(x);z2=z+gate2[:,:,None]*x2[:,None,:]
    multi=evaluate(t,z2);multi_replay=float((multi-evaluate(s,z2)-evaluate(a,z2)).norm()/multi.norm())
    anti_live=float(evaluate(a,z2).norm()/multi.norm())
    # Determinant counterexample and Cauchy-Binet source-pair factorization:
    # det(A X) = sum_{p<q} det(A[:,[p,q]]) det(X[[p,q],:]).
    routing=torch.tensor([[1.,0.],[0.,1.]],dtype=dt)
    values=torch.eye(2,dtype=dt)
    determinant=float(torch.linalg.det(routing@values))
    gates=torch.randn(2,4,dtype=dt);features=torch.randn(4,2,dtype=dt)
    contracted=gates@features;terms=[]
    for p in range(4):
        for k in range(p+1,4):
            terms.append(torch.linalg.det(gates[:,[p,k]])*torch.linalg.det(features[[p,k],:]))
    cauchy_binet_error=float(abs(torch.linalg.det(contracted)-sum(terms)))
    out=dict(native_like_value_mixing_pullback_relative_error=full_error,
             one_source_symmetric_projection_relative_error=one_source_error,
             one_source_antisymmetric_relative_value=anti_zero,
             energy_split_error=energy_split,subspace_orthogonality_error=orthogonality,
             low_rank_energy_relative_error=lowrank_error,
             multiple_source_split_replay_relative_error=multi_replay,
             multiple_source_antisymmetric_relative_value=anti_live,
             two_source_determinant_counterexample=determinant,
             source_pair_cauchy_binet_absolute_error=cauchy_binet_error)
    out['passed']=max(full_error,one_source_error,anti_zero,energy_split,orthogonality,lowrank_error,multi_replay,cauchy_binet_error)<=1e-10 and anti_live>.01 and determinant==1
    return out

if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);r=controls();print(json.dumps(r,indent=2))
    Path(__file__).with_name('SHARED_SOURCE_ATTENTION_QUADRATIC_V1_CONTROL.json').write_text(json.dumps(r,indent=2)+'\n');assert r['passed']
