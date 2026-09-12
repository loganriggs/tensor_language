"""Sparse mixed-edge execution after exact producer-reader folding.

Port-local removal holds native background and denominator fixed. It is not
an upstream model intervention. Symmetric off-diagonal features need sqrt(2).
"""
import torch


def execute(values, edges, writers, denominator, keep=None):
    i,j=edges
    features=values[:,i]*values[:,j]*torch.where(i==j,1.,2.**.5)
    if keep is not None:features=features*keep
    return features@writers.T/denominator[:,None]


def incident(edges,nodes):
    nodes=torch.as_tensor(nodes,device=edges.device)
    return torch.isin(edges[0],nodes)|torch.isin(edges[1],nodes)


def control():
    torch.set_default_dtype(torch.float64);g=torch.Generator().manual_seed(120467)
    x=torch.randn(23,6,generator=g)
    l0,r0=[torch.randn(9,6,generator=g) for _ in range(2)]
    d0=torch.randn(4,9,generator=g)
    l1,r1=[torch.randn(7,4,generator=g) for _ in range(2)]
    d1=torch.randn(5,7,generator=g)
    q=torch.linalg.qr(torch.randn(4,4,generator=g)).Q
    root=torch.randn(4,4,generator=g);h=root@root.T+torch.eye(4)
    he,hv=torch.linalg.eigh(h);hs=(hv*he.sqrt())@hv.T;hi=(hv*he.rsqrt())@hv.T
    scale=1.0234375
    hidden=(x@l0.T)*(x@r0.T);p=scale*hidden@d0.T
    folded=scale*d0.T@hi@q
    z=hidden@folded;direct=p@hi@q
    edges=torch.triu_indices(4,4)
    from sparse_frame_function_inner_v1 import coefficients
    writers=coefficients(q,l1@hs,r1@hs,d1,edges)
    den=torch.randn(23,4,generator=g).square().mean(-1)+torch.finfo(torch.float32).eps
    graph=execute(z,edges,writers,den)
    native=((p@l1.T)*(p@r1.T))@d1.T/den[:,None]
    a,b=incident(edges,[0]),incident(edges,[1])
    removed_a=execute(z,edges,writers,den,a)
    removed_b=execute(z,edges,writers,den,b)
    joint=execute(z,edges,writers,den,a|b)
    overlap=execute(z,edges,writers,den,a&b)
    errors=dict(reader=float((z-direct).norm()/direct.norm()),
                full_graph=float((graph-native).norm()/native.norm()),
                union=float((joint-removed_a-removed_b+overlap).norm()/joint.norm()))
    return dict(errors=errors,shared_edge_effect_norm=float(overlap.norm()),
                passed=max(errors.values())<1e-12 and float(overlap.norm())>1e-6,
                scope='Exact compiled quartic and overlapping node removal, fixed local denominator; no native behavioral evidence.')


if __name__=='__main__':
    import json
    print(json.dumps(control()))
