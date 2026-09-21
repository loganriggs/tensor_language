"""Rank-one approximation of pairs of three-mode CP terms in their 2D spans.

Grams describe each pair's unit vectors under a fixed separable metric.
No tensor with ambient input/output dimensions is materialized. This implements
one candidate DAG edit, not global circuit search or semantic identification.
"""
import torch

def fit_pair_cores(grams, amplitudes, restarts=8, steps=40, seed=261309):
    # grams: batch x 3 modes x 2 x 2; amplitudes: batch x 2.
    eigen, vectors = torch.linalg.eigh(grams)
    root = eigen.clamp_min(0).sqrt().unsqueeze(-1) * vectors.transpose(-1,-2)
    t = torch.einsum('bik,bjk,blk,bk->bijl',root[:,0],root[:,1],root[:,2],amplitudes)
    gen=torch.Generator(device=t.device).manual_seed(seed)
    shape=(len(t),restarts,2)
    v=torch.randn(shape,dtype=t.dtype,device=t.device,generator=gen)
    w=torch.randn(shape,dtype=t.dtype,device=t.device,generator=gen)
    def norm(x):return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    # Include both original products as starts; remaining starts are random.
    v[:,:2]=root[:,1].transpose(1,2);w[:,:2]=root[:,2].transpose(1,2)
    v=norm(v);w=norm(w)
    for _ in range(steps):
        u=norm(torch.einsum('bijk,brj,brk->bri',t,v,w))
        v=norm(torch.einsum('bijk,bri,brk->brj',t,u,w))
        w=norm(torch.einsum('bijk,bri,brj->brk',t,u,v))
    score=torch.einsum('bijk,bri,brj,brk->br',t,u,v,w)
    best=score.square().argmax(1); ids=torch.arange(len(t),device=t.device)
    factors=torch.stack([u[ids,best],v[ids,best],w[ids,best]],1)
    gain=score[ids,best]
    energy=t.square().sum((1,2,3))
    error=(energy-gain.square()).clamp_min(0)
    return dict(error=error,energy=energy,gain=gain,factors=factors,roots=root,core=t)

def self_test():
    dtype=torch.float64
    # Same input product, orthogonal output effects: exact cross-branch reuse.
    gram=torch.eye(2,dtype=dtype).repeat(2,3,1,1)
    gram[0,:2]=1
    result=fit_pair_cores(gram,torch.tensor([[1.,2.],[1.,2.]],dtype=dtype))
    assert result['error'][0]<1e-12
    # Three mutually orthogonal pairs: best rank-one keeps the stronger term.
    assert abs(float(result['error'][1])-1)<1e-12
    u,v,w=result['factors'].unbind(1)
    residual=result['core']-torch.einsum('b,bi,bj,bk->bijk',result['gain'],u,v,w)
    assert torch.allclose(residual.square().sum((1,2,3)),result['error'],atol=1e-12)
    return dict(exact_shared_error=float(result['error'][0]),orthogonal_error=float(result['error'][1]),explicit_replay=True)

if __name__=='__main__':print(self_test())
