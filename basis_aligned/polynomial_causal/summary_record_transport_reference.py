"""Affine incidence operator and norm-bounded summary transport obstruction."""
import torch


def contrasts(n,dtype=torch.float64):
    c=torch.zeros(n,n-1,dtype=dtype)
    for j in range(n-1):
        scale=((j+1)*(j+2))**.5;c[:j+1,j]=1/scale;c[j+1,j]=-(j+1)/scale
    return c


def operator(program):
    layer=program.background.layers[0];device=program.folded.device;dtype=program.folded.dtype
    e=layer.norm(program.background.embed(torch.arange(24,device=device)))
    query=layer.norm(program.background.embed(torch.arange(25,29,device=device)))
    def rotate(z,pos):
        a,b=z.chunk(2,-1);return z*layer.rotary.cos_cached[:,pos]+torch.cat((-b,a),-1)*layer.rotary.sin_cached[:,pos]
    def key(name):
        z=getattr(layer,name)(e).reshape(24,1,4,32).expand(-1,48,-1,-1)
        return rotate(z,torch.arange(48,device=device))
    def q(name):return rotate(getattr(layer,name)(query).reshape(4,1,4,32),torch.tensor([50],device=device))[:,0]
    scores=torch.einsum('lhd,eshd->lehs',q('q1'),key('k1'))*torch.einsum('lhd,eshd->lehs',q('q2'),key('k2'))/32**2
    value=layer.v(e).reshape(24,4,32)
    decoded=.5*torch.einsum('ehd,ihd->hei',value,layer.o.weight.reshape(128,4,32))
    table=torch.einsum('lehs,hei->lsei',scores,decoded)
    w=table.reshape(4,24,2,24,128).permute(0,4,2,1,3)
    c=contrasts(24,dtype).to(device)
    offset=(24*w.mean((-1,-2)).sum(-1)).reshape(512)
    a=torch.einsum('ldpje,ja,eb->ldpab',w,c,c).reshape(512,1058)
    return a,offset,c


def coordinates(binding_tokens,c):
    x=torch.nn.functional.one_hot(binding_tokens.reshape(24,2),24).permute(1,0,2).to(c)
    return torch.einsum('ja,pje,eb->pab',c,x,c).flatten()


def permuted_operator(a,c,permutation):
    r=c.T@c[permutation]
    return torch.einsum('opab,ac->opcb',a.reshape(512,2,23,23),r).reshape_as(a)


def transport_bound(a,b,r,norm_cap=1e6,vh=None):
    if vh is None:vh=torch.linalg.svd(a,full_matrices=False).Vh
    v=vh[:r];aq=a-(a@v.T)@v;bq=b-(b@v.T)@v
    # Valid for any computed V, without declaring its small singular values zero.
    q_norm_upper=1+float((v@v.T).abs().sum(-1).max())+1e-10
    tail=float(aq.norm());missing=float(bq.norm());target=float(b.norm())
    lower=max(0.,missing-norm_cap*tail)/(q_norm_upper*max(target,1e-30))
    return {'relative_frobenius_lower_bound':lower,'transport_spectral_norm_cap':norm_cap,'chosen_projection_rows':r,
            'a_q_frobenius':tail,'b_q_frobenius':missing,'q_spectral_norm_upper_bound':q_norm_upper,'target_frobenius':target,
            'one_percent_transport_ruled_out':lower>.01}


def controls():
    c=contrasts(4);a=torch.eye(3,dtype=torch.float64);b=a[[1,2,0]]
    positive=transport_bound(a,b,3);negative=transport_bound(torch.tensor([[1.,0.]],dtype=a.dtype),torch.tensor([[0.,1.]],dtype=a.dtype),1)
    near=torch.diag(torch.tensor([1.,1e-12],dtype=a.dtype));target=near[:,[1,0]];bounded=transport_bound(near,target,1)
    exact=target@torch.linalg.inv(near)
    checks={'orthonormal_contrasts':bool(torch.allclose(c.T@c,torch.eye(3,dtype=c.dtype),atol=1e-12)),
            'zero_sum_contrasts':float(c.sum(0).abs().max())<1e-12,'exact_transport_positive':positive['relative_frobenius_lower_bound']<1e-12,
            'missing_coordinate_negative':negative['relative_frobenius_lower_bound']>.1,
            'near_null_bounded_obstruction':bounded['relative_frobenius_lower_bound']>.1,
            'unbounded_transport_not_ruled_out':bool(torch.allclose(exact@near,target)) and float(torch.linalg.matrix_norm(exact,2))>1e6}
    return {'passed':all(checks.values()),'checks':checks}
