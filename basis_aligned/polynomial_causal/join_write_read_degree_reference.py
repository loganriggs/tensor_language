"""Exact final-query response to source-only write scaling, with live RMS."""
import torch


def normalizer(curve,alpha):
    norm=curve['norm']
    square=norm[...,0]*(alpha+norm[...,1]).square()+norm[...,2]+curve['epsilon']
    return square.pow(-1.5)


def contribution(curve,alpha):
    coeff=curve['coefficients']
    numerator=((coeff[:,:,3]*alpha+coeff[:,:,2])*alpha+coeff[:,:,1])*alpha+coeff[:,:,0]
    return numerator*normalizer(curve,alpha)[...,None]


def evaluate(curve,alpha):
    return curve['background']+contribution(curve,alpha).sum(1)


def evaluate_degree(curve,alpha,degree):
    assert degree in (1,2,3)
    c=curve['coefficients'];numerator=c[:,:,0]+alpha**degree*c[:,:,degree]
    return curve['background']+(normalizer(curve,alpha)[...,None]*numerator).sum(1)


def compile_curve(program,context,write):
    assert write.shape==context['x'].shape and not bool(write[:,-1].ne(0).any())
    positions=torch.where(write.ne(0).any((0,2)))[0];assert len(positions)>0
    layer=program.background.layers[-1];d=write[:,positions];z=context['x'][:,positions]-d
    b,s,width=z.shape;h=layer.n_head;hd=layer.d_head
    assert isinstance(layer.norm,torch.nn.RMSNorm) and not layer.norm.elementwise_affine
    epsilon=layer.norm.eps if layer.norm.eps is not None else torch.finfo(z.dtype).eps
    def project(x,name):
        value=getattr(layer,name)(x).reshape(b,s,h,hd)
        if name!='v':
            a,c=value.chunk(2,-1)
            value=value*layer.rotary.cos_cached[:,positions]+torch.cat((-c,a),-1)*layer.rotary.sin_cached[:,positions]
        return value
    def key_read(x,name):
        return torch.einsum('bhd,bshd->bsh',context['features']['q'+name[-1]][:,-1],project(x,name))/hd
    a0,a1=key_read(z,'k1'),key_read(d,'k1');b0,b1=key_read(z,'k2'),key_read(d,'k2')
    v0,v1=project(z,'v'),project(d,'v')
    scale=lambda a,b,v:(a*b)[...,None]*v
    raw=[scale(a0,b0,v0),scale(a1,b0,v0)+scale(a0,b1,v0)+scale(a0,b0,v1),
         scale(a1,b1,v0)+scale(a1,b0,v1)+scale(a0,b1,v1),scale(a1,b1,v1)]
    coefficients=torch.stack([.5*r.flatten(-2)@program.folded.T for r in raw],2)
    d2=d.square().mean(-1);shift=(z*d).mean(-1)/torch.where(d2>0,d2,torch.ones_like(d2))
    orthogonal=(z-shift[...,None]*d).square().mean(-1)
    curve={'coefficients':coefficients,'norm':torch.stack((d2,shift,orthogonal),-1),
           'epsilon':epsilon,'source_positions':positions}
    curve['background']=context['logits'][:,-1]-contribution(curve,1.).sum(1)
    return curve


def removal_terms(curve):
    c=curve['coefficients'];g1=normalizer(curve,1.);g0=normalizer(curve,0.)
    return {'normalization':((g1-g0)[...,None]*c[:,:,0]).sum(1),
            **{name:(g1[...,None]*c[:,:,k]).sum(1) for k,name in enumerate(('constant','linear','quadratic','cubic')) if k}}


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(40908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        tokens=torch.randint(8,(2,8));write=torch.zeros(2,8,16,dtype=torch.float64);write[:,2:4]=.2*torch.randn(2,2,16,dtype=torch.float64)
    program=SourceEditProgram(model);checks={}
    with torch.inference_mode():
        context=program.prepare(tokens);curve=compile_curve(program,context,write)
        for alpha in (-1.,0.,.5,1.,2.):
            actual=evaluate(curve,alpha);oracle=program.edit(context,(alpha-1)*write)[:,-1]
            checks['native_scale_'+str(alpha)]=bool(torch.allclose(actual,oracle,atol=1e-9,rtol=1e-10))
        terms=removal_terms(curve)
        checks['removal_partition']=bool(torch.allclose(sum(terms.values()),evaluate(curve,1.)-evaluate(curve,0.),atol=1e-9,rtol=1e-10))
        checks['live_curve']=float((evaluate(curve,2.)-evaluate(curve,0.)).abs().max())>1e-12
        checks['finite_norm']=bool((curve['norm'][...,[0,2]]>=0).all())
        for degree in (1,2):
            planted={**curve,'coefficients':curve['coefficients'].clone()}
            for k in (1,2,3):
                if k!=degree:planted['coefficients'][:,:,k]=0
            checks['planted_degree_'+str(degree)]=all(torch.allclose(evaluate(planted,a),evaluate_degree(planted,a,degree),atol=1e-12,rtol=1e-12) for a in (-1.,.5,2.))
        checks['omitted_degree_live']=float((evaluate(curve,2.)-evaluate_degree(curve,2.,1)).abs().max())>1e-12
        bad=write.clone();bad[:,-1]=1
        try:compile_curve(program,context,bad)
        except AssertionError:checks['changed_query_rejected']=True
        else:checks['changed_query_rejected']=False
    return {'passed':all(checks.values()),'checks':checks}
