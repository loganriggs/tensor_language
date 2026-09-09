"""Context-free first-layer query initialization with the final three tokens."""
import types
import torch


def local_state(program,tokens):
    layer=program.background.layers[0];b,t=tokens.shape;positions=torch.arange(t-3,t,device=tokens.device)
    e=program.background.embed(tokens[:,-3:]);x=layer.norm(e)
    def project(name,query=False):
        z=getattr(layer,name)(x[:,-1:] if query else x).reshape(b,1 if query else 3,layer.n_head,layer.d_head)
        pos=positions[-1:] if query else positions;a,c=z.chunk(2,-1)
        return z*layer.rotary.cos_cached[:,pos]+torch.cat((-c,a),-1)*layer.rotary.sin_cached[:,pos]
    q1,q2=project('q1',True),project('q2',True);k1,k2=project('k1'),project('k2')
    p=torch.einsum('bthd,bshd->bhts',q1,k1)*torch.einsum('bthd,bshd->bhts',q2,k2)/layer.d_head**2
    v=layer.v(x).reshape(b,3,layer.n_head,layer.d_head)
    z=torch.einsum('bhts,bshd->bthd',p,v).flatten(-2)
    return (.5*e[:,-1:]+.5*layer.o(z))[:,0]


def execute(program,tokens):
    x=program.background.layers[0](program.background.embed(tokens[:,:-1]))
    x=torch.cat((x,local_state(program,tokens)[:,None]),dim=1)
    for layer in program.background.layers[1:-1]:x=layer(x)
    positions=torch.arange(tokens.shape[1],device=tokens.device);features=program.features(x,positions)
    z=program.aggregate(features,features,positions,positions)
    return .5*program.background.head(x)+.5*z@program.folded.T


def native(model,tokens,first_layer_only=False):
    layer=model.layers[0];original=layer.pattern
    def pattern(_self,x):
        p=original(x).clone();p[:,:,-1,:tokens.shape[1]-3]=0;return p
    layer.pattern=types.MethodType(pattern,layer)
    try:return layer(model.embed(tokens)) if first_layer_only else model(tokens)
    finally:del layer.pattern


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(34908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        tokens=torch.randint(8,(2,12));other=tokens.clone();other[:,:-3]=(other[:,:-3]+1)%8
    program=SourceEditProgram(model)
    with torch.inference_mode():
        before=model(tokens);candidate=execute(program,tokens);local=local_state(program,tokens)
        checks={'local_native_state':bool(torch.allclose(local,native(model,tokens,True)[:,-1],atol=1e-9,rtol=1e-10)),
            'full_native_mask':bool(torch.allclose(candidate,native(model,tokens),atol=1e-9,rtol=1e-10)),
            'binding_prefix_independence':bool(torch.equal(local,local_state(program,other))),
            'earlier_outputs':bool(torch.allclose(candidate[:,:-1],before[:,:-1],atol=1e-9,rtol=1e-10)),
            'live_omitted_edges':float((candidate-before).abs().max())>1e-12,'hooks_restored':bool(torch.equal(before,model(tokens)))}
    return {'passed':all(checks.values()),'checks':checks}
