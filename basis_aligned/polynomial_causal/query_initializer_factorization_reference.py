"""Exact first-layer document summary plus local query initialization."""
import types
import torch
from local_query_initializer_reference import local_state


def document_summary(program,binding_tokens,hop_tokens):
    layer=program.background.layers[0];b,n=binding_tokens.shape;query_position=n+2
    keys=layer.norm(program.background.embed(binding_tokens));query=layer.norm(program.background.embed(hop_tokens[:,None]))
    def project(name,x,positions):
        z=getattr(layer,name)(x).reshape(b,x.shape[1],layer.n_head,layer.d_head);a,c=z.chunk(2,-1)
        return z*layer.rotary.cos_cached[:,positions]+torch.cat((-c,a),-1)*layer.rotary.sin_cached[:,positions]
    qp=torch.tensor([query_position],device=binding_tokens.device);sp=torch.arange(n,device=binding_tokens.device)
    q1,q2=project('q1',query,qp),project('q2',query,qp);k1,k2=project('k1',keys,sp),project('k2',keys,sp)
    p=torch.einsum('bthd,bshd->bhts',q1,k1)*torch.einsum('bthd,bshd->bhts',q2,k2)/layer.d_head**2
    v=layer.v(keys).reshape(b,n,layer.n_head,layer.d_head)
    return (.5*layer.o(torch.einsum('bhts,bshd->bthd',p,v).flatten(-2)))[:,0]


def execute(program,tokens,keep=3,summary=None,local=None):
    # bit1 keeps document summary; bit2 keeps local tokens/residual.
    if summary is None:summary=document_summary(program,tokens[:,:-3],tokens[:,-1])
    if local is None:local=local_state(program,tokens)
    query=summary*bool(keep&1)+local*bool(keep&2)
    x=program.background.layers[0](program.background.embed(tokens[:,:-1]))
    x=torch.cat((x,query[:,None]),1)
    for layer in program.background.layers[1:-1]:x=layer(x)
    positions=torch.arange(tokens.shape[1],device=tokens.device);f=program.features(x,positions)
    return .5*program.background.head(x)+.5*program.aggregate(f,f,positions,positions)@program.folded.T


def native(model,tokens,keep):
    layer=model.layers[0];original=layer.pattern;handle=None
    def pattern(_self,x):
        p=original(x).clone()
        if not keep&1:p[:,:,-1,:tokens.shape[1]-3]=0
        if not keep&2:p[:,:,-1,tokens.shape[1]-3:]=0
        return p
    layer.pattern=types.MethodType(pattern,layer)
    if not keep&2:
        residual=.5*model.embed(tokens[:,-1])
        def remove_residual(_module,_args,output):
            output=output.clone();output[:,-1]-=residual;return output
        handle=layer.register_forward_hook(remove_residual)
    try:return model(tokens)
    finally:
        if handle is not None:handle.remove()
        del layer.pattern


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(35908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        tok=torch.randint(8,(2,12));other=tok.clone();other[:,-2]=(other[:,-2]+1)%8
        changed=tok.clone();changed[:,:-3]=(changed[:,:-3]+1)%8
    program=SourceEditProgram(model);checks={}
    with torch.inference_mode():
        before=model(tok);summary=document_summary(program,tok[:,:-3],tok[:,-1]);local=local_state(program,tok)
        checks['state_partition']=bool(torch.allclose(summary+local,model.layers[0](model.embed(tok))[:,-1],atol=1e-9,rtol=1e-10))
        checks['summary_query_independence']=bool(torch.equal(summary,document_summary(program,other[:,:-3],other[:,-1])))
        checks['local_document_independence']=bool(torch.equal(local,local_state(program,changed)))
        for keep in range(4):
            actual=execute(program,tok,keep,summary,local);oracle=native(model,tok,keep)
            checks['native_keep_'+str(keep)]=bool(torch.allclose(actual,oracle,atol=1e-9,rtol=1e-10))
            if keep in (1,2):checks['live_removal_'+str(keep)]=float((actual-before).abs().max())>1e-12
            if keep==0:checks['zero_query_absorbing']=not bool(actual[:,-1].ne(0).any())
        checks['both_native_tokens']=bool(torch.allclose(execute(program,tok),before,atol=1e-9,rtol=1e-10))
        checks['hooks_restored']=bool(torch.equal(before,model(tok)))
    return {'passed':all(checks.values()),'checks':checks}
