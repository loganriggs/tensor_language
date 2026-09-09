"""Final-reader source messages and independently hooked binding-pair cuts."""
import types
import torch


def messages(program,context):
    layer=program.background.layers[-1];p=layer.pattern(context['x'])[:,:,-1]
    z=p.transpose(1,2).unsqueeze(-1)*context['features']['v']
    source=.5*z.flatten(-2)@program.folded.T
    residual=.5*program.background.head(context['x'][:,-1])
    return source,residual


def native_binding_cuts(model,base):
    assert len(base)==1
    layer=model.layers[-1];original=layer.pattern
    def pattern(_self,x):
        p=original(x).clone()
        for j in range(24):p[j,:,:,2*j:2*j+2]=0
        return p
    layer.pattern=types.MethodType(pattern,layer)
    try:return model.head(layer(base.expand(24,-1,-1)))[:,-1]
    finally:del layer.pattern


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(28908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval();tok=torch.randint(8,(1,51))
    program=SourceEditProgram(model)
    with torch.inference_mode():
        native=model(tok);context=program.prepare(tok);source,residual=messages(program,context)
        pairs=source[:,:48].reshape(1,24,2,8).sum(2);cuts=native_binding_cuts(model,context['x'])
        checks={'sum_all_sources_and_residual':bool(torch.allclose(source.sum(1)+residual,native[:,-1],atol=1e-9,rtol=1e-10)),
                'all24_native_pair_cuts':bool(torch.allclose(native[:,-1]-pairs[0],cuts,atol=1e-9,rtol=1e-10)),
                'every_cut_live':bool((pairs.abs().amax(-1)>1e-12).all()),
                'hooks_restored':bool(torch.equal(native,model(tok)))}
    return {'passed':all(checks.values()),'checks':checks}
