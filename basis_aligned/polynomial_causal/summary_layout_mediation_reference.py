"""Two first-layer inputs to later queries: binding prefix and document summary."""
import torch
import query_initializer_factorization_reference as Q


def execute(program,prefix_tokens,summary_tokens):
    assert torch.equal(prefix_tokens[:,-3:],summary_tokens[:,-3:])
    summary=Q.document_summary(program,summary_tokens[:,:-3],summary_tokens[:,-1])
    return Q.execute(program,prefix_tokens,summary=summary)


def native(model,prefix_tokens,summary_tokens):
    assert torch.equal(prefix_tokens[:,-3:],summary_tokens[:,-3:])
    state=model.layers[0](model.embed(summary_tokens))[:,-1].clone()
    def replace(_module,_args,output):
        result=output.clone();result[:,-1]=state;return result
    handle=model.layers[0].register_forward_hook(replace)
    try:return model(prefix_tokens)
    finally:handle.remove()


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(36908)
        model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        old=torch.randint(8,(2,12));new=old.clone();new[:,:-3]=old[:,:-3].roll(1,1)
    program=SourceEditProgram(model);checks={};compiled={};oracles={}
    with torch.inference_mode():
        before=model(old)
        for p,prefix in enumerate((old,new)):
            for s,summary in enumerate((old,new)):
                key=str(p)+str(s);compiled[key]=execute(program,prefix,summary);oracles[key]=native(model,prefix,summary)
                checks['native_'+key]=bool(torch.allclose(compiled[key],oracles[key],atol=1e-9,rtol=1e-10))
        checks['native_original']=bool(torch.allclose(compiled['00'],before,atol=1e-9,rtol=1e-10))
        checks['native_donor']=bool(torch.allclose(compiled['11'],model(new),atol=1e-9,rtol=1e-10))
        checks['local_equal']=bool(torch.equal(Q.local_state(program,old),Q.local_state(program,new)))
        checks['summary_earlier_zero']=bool(torch.equal(compiled['00'][:,:-1],compiled['01'][:,:-1]))
        checks['summary_live']=float((compiled['01'][:,-1]-compiled['00'][:,-1]).abs().max())>1e-12
        checks['prefix_live']=float((compiled['10'][:,-1]-compiled['00'][:,-1]).abs().max())>1e-12
        mixed=lambda d:d['11']-d['10']-d['01']+d['00']
        checks['mixed_correspondence']=bool(torch.allclose(mixed(compiled),mixed(oracles),atol=1e-9,rtol=1e-10))
        checks['mixed_live']=float(mixed(compiled)[:,-1].abs().max())>1e-12
        checks['hooks_restored']=bool(torch.equal(before,model(old)))
    return {'passed':all(checks.values()),'checks':checks}
