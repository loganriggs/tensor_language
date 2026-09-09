"""Exact score-cell replacement as a residual write versus native attention hook."""
import types
import torch


def execute(program,tokens,mask,head,scores):
    assert bool((mask.sum((-1,-2))==scores.shape[-1]).all())
    x=program.background.embed(tokens)
    for layer in program.background.layers[:2]:x=layer(x)
    layer=program.background.layers[2]
    old=layer.pattern(x)[:,head];change=torch.zeros_like(old)
    change[mask]=scores.flatten()-old[mask]
    value=layer.v(layer.norm(x)).reshape(len(tokens),tokens.shape[1],layer.n_head,layer.d_head)[:,:,head]
    delta=layer.scale*(change@value)@layer.o.weight[:,head*layer.d_head:(head+1)*layer.d_head].T
    assert layer.residual=='lerp'
    context=program.prepare(tokens)
    return program.edit(context,delta)


def native(model,tokens,mask,head,scores):
    layer=model.layers[2];original=layer.pattern
    def replace(_self,x):
        p=original(x).clone();p[:,head][mask]=scores.flatten();return p
    layer.pattern=types.MethodType(replace,layer)
    try:return model(tokens)
    finally:del layer.pattern


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    from matcher_factor_truth_table_reference import factors
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(39908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        tokens=torch.randint(8,(2,8));donor=tokens.clone();donor[:,:5]=(donor[:,:5]+1)%8
    program=SourceEditProgram(model);mask=torch.zeros(2,8,8,dtype=torch.bool);mask[:,4:6,0:2]=True
    checks={};compiled={};oracles={}
    with torch.inference_mode():
        before=model(tokens);old,_=factors(model,tokens,mask,1);new,_=factors(model,donor,mask,1)
        for p in range(2):
            for q in range(2):
                name=str(p)+str(q);scores=(new[0] if p else old[0])*(new[1] if q else old[1])
                compiled[name]=execute(program,tokens,mask,1,scores);oracles[name]=native(model,tokens,mask,1,scores)
                checks['native_'+name]=bool(torch.allclose(compiled[name],oracles[name],atol=1e-9,rtol=1e-10))
        zero=torch.zeros_like(old[0]);cut=execute(program,tokens,mask,1,zero)
        checks['native_cut']=bool(torch.allclose(cut,native(model,tokens,mask,1,zero),atol=1e-9,rtol=1e-10))
        checks['identity']=bool(torch.allclose(compiled['00'],before,atol=1e-9,rtol=1e-10))
        checks['live_cut']=float((cut-before).abs().max())>1e-12
        checks['live_transfer']=float((compiled['11']-before).abs().max())>1e-12
        mixed=lambda d:d['11']-d['10']-d['01']+d['00']
        checks['mixed']=bool(torch.allclose(mixed(compiled),mixed(oracles),atol=1e-9,rtol=1e-10))
        checks['mixed_live']=float(mixed(compiled).abs().max())>1e-12
        other_query=tokens.clone();other_query[:,-1]=(other_query[:,-1]+1)%8
        reused,_=factors(model,other_query,mask,1)
        checks['binding_factor_query_independence']=all(torch.equal(a,b) for a,b in zip(old,reused))
        checks['unchanged_earlier']=bool(torch.allclose(compiled['11'][:,:4],before[:,:4],atol=1e-9,rtol=1e-10))
        checks['hooks_restored']=bool(torch.equal(before,model(tokens)))
    return {'passed':all(checks.values()),'checks':checks}
