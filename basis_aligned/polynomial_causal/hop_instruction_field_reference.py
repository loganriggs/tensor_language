"""Literal final hop-token field and its complementary computed dependence."""
import torch


def fields(program,recipient,donor,old_tokens,new_tokens):
    assert torch.equal(old_tokens[:,:-1],new_tokens[:,:-1])
    direct=torch.zeros_like(recipient['x'])
    direct[:,-1]=(program.background.embed(new_tokens[:,-1])-program.background.embed(old_tokens[:,-1]))/8
    total=donor['x']-recipient['x']
    return direct,total-direct,total


def removal(program,context,tokens):
    delta=torch.zeros_like(context['x']);delta[:,-1]=-program.background.embed(tokens[:,-1])/8
    return delta


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(33908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval()
        old=torch.randint(8,(2,12));new=old.clone();new[:,-1]=(new[:,-1]+1)%8
    program=SourceEditProgram(model);checks={}
    with torch.inference_mode():
        before=model(old);target=model(new);a=program.prepare(old);b=program.prepare(new)
        direct,computed,total=fields(program,a,b,old,new)
        checks['only_final_state_changes']=not bool(total[:,:-1].ne(0).any())
        checks['field_sum']=bool(torch.allclose(direct+computed,total,atol=1e-12,rtol=1e-12))
        for name,delta in (('direct',direct),('computed',computed),('joint',total),('removal',removal(program,a,old)),('identity',direct*0)):
            actual=program.edit(a,delta);native=model.head(model.layers[-1](a['x']+delta))
            checks['native_'+name]=bool(torch.allclose(actual,native,atol=1e-9,rtol=1e-10))
            checks['earlier_outputs_'+name]=bool(torch.equal(actual[:,:-1],a['logits'][:,:-1]))
            if name!='identity':checks['live_'+name]=float((actual-a['logits']).abs().max())>1e-12
        checks['joint_equals_full_donor']=bool(torch.allclose(program.edit(a,total),target,atol=1e-9,rtol=1e-10))
        checks['model_unchanged']=bool(torch.equal(before,model(old)))
    return {'passed':all(checks.values()),'checks':checks}
