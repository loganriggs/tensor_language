"""Fixed-gain content-path cuts and graph-matched native failure cases."""
import torch
import source_port_interchange_reference as P


def execute(program,context,write):
    layer=program.background.layers[-1];x=context['x']
    eps=torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
    gain=torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)
    removed=layer.v(write*gain).reshape_as(context['features']['v'])
    mixed=dict(context['features']);mixed['v']=mixed['v']-removed
    return P.execute(program,context,mixed)


def native(model,base,write):
    layer=model.layers[-1];eps=torch.finfo(base.dtype).eps if layer.norm.eps is None else layer.norm.eps
    gain=torch.rsqrt(base.square().mean(-1,keepdim=True)+eps);removed=layer.v(write*gain)
    def remove(_module,_args,output):return output-removed
    handle=layer.v.register_forward_hook(remove)
    try:return model.head(layer(base))
    finally:handle.remove()


def manifest(atlas,corpus):
    lookup={(c['population'],c['row']):c for c in corpus['all_pairs']};out=[]
    for source_case in atlas['states']:
        if not source_case['native_wrong']:continue
        c=lookup[(source_case['population'],source_case['row'])];j=source_case['top_wrong_margin_source'];side=source_case['side']
        tokens=c[side+'_tokens'];f=dict(zip(tokens[:48:2],tokens[1:48:2]));positions={k:i for i,k in enumerate(tokens[:48:2])}
        key=tokens[2*j];wrong=source_case['native_prediction'];kind='raw' if f[key]==wrong else 'join'
        if kind=='join':assert f[f[key]]==wrong and positions[f[key]]<j
        sigma=c['entity_permutation']+list(range(24,29));inverse=[sigma.index(i) for i in range(29)]
        canonical_wrong=wrong if side=='old' else inverse[wrong]
        for target_side in ('old','renamed'):
            target=c[target_side+'_tokens'];ff=dict(zip(target[:48:2],target[1:48:2]));endpoint=ff[target[2*j]]
            if kind=='join':endpoint=ff[endpoint]
            foil=canonical_wrong if target_side=='old' else sigma[canonical_wrong]
            assert endpoint==foil
            out.append({'population':c['population'],'row':c['row'],'primary_side':side,'side':target_side,
                        'is_primary':side==target_side,'binding':j,'field':kind,'tokens':target,
                        'answer':c['correct_answer_original_labels'] if target_side=='old' else sigma[c['correct_answer_original_labels']],
                        'foil':foil})
    assert len(out)==42
    return out


def controls():
    from deep_model import DeepModel
    from exact_source_edit_reference import SourceEditProgram
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29908);model=DeepModel(8,16,4,['attn']*4,240,norm='rms').double().eval();tok=torch.randint(8,(2,12))
        a=torch.zeros(2,12,16,dtype=torch.float64);b=a.clone();a[:,5]=torch.randn(2,16)*.2;b[:,5]=torch.randn(2,16)*.2
    program=SourceEditProgram(model)
    with torch.inference_mode():
        before=model(tok);context=program.prepare(tok);x=execute(program,context,a);y=execute(program,context,b);joint=execute(program,context,a+b)
        checks={'native_projection_hook':bool(torch.allclose(x,native(model,context['x'],a),atol=1e-9,rtol=1e-10)),
                'overlapping_path_additivity':bool(torch.allclose(joint-before,(x-before)+(y-before),atol=1e-9,rtol=1e-10)),
                'identity':bool(torch.allclose(execute(program,context,a*0),before,atol=1e-9,rtol=1e-10)),
                'live_path':float((x-before).abs().max())>1e-12,'hooks_restored':bool(torch.equal(before,model(tok)))}
    return {'passed':all(checks.values()),'checks':checks}
