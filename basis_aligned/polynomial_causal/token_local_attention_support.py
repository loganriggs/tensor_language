"""Exact edit-support masks for token-local, unnormalized product attention.

Applies to the first attention layer before contextual mixing. Later-layer
queries/keys/values are contextual and do not automatically satisfy this rule.
"""
import torch


def single_edit_edges(changed):
    t=changed.shape[-1]
    causal=torch.ones(t,t,dtype=torch.bool,device=changed.device).tril()
    return (changed[..., :,None] | changed[..., None,:]) & causal


def mixed_edit_edges(cue,context):
    assert cue.shape==context.shape and not bool((cue & context).any())
    t=cue.shape[-1]
    causal=torch.ones(t,t,dtype=torch.bool,device=cue.device).tril()
    return ((cue[..., :,None] & context[..., None,:]) |
            (context[..., :,None] & cue[..., None,:])) & causal


def read(factors,edges=None):
    q,k,v,q2,k2=factors;t=q.shape[1];d=q.shape[-1]
    dot=lambda x,y:torch.einsum('bthd,bshd->bhts',x,y)
    pattern=dot(q,k)*dot(q2,k2)/d**2
    causal=torch.ones(t,t,dtype=torch.bool,device=q.device).tril()
    pattern=pattern*causal
    if edges is not None:pattern=pattern*edges.unsqueeze(-3)
    return torch.einsum('bhts,bshd->bhtd',pattern,v)


def mixed(values):
    return (values[3]-values[2])-(values[1]-values[0])


def controls():
    from jacclust.tt_model import GPT,GPTConfig
    from circuit_fast_screen_producer import Bilin18TorchBackend,ModelBatch
    import norm_preserving_response_hooks as H
    import torch.nn.functional as F
    torch.set_num_threads(2)
    class TinyBackend(Bilin18TorchBackend):
        def __init__(self,model):self.model=model;self.torch=torch;self.F=F;self.device='cpu'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(910401)
        model=GPT(GPTConfig(vocab_size=16,n_layer=2,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for block in model.transformer.h:
            block.attn.c_proj.weight.data.normal_(std=.05);block.mlp.Down.weight.data.normal_(std=.05)
    tokens=((1,3,4,5,6,7),(2,3,4,5,6,7),(1,3,4,8,6,7),(2,3,4,8,6,7))
    batch=ModelBatch(('00','01','10','11'),'base',tokens,(1,1,1,1),(2,2,2,2),(5,5,5,5));backend=TinyBackend(model)
    with H.capture(model,layers=(0,)) as native:backend.native(batch,capture=False)
    factors=native[0]['factors'];full=read(factors)
    cue=torch.tensor([True,False,False,False,False,False]);context=torch.tensor([False,False,False,True,False,False])
    cross=mixed_edit_edges(cue,context);single=single_edit_edges(cue)
    masked=read(factors,cross);cue_masked=read(factors,single)
    diff=mixed(full);pred=mixed(masked)
    error=lambda x,y:float((x-y).abs().max())
    errors={'native_factor_read':error(full,native[0]['read']),
            'single_cue_edges':max(error(full[1]-full[0],cue_masked[1]-cue_masked[0]),error(full[3]-full[2],cue_masked[3]-cue_masked[2])),
            'mixed_cross_edges':error(diff,pred),
            'unchanged_query_mixed_zero':float(diff[:,~(cue|context)].abs().max()),
            'no_causal_opposite_source_zero':float(diff[:,~cross.any(-1)].abs().max())}
    assert all(v<1e-10 for v in errors.values())
    assert float(diff.norm())>1e-8
    assert cross.nonzero().tolist()==[[3,0]]
    # Reverse temporal order: only the changed later query may interact with
    # the earlier source. This is a mask property, not another model run.
    reversed_edges=mixed_edit_edges(context,cue)
    assert torch.equal(reversed_edges,cross)
    # With fixed queries and no softmax, summing independent cue/context source
    # changes creates no mixed read. Softmax couples sources and breaks that rule.
    values=torch.tensor([0.,1.,0.],dtype=torch.float64)
    scores=[torch.tensor(x,dtype=torch.float64) for x in ((1.,1.,0.),(2.,1.,0.),(1.,3.,0.),(2.,3.,0.))]
    linear=[(s*values).sum() for s in scores];soft=[(s.softmax(-1)*values).sum() for s in scores]
    assert float(mixed(linear))==0 and abs(float(mixed(soft)))>.01
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()) and 'squared_attention' not in model.transformer.h[0].attn.__dict__
    assert restored
    return {'passed':True,'errors':errors,'mixed_read_norm':float(diff.norm()),
            'allowed_mixed_edges':cross.nonzero().tolist(),'single_edit_edge_count':int(single.sum()),
            'softmax_counterexample_mixed_value':float(mixed(soft)),
            'hooks_and_methods_restored':restored,'tiny_forwards':1,'tiny_sequence_instances':4,
            'trained_model_loaded':False,'gpu_accessed':False,
            'scope':'Exact first-layer edge-support control with actual RMS/RoPE factors. No later-layer extension or trained causal-relevance claim.'}


if __name__=='__main__':
    import json
    print(json.dumps(controls(),indent=2))
