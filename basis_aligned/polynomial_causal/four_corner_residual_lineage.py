"""Exact raw pre-MLP source lineage for cue/context four-corner diagnostics.

Sources are transported native writes, not independently extracted programs.
The original embedding's mixed difference vanishes for disjoint token edits.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def coefficients(model, layer=4):
    result={'embedding':1.}
    for i,block in enumerate(model.transformer.h[:layer+1]):
        keep,inject=(float(x) for x in block.lambdas.detach())
        result={k:keep*v for k,v in result.items()}
        result['embedding']+=inject
        result[f'attn{i}']=1.
        if i<layer:result[f'mlp{i}']=1.
    return result


@contextmanager
def capture(model, layer=4):
    """Track native recurrence in its operation order; capture unscaled writes."""
    record={'sources':{}};state={};handles=[];d=model.config.n_embd
    def embedding(_m,_a,y):
        state['embedding']=F.rms_norm(y,(d,)).detach()
        state['x']=state['embedding'];record['sources']['embedding']=state['embedding'].clone()
    def attention(i,block):
        def hook(_m,_a,y):
            record['sources'][f'attn{i}']=y[0].detach().clone()
            state['x']=block.lambdas[0]*state['x']+block.lambdas[1]*state['embedding']+y[0].detach()
        return hook
    def mlp(i):
        def hook(_m,_a,y):
            record['sources'][f'mlp{i}']=y.detach().clone();state['x']=state['x']+y.detach()
        return hook
    def target(_m,args):
        record['raw']=state['x'].clone();record['normalized']=args[0].detach().clone()
        record['recurrence_bitwise']=torch.equal(F.rms_norm(state['x'],(d,)),args[0])
    try:
        handles.append(model.transformer.wte.register_forward_hook(embedding))
        for i,block in enumerate(model.transformer.h[:layer+1]):
            handles.append(block.attn.register_forward_hook(attention(i,block)))
            if i<layer:handles.append(block.mlp.register_forward_hook(mlp(i)))
        handles.append(model.transformer.h[layer].mlp.register_forward_pre_hook(target))
        yield record
    finally:
        for h in handles:h.remove()


def mixed(corners):
    """Corner batch order:00,01,10,11, with equal token lengths."""
    # Pair equal cue edges first so disjoint-token cancellation is bitwise.
    return (corners[3]-corners[2])-(corners[1]-corners[0])


def controls():
    from jacclust.tt_model import GPT,GPTConfig
    from circuit_fast_screen_producer import Bilin18TorchBackend,ModelBatch
    torch.set_num_threads(2)
    class TinyBackend(Bilin18TorchBackend):
        def __init__(self,model):self.model=model;self.torch=torch;self.F=F;self.device='cpu'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(910345)
        model=GPT(GPTConfig(vocab_size=16,n_layer=5,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        for block in model.transformer.h:
            block.attn.c_proj.weight.data.normal_(std=.05);block.mlp.Down.weight.data.normal_(std=.05)
            block.lambdas.data.copy_(torch.tensor([.7,.2]))
    tokens=((1,3,5,7),(2,3,5,7),(1,3,6,7),(2,3,6,7))
    batch=ModelBatch(('00','01','10','11'),'base',tokens,(1,1,1,1),(2,2,2,2),(3,3,3,3))
    backend=TinyBackend(model)
    with capture(model) as record:backend.native(batch,capture=False)
    weights=coefficients(model)
    total=sum(weights[k]*v for k,v in record['sources'].items())
    mixed_parts={k:weights[k]*mixed(v) for k,v in record['sources'].items()}
    errors={'raw_source_sum':float((total-record['raw']).abs().max()),
            'mixed_source_sum':float((sum(mixed_parts.values())-mixed(record['raw'])).abs().max())}
    assert set(weights)==set(record['sources']) and len(weights)==10
    assert record['recurrence_bitwise'] and all(v<1e-10 for v in errors.values())
    # Per-token lookup/RMS sees either the cue change or the context change,
    # never both at one position; its mixed difference cancels algebraically.
    embedding_zero=bool((mixed_parts['embedding']==0).all());assert embedding_zero
    upstream_live=float(mixed(record['raw']).norm());assert upstream_live>1e-8
    # Violate disjoint editing at one token: token11 is a new token, so the
    # embedding mixed difference need not vanish. This controls the theorem's domain.
    e=record['sources']['embedding'][:,0]
    arbitrary_fourth=F.rms_norm(model.transformer.wte(torch.tensor([8])),(16,))[0].detach()
    violated=arbitrary_fourth-e[2]-e[1]+e[0]
    assert float(violated.norm())>.01
    restored=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules());assert restored
    return {'passed':True,'errors':errors,'recurrence_bitwise':record['recurrence_bitwise'],
            'embedding_mixed_exact_zero':embedding_zero,'raw_mixed_norm':upstream_live,
            'overlapping_edit_embedding_mixed_norm':float(violated.norm()),'coefficients':weights,
            'mixed_source_norms':{k:float(v.norm()) for k,v in mixed_parts.items()},
            'hooks_restored':restored,'tiny_forwards':1,'tiny_sequence_instances':4,
            'trained_model_loaded':False,'gpu_accessed':False,
            'scope':'Source lineage and disjoint-token algebra only; no trained interaction localization or independent extraction.'}


if __name__=='__main__':
    import json
    print(json.dumps(controls(),indent=2))
