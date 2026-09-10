"""Differentiate the unchanged native suffix with respect to one MLP output.

The native backend enters no_grad. A source hook enables recording from the
detached source output onward; the backend's context and our finally restore
the caller's grad mode. No weights are trained or gradients accumulated.
"""
from contextlib import contextmanager
import torch


@contextmanager
def capture(model, layer=4):
    record = {}; handles = []
    module = model.transformer.h[layer].mlp
    try:
        handles.append(module.Down.register_forward_pre_hook(lambda _m, a: record.update(phi=a[0].detach().clone())))
        handles.append(module.register_forward_hook(lambda _m, _a, y: record.update(source=y.detach().clone())))
        handles.append(model.lm_head.register_forward_hook(lambda _m, _a, y: record.update(raw_logits=y.detach().clone())))
        yield record
    finally:
        for h in handles: h.remove()


def endpoint_logits(raw, batch):
    return torch.stack([30*torch.tanh(raw[i, pos]/30) for i, pos in enumerate(batch.semantic_positions)])


def gradient(backend, batch, layer=4):
    model = backend.model; module = model.transformer.h[layer].mlp
    flags = [(p, p.requires_grad) for p in model.parameters()]
    original_grad = torch.is_grad_enabled(); handles = []; record = {}
    def source(_m, _a, y):
        torch.set_grad_enabled(True)
        record['source'] = y.detach().clone().requires_grad_(True)
        return record['source']
    try:
        for p, _flag in flags: p.requires_grad_(False)
        handles.append(module.Down.register_forward_pre_hook(lambda _m, a: record.update(phi=a[0].detach().clone())))
        handles.append(module.register_forward_hook(source))
        handles.append(model.lm_head.register_forward_hook(lambda _m, _a, y: record.update(raw_logits=y)))
        native = backend.native(batch, capture=False)
        with torch.enable_grad():
            logits = endpoint_logits(record['raw_logits'], batch)
            margin = torch.stack([logits[i, a]-logits[i, f] for i, (a, f) in enumerate(zip(batch.answer_ids, batch.foil_ids))])
            grad, = torch.autograd.grad(margin.sum(), record['source'])
        return {'gradient': grad.detach(), 'source': record['source'].detach(),
                'phi': record['phi'], 'logits': logits.detach(), 'native': native}
    finally:
        for h in handles: h.remove()
        for p, flag in flags: p.requires_grad_(flag)
        torch.set_grad_enabled(original_grad)


def controls():
    from jacclust.tt_model import GPT, GPTConfig
    from circuit_fast_screen_producer import Bilin18TorchBackend, ModelBatch
    import torch.nn.functional as F
    import value_lineage_capture as C
    class TinyBackend(Bilin18TorchBackend):
        def __init__(self, model): self.model=model; self.torch=torch; self.F=F; self.device='cpu'
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(910320)
        model=GPT(GPTConfig(vocab_size=16,n_layer=4,n_head=2,n_embd=16,bilinear=True,squared_attn=True,bilinear_attn=True)).double().eval()
        model.lm_head.weight.data.normal_(std=.1)
        for b in model.transformer.h:
            b.attn.c_proj.weight.data.normal_(std=.05); b.mlp.Down.weight.data.normal_(std=.05)
            b.lambdas.data.copy_(torch.tensor([.7,.2]))
        delta=torch.randn(2,5,16,dtype=torch.float64)*.1
    batch=ModelBatch(('a','b'),'base',((1,2,3),(4,5,6,7,8)),(1,3),(2,4),(2,4))
    backend=TinyBackend(model); flags=[p.requires_grad for p in model.parameters()]
    with capture(model,1) as native: backend.native(batch,capture=False)
    with torch.no_grad():
        record=gradient(backend,batch,1)
        restored_grad=not torch.is_grad_enabled()
    values=[]
    for sign in (1,-1):
        with C.replace_mlp(model.transformer.h[1].mlp,record['source']+sign*1e-5*delta,[3,5]):
            with capture(model,1) as changed: backend.native(batch,capture=False)
        z=endpoint_logits(changed['raw_logits'],batch)
        values.append(sum(z[i,a]-z[i,f] for i,(a,f) in enumerate(zip(batch.answer_ids,batch.foil_ids))))
    numerical=(values[0]-values[1])/2e-5; predicted=(record['gradient']*delta).sum()
    error=float(abs(numerical-predicted))
    # Exercise the real cleanup path after the enabling source hook has fired.
    def fail(_m,_a): raise RuntimeError('planted downstream failure')
    hook=model.transformer.h[2].mlp.register_forward_pre_hook(fail)
    caught=False
    try:
        with torch.no_grad():
            try: gradient(backend,batch,1)
            except RuntimeError as e:
                assert str(e)=='planted downstream failure'; caught=not torch.is_grad_enabled()
    finally: hook.remove()
    checks={'primal_exact':torch.equal(record['logits'],endpoint_logits(native['raw_logits'],batch)),
            'central_difference':error<1e-8,'live_gradient':float(record['gradient'].norm())>1e-3,
            'padded_gradient_zero':bool((record['gradient'][0,3:]==0).all()),
            'grad_mode_restored':restored_grad and caught,
            'parameter_flags_restored':flags==[p.requires_grad for p in model.parameters()],
            'hooks_restored':not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()),
            'no_weight_gradients':all(p.grad is None for p in model.parameters())}
    return {'passed':all(checks.values()),'checks':checks,'central_difference_abs_error':error,
            'tiny_complete_forwards':4,'tiny_partial_exception_forwards':1,'tiny_backwards':1}
