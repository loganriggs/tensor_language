"""Known TT/X/CC algebra applied to the small contextual hop checkpoint.

Formula follows mlp0_token_context_tensor_factorial.quadratic_tensor_branches;
kept dependency-light here so importing the small reference cannot load large-model
row/fitting machinery. No novelty or compression credit for this expansion.
"""
import itertools
import torch

TERMS = ('TT', 'X', 'CC')
ARMS = {('full' if len(s) == 3 else '+'.join(s) or 'empty'): s
        for k in range(4) for s in itertools.combinations(TERMS, k)}


def terms(model, tokens):
    embed = model.embed(tokens)
    first = model.layers[0]
    before = first(embed)
    e = embed*(1. if first.residual == 'add' else 1-first.scale)
    a = before-e
    mlp = model.layers[1]
    if isinstance(mlp.norm, torch.nn.RMSNorm):
        assert not mlp.norm.elementwise_affine
        eps = torch.finfo(before.dtype).eps if mlp.norm.eps is None else mlp.norm.eps
        gain = torch.rsqrt(before.square().mean(-1, keepdim=True)+eps)
        e, a = gain*e, gain*a
    else:
        assert isinstance(mlp.norm, torch.nn.Identity)
    le, re, la, ra = mlp.L(e), mlp.R(e), mlp.L(a), mlp.R(a)
    scale, residual = (1., 1.) if mlp.residual == 'add' else (mlp.scale, 1-mlp.scale)
    return residual*before, {'TT': scale*mlp.D(le*re), 'X': scale*mlp.D(le*ra+la*re), 'CC': scale*mlp.D(la*ra)}


def forward(model, tokens, keep=TERMS):
    residual, pieces = terms(model, tokens)
    x = residual+sum((pieces[n] for n in keep), torch.zeros_like(residual))
    return model.head(model.layers[-1](x))


def native_cut(model, tokens, keep=TERMS):
    _, pieces = terms(model, tokens)
    def subtract(_module, _args, output):
        for n in TERMS:
            if n not in keep:
                output = output-pieces[n]
        return output
    handle = model.layers[1].register_forward_hook(subtract)
    try:
        return model(tokens)
    finally:
        handle.remove()


def controls():
    from deep_model import DeepModel
    from hop_data import sample_docs
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8908)
        model = DeepModel(29, 16, 4, ['attn', 'mlp', 'attn'], 240, norm='rms').double().eval()
        tokens = sample_docs(2, torch.Generator().manual_seed(8908))[0][:, :-1]
    checks = {}
    with torch.inference_mode():
        for a, keep in ARMS.items():
            checks['native_cut_'+a] = bool(torch.allclose(forward(model, tokens, keep), native_cut(model, tokens, keep), atol=1e-9, rtol=1e-9))
        residual, pieces = terms(model, tokens)
        actual = model.layers[1](model.layers[0](model.embed(tokens)))
        checks['mlp_reassembly'] = bool(torch.allclose(residual+sum(pieces.values()), actual, atol=1e-9, rtol=1e-9))
        checks['live_terms'] = all(float(v.square().mean()) > 1e-12 for v in pieces.values())
        changed = tokens.clone(); changed[:, 120:] = (changed[:, 120:]+1)%29
        checks['causal_mixed_only'] = bool(torch.allclose(forward(model, tokens, ('X',))[:, :120], forward(model, changed, ('X',))[:, :120], atol=1e-9, rtol=1e-9))
    return {'passed': all(checks.values()), 'checks': checks}
