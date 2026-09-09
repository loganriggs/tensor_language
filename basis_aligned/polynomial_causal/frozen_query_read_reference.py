"""Final attention payload contribution conditioned on native routing and RMS.

This linear attribution is neither a live source edit nor an input Jacobian.
The residual/query skip is outside its scope. All native coefficients are used.
"""
import torch


def read(program, context, payload):
    x = context['x']; assert payload.shape == x.shape
    layer = program.background.layers[-1]
    eps = torch.finfo(x.dtype).eps if layer.norm.eps is None else layer.norm.eps
    gain = torch.rsqrt(x.square().mean(-1, keepdim=True)+eps)
    features = {**context['features'], 'v': layer.v(payload*gain).reshape(
        len(x), x.shape[1], layer.n_head, layer.d_head)}
    query = {k: v[:, -1:] for k, v in context['features'].items()}
    z = program.aggregate(query, features, context['positions'][-1:], context['positions'])
    return .5*(z[:, 0]@program.folded.T)
