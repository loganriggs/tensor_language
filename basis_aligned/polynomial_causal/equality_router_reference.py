"""Explicit equality/type routing and its native intervention correspondence.

No checkpoint or accelerator access on import. The first-layer domain is token IDs;
later contextual layers remain in the priced background and are recomputed live.
"""
from __future__ import annotations

import copy
import types

import torch
from torch import nn

ENTITIES = 24
VOCAB = 29
ORBITS = 37


def orbit_index(a, b):
    """Two entity orbits, ten directed mixed orbits, 25 special-token pairs."""
    ae, be = a < ENTITIES, b < ENTITIES
    return torch.where(ae & be, (a != b).long(),
                       torch.where(ae, 2 + b - ENTITIES,
                                   torch.where(be, 7 + a - ENTITIES,
                                               12 + 5*(a-ENTITIES) + b-ENTITIES)))


def orbit_projection(kernel):
    """Uniform Frobenius projection. Last two axes are ordered token pairs."""
    ids = torch.arange(VOCAB, device=kernel.device)
    orbit = orbit_index(ids[:, None], ids[None, :])
    return torch.stack([kernel[..., orbit == k].mean(-1) for k in range(ORBITS)], -1)


def lag_kernel(model):
    """First-layer joint QK1*QK2 kernel, keeping both factors coupled."""
    layer = model.layers[0]
    n = layer.norm(model.embed.weight)
    h, p = layer.n_head, layer.d_head
    vectors = {name: getattr(layer, name)(n).reshape(VOCAB, h, p)
               for name in ('q1', 'k1', 'q2', 'k2')}
    c = layer.rotary.cos_cached[0, :, 0][:, None, None, :]
    s = layer.rotary.sin_cached[0, :, 0][:, None, None, :]
    def rotate(q):
        left, right = q.chunk(2, -1)
        return q[None]*c + torch.cat((-right, left), -1)[None]*s
    first = torch.einsum('dvhi,whi->hdvw', rotate(vectors['q1']), vectors['k1'])
    second = torch.einsum('dvhi,whi->hdvw', rotate(vectors['q2']), vectors['k2'])
    return first*second/(p*p)


def edited_pattern(pattern, tokens, remove):
    if not remove:
        return pattern
    pattern = pattern.clone()
    equal = (tokens[:, :, None] == tokens[:, None, :]) & (tokens[:, :, None] < ENTITIES)
    for head in remove:
        pattern[:, head].masked_fill_(equal, 0)
    return pattern


def native_forward(model, tokens, remove=()):
    """Execute the original module forwards, editing exactly the declared edges."""
    layer = model.layers[0]
    old = layer.pattern
    def pattern(_self, x):
        return edited_pattern(old(x), tokens, remove)
    layer.pattern = types.MethodType(pattern, layer)
    try:
        return model(tokens)
    finally:
        # Remove the temporary instance override, restoring the class method.
        del layer.pattern


def reference_forward(model, tokens, remove=()):
    """Independent first-layer contraction control; all later native layers live."""
    x = model.embed(tokens)
    layer = model.layers[0]
    pattern = edited_pattern(layer.pattern(x), tokens, remove)
    value = layer.v(layer.norm(x)).reshape(*tokens.shape, layer.n_head, layer.d_head)
    update = layer.o(torch.einsum('bhts,bshp->bthp', pattern, value).flatten(2))
    x = x+update if layer.residual == 'add' else torch.lerp(x, update, layer.scale)
    for layer in model.layers[1:]:
        x = layer(x)
    return model.head(x)


class RouterProgram(nn.Module):
    """Executable partial extraction: replaced Q/K parameters physically removed."""
    def __init__(self, model, table, invariant):
        super().__init__()
        self.background = copy.deepcopy(model)
        self.register_buffer('table', table.clone())
        self.invariant = invariant
        first = self.background.layers[0]
        for name in ('q1', 'k1', 'q2', 'k2', 'rotary', 'mask'):
            delattr(first, name)

    def pattern(self, tokens):
        length = tokens.shape[1]
        pos = torch.arange(length, device=tokens.device)
        delta = pos[:, None]-pos[None, :]
        heads = torch.arange(self.table.shape[0], device=tokens.device)[None, :, None, None]
        lag = delta.clamp_min(0)[None, None]
        a, b = tokens[:, None, :, None], tokens[:, None, None, :]
        if self.invariant:
            pattern = self.table[heads, lag, orbit_index(a, b)]
        else:
            pattern = self.table[heads, lag, a, b]
        return pattern.masked_fill(delta[None, None] < 0, 0)

    def forward(self, tokens, remove=()):
        x = self.background.embed(tokens)
        layer = self.background.layers[0]
        value = layer.v(layer.norm(x)).reshape(*tokens.shape, layer.n_head, layer.d_head)
        pattern = edited_pattern(self.pattern(tokens), tokens, remove)
        update = layer.o(torch.einsum('bhts,bshp->bthp', pattern, value).flatten(2))
        x = x+update if layer.residual == 'add' else torch.lerp(x, update, layer.scale)
        for layer in self.background.layers[1:]:
            x = layer(x)
        return self.background.head(x)


def control_fixtures():
    g = torch.Generator().manual_seed(2908)
    ids = torch.arange(VOCAB)
    orbit = orbit_index(ids[:, None], ids[None, :])
    coefficients = torch.randn(4, 3, ORBITS, generator=g, dtype=torch.float64)
    planted = coefficients[..., orbit]
    recovered = orbit_projection(planted)
    perturb = planted.clone()
    perturb[0, 0, 0, 1] += 2**-20
    generic = torch.randn(4, 3, VOCAB, VOCAB, generator=g, dtype=torch.float64)
    permutation = torch.cat((torch.randperm(ENTITIES, generator=g), ids[ENTITIES:]))
    perturbed_error = float((perturb-orbit_projection(perturb)[..., orbit]).abs().max())
    generic_error = float((generic-orbit_projection(generic)[..., orbit]).norm())
    checks = {
        '37_exhaustive_orbits': bool(torch.equal(orbit.unique(), torch.arange(ORBITS))),
        'planted_recovery': bool(torch.allclose(recovered, coefficients, atol=1e-12, rtol=1e-12)),
        'perturbation_rejected_as_exact': perturbed_error > 1e-8,
        'generic_rejected_as_exact': generic_error > 1,
        'permutation_equivariance': bool(torch.equal(orbit, orbit[permutation][:, permutation])),
        'independent_consumers_not_merged': bool((recovered[0]-recovered[1]).norm() > 1),
        'projection_idempotent': bool(torch.allclose(orbit_projection(recovered[..., orbit]), recovered,
                                                   atol=1e-12, rtol=1e-12)),
    }
    # Independent-router/shared-payload control: same value vector, unequal reads.
    payload = torch.randn(VOCAB, 2, generator=g, dtype=torch.float64)
    outputs = torch.einsum('hdab,bv->hdav', planted, payload)
    checks['same_payload_distinct_reads'] = bool((outputs[0]-outputs[1]).norm() > 1)
    return dict(passed=all(checks.values()), checks=checks,
                perturbed_max_residual=perturbed_error, generic_residual_l2=generic_error)
