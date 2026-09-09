"""Explicit cold-query binding-source sets; repeated chain positions are deduplicated."""
import torch
from contextual_history_reference import HistoryReadout, native_forward


def source_masks(tokens):
    assert tokens.device.type == 'cpu' and tokens.shape[1] == 51
    b, t = tokens.shape
    masks = {k: torch.zeros(b, t, t, dtype=torch.bool) for k in ('H', 'B', 'C')}
    for row in range(b):
        keys, vals = tokens[row, :48:2].tolist(), tokens[row, 1:48:2].tolist()
        assert sorted(keys) == list(range(24)) and all(0 <= v < 24 for v in vals)
        fmap = dict(zip(keys, vals)); positions = {e: 2*j for j, e in enumerate(keys)}
        entity = int(tokens[row, 49]); hop = int(tokens[row, 50])-25
        assert int(tokens[row, 48]) == 24 and 0 <= entity < 24 and 0 <= hop <= 3
        path = []; current = entity
        for _ in range(hop):
            path.append(current); current = fmap[current]
        target = set(path[-1:]); previous = set(path[:-1])-target
        for kind, entities in (('H', target), ('B', previous)):
            for e in entities:
                p = positions[e]; masks[kind][row, 50, p:p+2] = True
        masks['C'][row, 50, 48:51] = True
    assert not (masks['H'] & masks['B']).any()
    assert not ((masks['H'] | masks['B']) & masks['C']).any()
    return masks


def controls():
    from deep_model import DeepModel
    # First row24-cycle, second row identity: path collisions must not duplicate cuts.
    keys = torch.arange(24).expand(2, -1)
    values = torch.stack(((torch.arange(24)+1)%24, torch.arange(24)))
    bindings = torch.stack((keys, values), -1).flatten(1)
    tokens = torch.cat((bindings, torch.tensor([[24, 0, 28], [24, 0, 28]])), 1)
    masks = source_masks(tokens)
    checks = {'cycle_target': bool(masks['H'][0, 50, 4:6].all() and masks['H'][0].sum() == 2),
              'cycle_prior': bool(masks['B'][0, 50, :4].all() and masks['B'][0].sum() == 4),
              'overlap_deduplicated': bool(masks['H'][1].sum() == 2 and masks['B'][1].sum() == 0),
              'query_three': bool((masks['C'].sum((1, 2)) == 3).all())}
    zero = tokens.clone(); zero[:, 50] = 25; zm = source_masks(zero)
    checks['hop0_empty_binding'] = not bool((zm['H'] | zm['B']).any())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10908)
        model = DeepModel(29, 16, 4, ['attn']*4, 240, norm='rms').double().eval()
    compiled = HistoryReadout(model)
    other = torch.zeros_like(masks['H']); other[:, 50] = ~(masks['H'] | masks['C'])[:, 50]
    kills = {'full': torch.zeros_like(other), 'removeT': masks['H'], 'removeP': masks['B'],
             'removeTP': masks['H'] | masks['B'], 'keepTQ': other}
    with torch.inference_mode():
        parts = compiled.parts(tokens, masks)
        full = parts['R']+parts['H']+parts['B']+parts['O']
        outputs = {'full': full, 'removeT': full-parts['H'], 'removeP': full-parts['B'],
                   'removeTP': full-parts['H']-parts['B'], 'keepTQ': full.clone()}
        outputs['keepTQ'][:, 50] = (parts['R']+parts['H']+parts['C'])[:, 50]
        for name, kill in kills.items():
            checks[name] = bool(torch.allclose(outputs[name], native_forward(model, tokens, kill), atol=1e-9, rtol=1e-9))
    return {'passed': all(checks.values()), 'checks': checks}
