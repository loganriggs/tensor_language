"""Exact selected L2 writes and their cross-context intervention interface."""
from contextlib import contextmanager
import types
import torch


def populations():
    out = {}
    for pop, seed in (('iid', 17909), ('ood_two_12cycles', 17910)):
        g = torch.Generator().manual_seed(seed); worlds = []
        for world in range(16):
            perm = torch.randperm(24, generator=g); fmap = torch.empty(24, dtype=torch.long)
            if pop == 'iid':
                fmap[perm] = perm.roll(-1)
            else:
                fmap[perm[:12]] = perm[:12].roll(-1); fmap[perm[12:]] = perm[12:].roll(-1)
            starts = (int(perm[0]), int(perm[12]))
            chains = []
            for start in starts:
                chain = [start]
                for _ in range(3): chain.append(int(fmap[chain[-1]]))
                chains.append(chain)
            rest = [int(v) for v in perm if int(v) not in chains[0][:3]+chains[1][:3]]
            for arrangement in range(2):
                fixed = {}; masks = {}; heads = {}
                for j, slots in enumerate(((3, 11, 19), (5, 13, 21))):
                    forward = (j == arrangement)
                    order = (0, 2, 1) if forward else (0, 1, 2)
                    fixed.update({slot: chains[j][k] for slot, k in zip(slots, order)})
                    mask = torch.zeros(51, 51, dtype=torch.bool)
                    mask[2*slots[2]:2*slots[2]+2, 2*slots[1]:2*slots[1]+2] = True
                    masks[j] = mask; heads[j] = 1 if forward else 2
                contexts = []
                for _ in range(2):
                    shuffled = torch.tensor(rest)[torch.randperm(18, generator=g)].tolist(); it = iter(shuffled)
                    keys = torch.tensor([fixed[s] if s in fixed else next(it) for s in range(24)])
                    contexts.append(torch.stack((keys, fmap[keys]), -1).flatten())
                recipient = []; answers = []; query = []; hops = []
                for j in range(2):
                    for hop in range(4):
                        recipient.append(torch.cat((contexts[1], torch.tensor([24, starts[j], 25+hop]))))
                        answers.append(chains[j][hop]); query.append(j); hops.append(hop)
                donor = torch.cat((contexts[0], torch.tensor([24, starts[0], 25])))[None]
                worlds.append({'world': world, 'arrangement': arrangement, 'donor': donor,
                               'recipient': torch.stack(recipient), 'masks': masks, 'heads': heads,
                               'answers': torch.tensor(answers), 'query': torch.tensor(query), 'hops': torch.tensor(hops)})
        out[pop] = worlds
    return out


def contributions(model, tokens, masks, heads):
    x = model.embed(tokens)
    for layer in model.layers[:2]: x = layer(x)
    layer = model.layers[2]; b, t = tokens.shape
    p = layer.pattern(x)
    v = layer.v(layer.norm(x)).reshape(b, t, layer.n_head, layer.d_head)
    assert layer.o.bias is None
    out = {}
    for j, mask in masks.items():
        h = heads[j]
        z = torch.einsum('bts,bsd->btd', p[:, h]*mask, v[:, :, h])
        out[j] = z @ layer.o.weight[:, h*layer.d_head:(h+1)*layer.d_head].T
        if layer.residual != 'add': out[j] *= layer.scale
    return out


@contextmanager
def intervene(model, masks, heads, selected, writes=None):
    layer = model.layers[2]; original = layer.pattern; handle = None
    def cut(_self, x):
        p = original(x).clone()
        for j in selected:
            p[:, heads[j]] = p[:, heads[j]].masked_fill(masks[j], 0)
        return p
    layer.pattern = types.MethodType(cut, layer)
    try:
        if writes is not None:
            def restore(_module, _args, output):
                return output+sum(writes[j] for j in selected)
            handle = layer.register_forward_hook(restore)
        yield
    finally:
        if handle is not None: handle.remove()
        del layer.pattern


def controls():
    from deep_model import DeepModel
    checks = {}; all_worlds = populations()
    for pop, worlds in all_worlds.items():
        identical_maps = fixed_facts = disjoint = causal = changed = True
        for w in worlds:
            d, r = w['donor'][0], w['recipient'][0]
            fd = torch.empty(24, dtype=torch.long).scatter_(0, d[:48:2], d[1:48:2])
            fr = torch.empty(24, dtype=torch.long).scatter_(0, r[:48:2], r[1:48:2])
            identical_maps &= torch.equal(fd, fr)
            for slot in (3, 11, 19, 5, 13, 21): fixed_facts &= torch.equal(d[slot*2:slot*2+2], r[slot*2:slot*2+2])
            disjoint &= not bool((w['masks'][0].any(-1)&w['masks'][1].any(-1)).any())
            causal &= not any(bool(m.triu(1).any()) for m in w['masks'].values())
            changed &= not torch.equal(d[:48], r[:48])
            for row, hop, answer in zip(w['recipient'], w['hops'], w['answers']):
                val = row[49]
                for _ in range(int(hop)): val = fr[val]
                assert val == answer
        checks.update({pop+'_maps': identical_maps, pop+'_fixed_facts': fixed_facts,
                       pop+'_disjoint': disjoint, pop+'_causal': causal, pop+'_changed': changed})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17908)
        model = DeepModel(29, 16, 4, ['attn']*4, 240, norm='rms').double().eval()
    w = all_worlds['iid'][0]; tokens = w['recipient']; masks, heads = w['masks'], w['heads']
    with torch.inference_mode():
        original = model(tokens); writes = contributions(model, tokens, masks, heads)
        checks['query_independent'] = all(torch.allclose(v, v[:1].expand_as(v), atol=1e-9, rtol=1e-9) for v in writes.values())
        for selected in ((0,), (1,), (0, 1)):
            with intervene(model, masks, heads, selected, writes): actual = model(tokens)
            checks['self_replay_'+str(selected)] = bool(torch.allclose(actual, original, atol=1e-9, rtol=1e-9))
            with intervene(model, masks, heads, selected): cut = model(tokens)
            checks['live_cut_'+str(selected)] = bool((cut-original).abs().max() > 1e-12)
        checks['restored_hooks'] = bool(torch.equal(model(tokens), original))
    return {'passed': all(checks.values()), 'checks': checks}
