"""Fixed later-suffix-binding source rule, nominated from an opened order audit."""
import itertools
import torch
from contextual_history_reference import HistoryReadout, native_forward

ORDERS = tuple(itertools.permutations(range(3)))
SLOTS = (3, 11, 19)


def source_masks(tokens):
    assert tokens.device.type == 'cpu' and tokens.shape[1] == 51
    masks = {k: torch.zeros(len(tokens), 51, 51, dtype=torch.bool) for k in ('H', 'B', 'C')}
    for row, tok in enumerate(tokens):
        keys, vals = tok[:48:2].tolist(), tok[1:48:2].tolist()
        assert sorted(keys) == list(range(24)) and int(tok[48]) == 24 and int(tok[50]) == 28
        fmap = dict(zip(keys, vals)); pos = {e: 2*i for i, e in enumerate(keys)}
        e = int(tok[49]); chain = [e, fmap[e], fmap[fmap[e]]]
        joined = set() if chain[1] == chain[2] else {max(chain[1:], key=lambda e: pos[e])}
        ignored = set(chain)-joined
        for kind, entities in (('H', joined), ('B', ignored)):
            for e in entities:
                masks[kind][row, 50, pos[e]:pos[e]+2] = True
        masks['C'][row, 50, 48:51] = True
    assert not (masks['H'] & masks['B']).any()
    return masks


def populations(torch):
    out = {}
    for kind, seed in (('iid', 11909), ('ood_two_12cycles', 11910), ('fixed_point', 11911)):
        g = torch.Generator().manual_seed(seed)
        rows, answers, orders, world_ids = [], [], [], []
        for world in range(32):
            perm = torch.randperm(24, generator=g)
            fmap = torch.empty(24, dtype=torch.long)
            if kind == 'iid':
                fmap[perm] = perm.roll(-1)
            elif kind == 'ood_two_12cycles':
                fmap[perm[:12]] = perm[:12].roll(-1); fmap[perm[12:]] = perm[12:].roll(-1)
            else:
                fmap = perm.clone()
            e = int(torch.randint(24, (), generator=g))
            if kind == 'fixed_point':
                predecessor = int(torch.where(fmap == e)[0][0]); old = int(fmap[e])
                fmap[e] = e; fmap[predecessor] = old
            chain = [e, int(fmap[e]), int(fmap[fmap[e]])]
            answer = int(fmap[chain[-1]])
            rest = [int(x) for x in torch.randperm(24, generator=g) if int(x) not in chain]
            for order_index, order in enumerate(ORDERS if kind != 'fixed_point' else (None,)):
                if order is None:
                    keys = torch.randperm(24, generator=g)
                else:
                    keys = torch.empty(24, dtype=torch.long); it = iter(rest)
                    assigned = {s: chain[j] for s, j in zip(SLOTS, order)}
                    for slot in range(24):
                        keys[slot] = assigned[slot] if slot in assigned else next(it)
                bindings = torch.stack((keys, fmap[keys]), -1).flatten()
                rows.append(torch.cat((bindings, torch.tensor([24, e, 28]))))
                answers.append(answer); orders.append(order_index); world_ids.append(world)
        out[kind] = (torch.stack(rows), torch.tensor(answers), torch.tensor(orders), torch.tensor(world_ids))
    return out


def controls():
    data = populations(torch); checks = {}
    for pop, (tokens, answers, orders, worlds) in data.items():
        masks = source_masks(tokens)
        checks[pop+'_counts'] = bool((masks['H'].sum((1, 2)) == (0 if pop == 'fixed_point' else 2)).all())
        checks[pop+'_disjoint'] = not bool((masks['H'] & masks['B']).any())
        checks[pop+'_query'] = bool((masks['C'].sum((1, 2)) == 3).all())
        if pop != 'fixed_point':
            for i, order in enumerate(ORDERS):
                expected_slot = max(SLOTS[order.index(1)], SLOTS[order.index(2)])
                checks[pop+'_order_'+str(i)] = bool(masks['H'][orders == i, 50, 2*expected_slot:2*expected_slot+2].all())
            checks[pop+'_same_world_all_orders'] = all(len(set(answers[worlds == w].tolist())) == 1 for w in range(32))
        else:
            checks['fixed_point_answer_identity'] = bool((tokens[:, 49] == answers).all())
    return {'passed': all(checks.values()), 'checks': checks}
