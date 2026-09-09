"""Valid permutation column interventions test matching of the middle entity."""
import torch
from causal_suffix_join_reference import ORDERS, SLOTS
from suffix_join_writer_reference import edge_masks

CASES = ('base', 'keys', 'values', 'both')


def populations():
    out = {}
    for pop, seed in (('iid', 14909), ('ood_two_12cycles', 14910)):
        g = torch.Generator().manual_seed(seed)
        rows, metadata, masks = [], [], []
        for world in range(16):
            perm = torch.randperm(24, generator=g); fmap = torch.empty(24, dtype=torch.long)
            if pop == 'iid':
                fmap[perm] = perm.roll(-1)
            else:
                fmap[perm[:12]] = perm[:12].roll(-1); fmap[perm[12:]] = perm[12:].roll(-1)
            a = int(torch.randint(24, (), generator=g)); b = int(fmap[a]); c = int(fmap[b]); d = int(fmap[c])
            inv = torch.empty_like(fmap).scatter_(0, fmap, torch.arange(24))
            excluded = {c}; left = right = c
            for _ in range(3):
                left, right = int(inv[left]), int(fmap[right]); excluded.update((left, right))
            legal = [t for t in range(24) if t not in excluded]
            foil = legal[int(torch.randint(len(legal), (), generator=g))]
            rename = torch.arange(24); rename[c] = foil; rename[foil] = c
            rest = [int(x) for x in torch.randperm(24, generator=g) if int(x) not in (a, b, c)]
            for oi, order in enumerate(ORDERS):
                assigned = {s: (a, b, c)[j] for s, j in zip(SLOTS, order)}; it = iter(rest)
                keys = torch.tensor([assigned[s] if s in assigned else next(it) for s in range(24)])
                values = fmap[keys]
                base = torch.cat((torch.stack((keys, values), -1).flatten(), torch.tensor([24, a, 28])))
                mask, _, _, orientation = edge_masks(base[None])
                for case in CASES:
                    changed_keys = rename[keys] if case in ('keys', 'both') else keys
                    changed_values = rename[values] if case in ('values', 'both') else values
                    current = torch.empty_like(fmap).scatter_(0, changed_keys, changed_values)
                    answer = int(current[current[current[a]]])
                    tokens = torch.cat((torch.stack((changed_keys, changed_values), -1).flatten(), base[48:]))
                    rows.append(tokens); masks.append(mask[0])
                    metadata.append({'world': world, 'order': oi, 'case': case, 'orientation': orientation[0],
                                     'entity': a, 'middle': c, 'foil': foil, 'answer': answer,
                                     'expected_answer': d if case in ('base', 'both') else int(fmap[foil])})
        out[pop] = torch.stack(rows), torch.stack(masks), metadata
    return out


def controls():
    checks = {}
    for pop, (tokens, masks, meta) in populations().items():
        checks[pop+'_unique_keys_values'] = bool((tokens[:, :48:2].sort(1).values == torch.arange(24)).all() and
                                                 (tokens[:, 1:48:2].sort(1).values == torch.arange(24)).all())
        checks[pop+'_answers'] = all(m['answer'] == m['expected_answer'] for m in meta)
        checks[pop+'_same_mask_four_cases'] = all(torch.equal(masks[i:i+4], masks[i:i+1].expand_as(masks[i:i+4])) for i in range(0, len(tokens), 4))
        checks[pop+'_query_unchanged'] = bool((tokens.reshape(-1, 4, 51)[:, :, 48:] == tokens.reshape(-1, 4, 51)[:, :1, 48:]).all())
        matched = True; conjugate = True; min_cycle = 24
        for i, (tok, m) in enumerate(zip(tokens, meta)):
            before = tokens[i-i%4]; c, foil = m['middle'], m['foil']
            rename = torch.arange(24); rename[c] = foil; rename[foil] = c
            if m['case'] == 'both':
                conjugate &= torch.equal(tok[:48], rename[before[:48]])
            base_keys = before[:48:2]; base_vals = before[1:48:2]
            bpos = int(torch.where(base_vals == c)[0][0])*2+1
            cpos = int(torch.where(base_keys == c)[0][0])*2
            matched &= bool(tok[bpos] == tok[cpos]) == (m['case'] in ('base', 'both'))
            f = torch.empty(24, dtype=torch.long).scatter_(0, tok[:48:2], tok[1:48:2])
            for start in range(24):
                cur = int(f[start]); length = 1
                while cur != start:
                    cur = int(f[cur]); length += 1
                    assert length <= 24
                min_cycle = min(min_cycle, length)
        checks[pop+'_conjugation'] = conjugate
        checks[pop+'_middle_truth_table'] = matched
        checks[pop+'_no_short_cycle'] = min_cycle >= 4
    return {'passed': all(checks.values()), 'checks': checks}
