"""Fork cold lookup queries and derive every answer from the emitted tokens."""
import torch


def fork(tokens, metadata, hops=(0, 1, 2, 3), n_entities=24):
    assert tokens.ndim == 2 and tokens.shape[1] == 2*n_entities+3
    assert len(tokens) == len(metadata) and hops and all(h in range(4) for h in hops)
    keys, values = tokens[:, :2*n_entities:2], tokens[:, 1:2*n_entities:2]
    assert bool((keys.sort(1).values == torch.arange(n_entities, device=tokens.device)).all())
    assert bool(((values >= 0) & (values < n_entities)).all())
    fmap = torch.empty_like(keys).scatter_(1, keys, values)
    answer = tokens[:, -2].clone(); answers = [answer]
    for _ in range(3):
        answer = fmap.gather(1, answer[:, None]).squeeze(1); answers.append(answer)
    out = tokens.repeat_interleave(len(hops), dim=0)
    out[:, -1] = torch.tensor(hops, device=tokens.device).repeat(len(tokens))+n_entities+1
    meta = [{**m, 'hop': h, 'answer': int(answers[h][i]),
             'expected_answer': int(answers[h][i])}
            for i, m in enumerate(metadata) for h in hops]
    return out, meta


def controls():
    tokens = torch.tensor([[2, 0, 0, 1, 1, 2, 3, 0, 7]])
    original = tokens.clone(); out, meta = fork(tokens, [{'answer': 99}], n_entities=3)
    checks = {'cycle_answers': [m['answer'] for m in meta] == [0, 1, 2, 0],
              'expected_answers': all(m['answer'] == m['expected_answer'] for m in meta),
              'query_tokens': out[:, -1].tolist() == [4, 5, 6, 7],
              'prefix_preserved': torch.equal(out[:, :-1], tokens[:, :-1].expand(4, -1)),
              'input_immutable': torch.equal(tokens, original)}
    return {'passed': all(checks.values()), 'checks': checks}
