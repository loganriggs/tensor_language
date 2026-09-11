"""Sum frozen source read-edge edits at one query; query roles stay fixed."""
import json
from pathlib import Path
import torch
from compiled_qk_vector_edit_v1 import compile_vector, predict_vector
from shared_position_qk_edit_v1 import prepare_from_head_vectors
from folded_normalized_router_v1 import direct


def predict_source_sum(compiled, source, query_heads, key_heads, full_value,
                       query_position, source_positions, edits):
    """Inputs [batch,source,...]; one [rank,rank] edit per source.

    Return the attention head output change, before c_proj. Query vectors,
    upstream source normalization and the base-value stream are background.
    No softmax or cross-source normalization exists in this model.
    """
    if source.shape[1] != len(source_positions) or len(edits) != len(source_positions):
        raise ValueError('Source/position/edit counts differ')
    if any(s > query_position or s < 0 for s in source_positions):
        raise ValueError('Only causal source positions are supported')
    if len(set(source_positions)) != len(source_positions):
        raise ValueError('Duplicate source positions would double count')
    delta = torch.zeros_like(full_value[:, 0])
    for j, (position, edit) in enumerate(zip(source_positions, edits)):
        ports = prepare_from_head_vectors(
            compiled, source[:, j], query_heads, [k[:, j] for k in key_heads],
            full_value[:, j], query_position, position)
        baseline = predict_vector(ports, compiled, torch.zeros_like(edit))
        changed = predict_vector(ports, compiled, edit)
        delta = delta + changed['contribution'] - baseline['contribution']
    return delta


def control():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1327)
    batch, count, width, dim, rank = 7, 6, 4, 9, 3
    weights = [torch.randn(width, dim) for _ in range(4)]
    basis = torch.linalg.qr(torch.randn(dim, rank)).Q
    value_weight = torch.randn(width, dim)
    mix = -.0888671875
    source = torch.randn(batch, count, dim)
    query = source[:, -1].clone()  # Self key/value may change; this query stays fixed.
    base = torch.randn(batch, count, width)
    value = (1-mix)*(source @ value_weight.T) + mix*base
    qh = [query @ weights[j].T for j in [0, 2]]
    kh = [source @ weights[j].T for j in [1, 3]]
    compiled = compile_vector(weights, basis, value_weight, mix)
    positions = list(range(count))
    edits = [torch.eye(rank) if j % 2 else .2*torch.randn(rank, rank)
             for j in positions]
    observed = predict_source_sum(compiled, source, qh, kh, value,
                                  count-1, positions, edits)
    expected = torch.zeros_like(observed)
    for j, edit in enumerate(edits):
        edited = source[:, j] - (source[:, j] @ basis @ edit.T) @ basis.T
        old = direct(weights, query, source[:, j], count-1, j)[:, None] * value[:, j]
        new_value = (1-mix)*(edited @ value_weight.T) + mix*base[:, j]
        new = direct(weights, query, edited, count-1, j)[:, None] * new_value
        expected += new-old
    zero = torch.zeros(rank, rank)
    even = [edit if j % 2 == 0 else zero for j, edit in enumerate(edits)]
    odd = [edit if j % 2 else zero for j, edit in enumerate(edits)]
    def run(e):
        return predict_source_sum(compiled, source, qh, kh, value, count-1, positions, e)
    replay = float((observed-expected).norm()/expected.norm())
    composition = float((run(even)+run(odd)-observed).norm()/observed.norm())
    zero_error = float(run([zero]*count).abs().max())
    rejected = False
    try:
        predict_source_sum(compiled, source, qh, kh, value, count-2, positions, edits)
    except ValueError:
        rejected = True
    result = dict(instrument_passed=max(replay, composition, zero_error)<1e-10 and rejected,
                  direct_replay_relative_error=replay,
                  disjoint_source_composition_relative_error=composition,
                  zero_edit_maximum=zero_error, noncausal_rejected=rejected,
                  self_key_value_read_included=True, query_role_held_fixed=True,
                  corpus_access=False, native_model_forwards=0,
                  scope='CPU algebra control. Sum at attention output only; no CE additivity, semantic circuit or native all-source validation claim.')
    out = Path(__file__).with_name('ALL_SOURCE_QK_EDIT_V1_CONTROL.json')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
    assert result['instrument_passed']


if __name__ == '__main__':
    control()
