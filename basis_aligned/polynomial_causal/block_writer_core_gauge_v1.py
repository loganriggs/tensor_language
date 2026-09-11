"""CPU-only same-function audit of the four-output block energy penalty.

Predictions registered on AGENT_BOARD.md before execution: coefficient replay
and attainment <=1e-10; native penalty reduction >=1%. No nonlinear fitting.
"""
import hashlib
import json
import time
from pathlib import Path

import torch


def balance(writer, core, output):
    """Full-rank QR implementation; fail closed on ill-conditioned writers."""
    folded = output @ writer
    _, rf = torch.linalg.qr(folded, mode="reduced")
    qc, rc = torch.linalg.qr(core, mode="reduced")
    condition = float(torch.linalg.cond(rf))
    assert condition < 1e10
    left, singular, right = torch.linalg.svd(rf @ rc.T)
    hadamard = writer.new_tensor([[1, 1, 1, 1], [1, -1, 1, -1],
                                 [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
    transform = torch.linalg.solve(rf, left * singular.sqrt()[None, :]) @ hadamard
    new_writer = writer @ transform
    new_core = (qc @ right.T * singular.sqrt()[None, :]) @ hadamard
    scale = new_core.norm(dim=0)
    new_core = new_core / scale
    new_writer = new_writer * scale
    original = writer @ core.T
    replay = float((new_writer @ new_core.T - original).norm() / original.norm())
    old_penalty = float((folded.square().sum(0) * core.square().sum(0)).sum())
    new_penalty = float(((output @ new_writer).square().sum(0)
                         * new_core.square().sum(0)).sum())
    lower_bound = float(singular.sum().square() / 4)
    report = dict(replay_relative_error=replay, writer_qr_condition=condition,
                  old_penalty=old_penalty, balanced_penalty=new_penalty,
                  exact_minimum=lower_bound,
                  minimum_relative_error=abs(new_penalty-lower_bound)/lower_bound,
                  singular_values=singular.tolist())
    return new_writer, new_core, report


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    assert not torch.cuda.is_initialized()
    torch.manual_seed(529)
    _, _, control = balance(torch.randn(12, 4), torch.randn(9, 4), torch.randn(17, 12))
    assert control['replay_relative_error'] < 1e-10
    assert control['minimum_relative_error'] < 1e-10
    # A unit rank-one function split into m identical terms costs 1/m.
    duplication_control = {str(m): m * (1 / m) ** 2 for m in (1, 2, 4)}
    poly = Path(__file__).resolve().parent
    receipt = json.loads((poly / 'BLOCK_TRUST_REGION_THIN_V1_RESULT.json').read_text())
    cache = Path(receipt['cache']['path'])
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == receipt['cache']['sha256']
    saved = torch.load(cache, map_location='cpu', weights_only=False)
    ckpt = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
    native = torch.load(ckpt, map_location='cpu', weights_only=True, mmap=True)
    output = native['lm_head.weight'].double()
    writer, core, bank = saved['writer'].double(), saved['packed'].double(), saved['bank'].double()
    assert writer.shape == (1152, 64) and core.shape == (136, 64)
    orthogonality = float((bank.transpose(-1, -2) @ bank - torch.eye(16)).abs().max())
    assert orthogonality < 1e-10
    rows = []
    for group in range(16):
        sl = slice(4 * group, 4 * group + 4)
        _, _, row = balance(writer[:, sl], core[:, sl], output)
        row['block'] = group
        rows.append(row)
    old = sum(row['old_penalty'] for row in rows)
    new = sum(row['balanced_penalty'] for row in rows)
    reduction = 1-new/old
    predicted_change = .01 * receipt['final']['component_energy'] * (new/old-1)
    result = dict(predictions={
        'pred_a_same_function': max(r['replay_relative_error'] for r in rows) <= 1e-10,
        'pred_b_minimum_attained': max(r['minimum_relative_error'] for r in rows) <= 1e-10,
        'pred_c_one_percent_penalty_reduction': reduction >= .01},
        control=control, rank_one_duplication_penalties=duplication_control,
        blocks=rows, orthogonality_error=orthogonality,
        native_penalty_fraction_removed=reduction,
        predicted_normalized_objective_change=predicted_change,
        normalization_source='Existing native receipt; no independent full-target objective replay',
        source_cache_sha256=receipt['cache']['sha256'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seconds=time.perf_counter()-start, corpus_access=False, gpu_access=False,
        scope='Exact fixed-block gauge minimum, not a new computation or global fit. No checkpoint mutation.')
    assert result['predictions']['pred_a_same_function'] and result['predictions']['pred_b_minimum_attained']
    with (poly / 'BLOCK_WRITER_CORE_GAUGE_V1_AUDIT.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('blocks', 'control')}, indent=2))


if __name__ == '__main__':
    main()
