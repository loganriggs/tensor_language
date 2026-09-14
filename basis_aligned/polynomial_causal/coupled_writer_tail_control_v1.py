"""Execute COUPLED_WRITER_TAIL_V1_PREREGISTRATION.md, CPU only.

A: native FP32 tail relative replay <=1e-4 on both contexts.
B: five-source derivative directional replay <=1e-5 on both contexts.
C: half step improves relative linearization error >=1.5x, or both <1e-8.
No compressed artifact or native-text claim. Exact tail weights remain charged.
"""
import hashlib
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from coupled_writer_tail_v1 import Tail, rms
from normalized_pair_ht_control_20260914_0256 import CHECKPOINT

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parents[1]))
from jacclust.tt_model import Block, GPTConfig, Rotary


def relative(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def native_blocks(sd, start=9, stop=17):
    config = GPTConfig(n_layer=18, n_head=9, n_embd=1152, bilinear=True,
                       squared_attn=True, bilinear_attn=True, gated=False)
    blocks = {}
    for layer in range(start, stop):
        with torch.device('meta'):
            block = Block(config)
        prefix = f'transformer.h.{layer}.'
        block.load_state_dict({k[len(prefix):]: v.float() for k, v in sd.items()
                               if k.startswith(prefix)}, assign=True, strict=True)
        block.attn.rotary = Rotary(128)
        blocks[layer] = block.eval()
    return blocks


def native_tail(blocks, sd, x, initial, inherited):
    x = x + blocks[9].mlp(F.rms_norm(x, (1152,)))
    for layer in range(10, 17):
        x, _ = blocks[layer](x, inherited, initial)
    coeff = sd['transformer.h.17.lambdas'].float()
    return coeff[0] * x + coeff[1] * initial


@torch.no_grad()
def main():
    assert not (P/'COUPLED_WRITER_TAIL_V1_RESULT.json').exists(), 'Preserve original receipt'
    torch.set_num_threads(2)
    start = time.monotonic()
    signal.alarm(120)
    sd = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
    blocks = native_blocks(sd)
    tail32 = Tail(sd, torch.float32)
    contexts, native_replays = [], []
    for seed in (170223000, 170223001):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(1, 5, 1152, dtype=torch.float64, generator=generator)
        initial = torch.randn(1, 5, 1152, dtype=torch.float64, generator=generator)
        child = torch.randn(5, dtype=torch.float64, generator=generator)
        remainder = torch.randn(5, dtype=torch.float64, generator=generator)
        initial32 = initial.float()
        inherited32 = F.rms_norm(initial32, (1152,)) @ sd['transformer.h.0.attn.c_v.weight'].float().T
        native = native_tail(blocks, sd, x.float(), initial32, inherited32)
        functional = tail32(x.float(), initial32, inherited32)
        error = relative(functional, native)
        native_replays.append({'seed': seed, 'relative_error': error,
                               'absolute_error': float((functional-native).norm())})
        contexts.append((seed, x, initial, child, remainder))
    pred_a = all(r['relative_error'] <= 1e-4 for r in native_replays)
    rows = []
    if pred_a:
        tail = Tail(sd)
        writer = torch.load(P/'MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt', weights_only=True)['direction'].double().reshape(1152)
        writer = writer / writer.square().mean().sqrt()
        h = 1e-4
        for seed, x, initial, child, remainder in contexts:
            inherited = rms(initial) @ sd['transformer.h.0.attn.c_v.weight'].double().T
            def execute(a):
                return tail(x + a[None, :, None] * writer, initial, inherited)
            baseline = execute(torch.zeros(5, dtype=torch.float64))
            directions = []
            for source in range(5):
                a = torch.zeros(5, dtype=torch.float64)
                a[source] = h
                directions.append(((execute(a) - execute(-a))/(2*h)).flatten())
            basis = torch.stack(directions)
            amplitude = child + remainder
            predicted = amplitude @ basis
            directional = ((execute(h*amplitude)-execute(-h*amplitude))/(2*h)).flatten()
            linearization = []
            for step in (1e-3, 5e-4):
                delta = (execute(step*amplitude)-baseline).flatten()
                approx = step*predicted
                linearization.append({'step': step, 'relative_error': relative(approx, delta),
                                      'absolute_error': float((approx-delta).norm()),
                                      'effect_norm': float(delta.norm())})
            large, small = (r['relative_error'] for r in linearization)
            rows.append({'seed': seed, 'directional_replay': relative(predicted, directional),
                         'singular_values': torch.linalg.svdvals(basis).tolist(),
                         'linearization': linearization,
                         'halving_improvement': large/max(small, 1e-30),
                         'pred_b': relative(predicted, directional) <= 1e-5,
                         'pred_c': large >= 1.5*small or max(large, small) < 1e-8})
            print(json.dumps(rows[-1]), flush=True)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'native_replays': native_replays,
              'contexts': rows, 'pred_a': pred_a,
              'pred_b': bool(rows) and all(r['pred_b'] for r in rows),
              'pred_c': bool(rows) and all(r['pred_c'] for r in rows),
              'seconds': time.monotonic()-start,
              'tail_loaded_parameter_bytes_fp32': sum(v.numel()*4 for v in tail32.weights.values()),
              'source_sha256': hashlib.sha256((P/'coupled_writer_tail_v1.py').read_bytes()).hexdigest(),
              'scope': 'Two synthetic backgrounds; actual fixed writer and weights; local tangent only. Full tail retained, no compression or native-text result.'}
    (P/'COUPLED_WRITER_TAIL_V1_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    signal.alarm(0)


if __name__ == '__main__':
    main()
