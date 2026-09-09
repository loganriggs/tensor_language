#!/usr/bin/env python3
"""CPU export and fresh replay of the complete, explicitly native-background program."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path('/workspace/tensor_language')
POLY = ROOT/'basis_aligned/polynomial_causal'
sys.path[:0] = [str(ROOT), str(POLY)]
ARTIFACT = POLY/'CONTEXTUAL_HISTORY_V1_PROGRAM.pt'


def restore(path):
    """Loads only the exported artifact; the original checkpoint is not consulted."""
    import torch
    from deep_model import DeepModel
    from contextual_history_reference import HistoryReadout
    payload = torch.load(path, map_location='cpu', weights_only=True)
    native_shape = DeepModel(**payload['constructor']).double().eval()
    program = HistoryReadout(native_shape).eval()
    program.load_state_dict(payload['state_dict'], strict=True)
    buffers = dict(program.named_buffers())
    with torch.no_grad():
        for key, value in payload['all_buffers'].items():
            buffers[key].copy_(value)
    return program


def main():
    import torch
    from hop_ablate import load
    from hop_data import sample_docs
    import contextual_history_reference as R
    started = time.perf_counter()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    model, _ = load('attn-mlp-attn-rms-seed0')
    model = model.double().eval()
    program = R.HistoryReadout(model).eval()
    constructor = dict(n_vocab=29, d_model=128, n_head=4, spec=['attn','mlp','attn'],
                       n_ctx=240, norm='rms', residual='lerp', attention='bilinear', mlp_residual='add')
    torch.save({'constructor': constructor, 'state_dict': program.state_dict(),
                # Includes nonpersistent native rotary tables, preserving exact numeric semantics.
                'all_buffers': {k:v.detach().clone() for k,v in program.named_buffers()}}, ARTIFACT)
    tokens, _, _ = sample_docs(8, torch.Generator().manual_seed(5919))
    masks, _, _, _ = R.source_masks(tokens)
    arms = ((), ('H',), ('B',), ('H','B'), ('C',))
    with torch.inference_mode():
        expected = {}
        for arm in arms:
            kill = None if not arm else torch.stack([masks[n] for n in arm]).any(0)
            expected[str(arm)] = R.native_forward(model, tokens[:, :-1], kill).clone()
    del model, program
    restored = restore(ARTIFACT)
    records = {}
    with torch.inference_mode():
        for arm in arms:
            actual = restored(tokens[:, :-1], masks, arm)
            ref = expected[str(arm)]
            records[str(arm)] = dict(max_abs=float((actual-ref).abs().max()),
                                      passed=bool(torch.allclose(actual, ref, atol=1e-9, rtol=1e-9)))
    result = dict(passed=all(r['passed'] for r in records.values()), fresh_seed=5919, documents=8,
                  original_checkpoint_needed_for_restore=False, records=records,
                  parameters_and_fold=sum(p.numel() for p in restored.parameters())+restored.folded.numel(),
                  artifact_bytes=ARTIFACT.stat().st_size,
                  artifact_sha256=hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256((POLY/'contextual_history_reference.py').read_bytes()).hexdigest(),
                  seconds=time.perf_counter()-started,
                  scope='Complete full-output program with native prefix/routers retained and priced; generic folding saving only.')
    (POLY/'CONTEXTUAL_HISTORY_V1_EXPORT_REPLAY.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
