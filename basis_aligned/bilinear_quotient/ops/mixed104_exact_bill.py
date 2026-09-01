"""RUNG 295 (CPU): exact standalone storage bill for corrected mixed104.

This is deliberately a static dependency audit, not an object-size traversal of
the hook harness.  It distinguishes the physically tested artifact (which stores
the fp16 block-0 token-value table) from a storage-minimal semantic realization
(which recomputes the same value from the already-required embedding and native
block-0 c_v matrix).  Historical coverage/replacement ledgers are not totals.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path('/workspace/tensor_language/basis_aligned/bilinear_quotient')
MODEL = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240')
OUT = ROOT / 'mixed104_exact_bill_results.json'


def row(role, shape, multiplicity, dtype, source, note=''):
    n = multiplicity
    for dim in shape:
        n *= dim
    bytes_per = {'bfloat16': 2, 'float16': 2, 'float32': 4}[dtype]
    return {
        'role': role, 'shape': list(shape), 'multiplicity': multiplicity,
        'dtype': dtype, 'scalars': n, 'raw_bytes': n * bytes_per,
        'source': source, 'note': note,
    }


cfg = json.load(open(MODEL / 'config.json'))
cfg.pop('step', None)
assert cfg == {
    'vocab_size': 50304, 'n_layer': 18, 'n_head': 9, 'n_embd': 1152,
    'squared_mlp': False, 'bilinear': True, 'expansion_factor': 4,
    'gated': False, 'squared_attn': True, 'bilinear_attn': True,
}
D, HD, VMODEL, VTOK, LAYERS, HEADS = 1152, 128, 50304, 50257, 18, 9

# Metadata-only load: validates the literal checkpoint without allocating weights.
state = torch.load(MODEL / 'pytorch_model.bin', map_location='meta',
                   weights_only=True, mmap=True)
native_scalars = sum(v.numel() for v in state.values())
native_bytes = sum(v.numel() * v.element_size() for v in state.values())
assert len(state) == 218 and native_scalars == 545_902_902
assert state['transformer.wte.weight'].dtype == torch.bfloat16
assert state['lm_head.weight'].dtype == torch.float32
assert state['transformer.wte.weight'].shape == state['lm_head.weight'].shape
assert (state['transformer.wte.weight'].untyped_storage()._cdata !=
        state['lm_head.weight'].untyped_storage()._cdata)  # no tied storage

receipt = json.load(open(ROOT / 'mixed104_native_a1v_results.json'))
wanted = list(range(96)) + list(range(120, 128))
assert receipt['qk_singular_indices'] == wanted
assert receipt['physical_qk_factor_widths'] == [104]

motifs = json.load(open(ROOT / 'attn_motifs3_results.json'))['motif_table']
motif_pairs = {(li, hd) for li, hd, kind, _ in motifs
               if 2 <= li <= 9 and kind in ('prev', 'self')}
assert len(motif_pairs) == 38
tail_pairs = {(li, hd) for li in range(10, 18) for hd in range(HEADS)}
assert len(tail_pairs) == 72 and motif_pairs.isdisjoint(tail_pairs)
replaced_heads = len(motif_pairs | tail_pairs)
assert replaced_heads == 110

# Fail if the evaluated configuration changes under this bill.
wrapper = (ROOT / 'ops/mixed104_native_a1v.py').read_text()
for fragment in ("'cp_swap': 4608", "'qk_r': 96", "'qk_extra_tail': 8",
                 "'qk_tail': True", "'drop_tailE': True", "'drop_a1v': True"):
    assert fragment in wrapper, fragment
builder = (ROOT / 'ops/cevdump_ct96.py').read_text()
for fragment in ('tab=torch.zeros(V,D,device=DEV,dtype=torch.float16)',
                 "S['a0']=('cv',0,tab)",
                 "S[_nm]=('cp',_li", "S[f'c{li}']=('cp',li"):
    assert fragment in builder, fragment

factor_width = len(wanted)
native_qk_headmaps = (2 * HEADS + (8 * HEADS - len(motif_pairs))) * 4
factor_maps = replaced_heads * 4
assert native_qk_headmaps == 208 and factor_maps == 440

common = [
    row('input token embedding', (VMODEL, D), 1, 'bfloat16',
        'transformer.wte.weight'),
    row('output logit map', (VMODEL, D), 1, 'float32',
        'lm_head.weight', 'not tied to input embedding'),
    row('unreplaced dense Q/K head rows', (HD, D), native_qk_headmaps,
        'float32', 'blocks 0-1 all heads; blocks 2-9 non-motif heads'),
    row('mixed104 Q/K left factors', (HD, factor_width), factor_maps,
        'float32', 'SVD U[:,indices]*sigma'),
    row('mixed104 Q/K right factors', (factor_width, D), factor_maps,
        'float32', 'SVD Vh[indices,:]'),
    row('native value matrices, blocks 1-17', (D, D), 17, 'float32',
        'attn.c_v.weight'),
    row('attention output matrices', (D, D), LAYERS, 'float32',
        'attn.c_proj.weight'),
    row('attention value-mix scalars', (), LAYERS, 'bfloat16',
        'attn.lamb'),
    row('bilinear MLP Left/Right/Down matrices', (D, 4 * D), 3 * LAYERS,
        'float32', 'blocks 0-9 CP full width; blocks 10-17 native',
        'CP hooks reorder exact native units and do not compress storage'),
    row('bilinear MLP output biases', (D,), LAYERS, 'bfloat16',
        'mlp.Down_bias'),
    row('block residual mixing scalars', (2,), LAYERS, 'bfloat16',
        'block.lambdas'),
]

artifact = common + [
    row('block-0 token-value lookup', (VTOK, D), 1, 'float16',
        "S['a0']", 'physically tested table; valid GPT-2 input ids 0..50256'),
]
minimal = common + [
    row('block-0 native value matrix', (D, D), 1, 'float32',
        'transformer.h.0.attn.c_v.weight',
        'recomputes the table from the already-required normalized embedding'),
]

def totals(rows):
    return {'scalars': sum(r['scalars'] for r in rows),
            'raw_bytes': sum(r['raw_bytes'] for r in rows)}


artifact_total = totals(artifact)
minimal_total = totals(minimal)
assert artifact_total == {'scalars': 596_164_022, 'raw_bytes': 2_152_921_964}
assert minimal_total == {'scalars': 539_595_062, 'raw_bytes': 2_042_438_252}

# Linear-map MACs only. Attention mixing is sequence-dependent and stated separately.
artifact_linear_macs_per_token = sum(r['scalars'] for r in artifact if r['role'] in {
    'output logit map', 'unreplaced dense Q/K head rows',
    'mixed104 Q/K left factors', 'mixed104 Q/K right factors',
    'native value matrices, blocks 1-17', 'attention output matrices',
    'bilinear MLP Left/Right/Down matrices',
})
minimal_linear_macs_per_token = artifact_linear_macs_per_token + D * D

result = {
    'decision': 'historical_180m_and_123.4m_totals_rejected',
    'reason': ('the historical ledger priced replacement coverage, while the standalone '
               'dependency graph retains embeddings, output map, all MLP weights, values, '
               'output projections, and unreplaced Q/K rows'),
    'model_config': cfg,
    'source_assertions': {
        'checkpoint_state_entries': len(state),
        'motif_heads': len(motif_pairs), 'tail_heads': len(tail_pairs),
        'replaced_heads': replaced_heads, 'factor_maps': factor_maps,
        'factor_width': factor_width, 'native_qk_headmaps': native_qk_headmaps,
        'drop_a1v': True, 'drop_tailE': True, 'cp_full_width': 4608,
    },
    'native_checkpoint': {
        'scalars': native_scalars, 'raw_tensor_bytes': native_bytes,
    },
    'tested_artifact_literal': {
        'manifest': artifact, **artifact_total,
        'delta_scalars_vs_native': artifact_total['scalars'] - native_scalars,
        'delta_raw_bytes_vs_native': artifact_total['raw_bytes'] - native_bytes,
        'linear_map_macs_per_token': artifact_linear_macs_per_token,
    },
    'storage_minimal_semantic_candidate': {
        'manifest': minimal, **minimal_total,
        'delta_scalars_vs_native': minimal_total['scalars'] - native_scalars,
        'delta_raw_bytes_vs_native': minimal_total['raw_bytes'] - native_bytes,
        'linear_map_macs_per_token': minimal_linear_macs_per_token,
        'status': 'requires live a0-table versus native-c_v0 equivalence gate',
    },
    'sequence_dependent_compute': {
        'causal_pairs_for_length_T': 'T*(T+1)/2',
        'attention_score_and_value_MACs_per_causal_pair': 3 * LAYERS * HEADS * HD,
        'explanation': 'two 128-d QK dot products plus one 128-d value accumulation per head',
    },
    'excluded_nonsemantic_harness_state': [
        'full overwritten native Q/K rows', 'overwritten native block-0 c_v output',
        'fit-row captures', 'class probes and dictionaries', 'SVD workspaces',
        'census rows and cached logits', 'hook objects',
    ],
    'behavior_receipt': {
        'census_damage': receipt['census_damage'],
        'certificates_valid': receipt['certificates_valid'],
        'max_fresh_damage': receipt['max_fresh_damage'],
    },
}

OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({
    'native': result['native_checkpoint'],
    'tested_artifact_literal': {k: artifact_total[k] for k in artifact_total},
    'storage_minimal_semantic_candidate': {k: minimal_total[k] for k in minimal_total},
    'decision': result['decision'],
}, indent=2))
