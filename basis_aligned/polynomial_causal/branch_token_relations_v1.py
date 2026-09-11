"""Weights-only spelling-relation atlas for all frozen canonical branches.

A full-U metric and pair-difference replay <=1e-9, unique pairs, >=50 per relation.
B >=1 branch/relation: |mean(delta)|/RMS(delta)>=.5, sign consistency>=.8,
  difference energy / paired loading energy >=.1.
C >=2 distinct passing relations on branches from different parents.
No fitted factors, corpus, grammatical labels, or circuit identification.
"""
import hashlib
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    root = Path(__file__).parent
    output = root / 'BRANCH_TOKEN_RELATIONS_V1.json'
    assert not output.exists()
    source = root / 'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt'
    token_path = Path('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    binding = json.loads((root/'FROZEN_BRANCH_TENSE_V6_BINDING.json').read_text())['files']
    checkpoint = Path(next(p for p in binding if p.endswith('/pytorch_model.bin')))
    weights = torch.load(checkpoint, weights_only=True, mmap=True, map_location='cpu')
    unembedding = weights['lm_head.weight'].double()
    saved = torch.load(source, weights_only=True, map_location='cpu')
    tokenizer = Tokenizer.from_file(str(token_path))
    decoded = [tokenizer.decode([i]) for i in range(50257)]
    lookup = {s:i for i,s in enumerate(decoded) if re.fullmatch(r' [A-Za-z]+', s)}
    assert len(lookup) == sum(bool(re.fullmatch(r' [A-Za-z]+', s)) for s in decoded)
    transforms = {
        'add_s': lambda s:s+'s',
        'add_es': lambda s:s+'es',
        'y_to_ies': lambda s:s[:-1]+'ies' if s.endswith('y') else None,
        'add_ed': lambda s:s+'ed',
        'add_ing': lambda s:s+'ing',
        'initial_capital': lambda s:' '+s[1:].capitalize(),
    }
    pairs = {}
    for name, transform in transforms.items():
        edges = []
        for s,i in sorted(lookup.items(), key=lambda item:item[1]):
            if not re.fullmatch(r' [a-z]{3,}', s):
                continue
            t = transform(s)
            if t in lookup:
                edges.append([i, lookup[t]])
        assert len(set(map(tuple,edges))) == len(edges)
        pairs[name] = edges
    unit_writers, labels = [], []
    for parent,node in enumerate(saved['nodes']):
        for branch in range(node['writers'].shape[1]):
            w = node['writers'][:,branch]
            unit_writers.append(w/w.norm())
            labels.append(dict(parent=parent,branch=branch))
    metric_writers = torch.stack(unit_writers,1)
    physical = torch.linalg.solve_triangular(saved['output_whitener'],metric_writers,upper=True)
    loadings = unembedding@physical
    errors = [float((loadings.T@loadings-metric_writers.T@metric_writers).abs().max())]
    centered = loadings[:50257]-loadings[:50257].mean(0)
    cells = []
    for name, edges in pairs.items():
        ids = torch.tensor(edges,dtype=torch.long)
        assert ids.ndim==2 and ids.shape[1]==2
        before,after = loadings[ids[:,0]],loadings[ids[:,1]]
        delta = after-before
        direct = (unembedding[ids[:,1]]-unembedding[ids[:,0]])@physical
        errors.append(float((direct-delta).norm()/delta.norm()))
        errors.append(float(((centered[ids[:,1]]-centered[ids[:,0]])-delta).norm()/delta.norm()))
        for j,label in enumerate(labels):
            d = delta[:,j]
            mean,rms = float(d.mean()),float(d.square().mean().sqrt())
            coherence = abs(mean)/rms if rms else 0.
            consistency = float((d.sign()==(1 if mean>=0 else -1)).double().mean())
            contrast_fraction = float(d.square().sum()/(before[:,j].square()+after[:,j].square()).sum())
            passes = len(edges)>=50 and coherence>=.5 and consistency>=.8 and contrast_fraction>=.1
            def examples(indices):
                return [dict(before=decoded[edges[i][0]],after=decoded[edges[i][1]],
                             delta=float(d[i])) for i in indices]
            order = torch.argsort(d).tolist()
            cells.append(dict(**label,relation=name,pairs=len(edges),mean=mean,rms=rms,
                coherence=coherence,sign_consistency=consistency,contrast_energy_ratio=contrast_fraction,
                qualifies=passes,negative_examples=examples(order[:5]),positive_examples=examples(order[-5:][::-1])))
    finite = all(torch.isfinite(torch.tensor([c[k] for c in cells])).all()
                 for k in ('mean','rms','coherence','sign_consistency','contrast_energy_ratio'))
    a = bool(finite) and max(errors)<=1e-9 and all(len(e)>=50 for e in pairs.values())
    passing = [c for c in cells if c['qualifies']]
    c = any(x['relation']!=y['relation'] and x['parent']!=y['parent'] for x in passing for y in passing)
    result = dict(pred_a=a,pred_b=a and bool(passing),pred_c=a and c,replay_errors=errors,
        relations=pairs,branch_count=len(labels),cells=cells,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,token_path)},
        checkpoint_sha256=binding[str(checkpoint)],
        scope='All fixed branches and all mechanically matched spelling pairs. No fitting, text validation, '
        'morphological correctness, holdout, or causal/circuit claim. Writers have unit full-U norm; '
        'real-token centering changes no pair differences. Pair-energy ratios are descriptive, not '
        'fractions of full tensor energy or probabilities. Standalone parent banks must not be summed '
        'as a whole-graph replacement because shared-pair terms would be double-counted.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('relations','cells','sources')},indent=2))
    print(json.dumps(dict(pair_counts={k:len(v) for k,v in pairs.items()},
        passing=[{k:v for k,v in cell.items() if not k.endswith('examples')} for cell in passing]),indent=2))


if __name__=='__main__':
    main()
