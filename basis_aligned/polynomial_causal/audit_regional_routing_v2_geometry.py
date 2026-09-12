"""CPU diagnostic: best scalar cannot repair directional routing-write error.

No learned gain is adopted. Also count exact causal cue-prefix classes, without
claiming those prefixes suffice to generate downstream queries or writers.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

P = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows_path = P / 'REGIONAL_BEHAVIOR_CONTROLS_V2_ROWS.json'
    rows = json.loads(rows_path.read_text())['rows']
    cells = []
    bindings = {rows_path.name: digest(rows_path)}
    for name in ['TOKEN', 'ANCHORED']:
        path = P / f'REGIONAL_{name}_ROUTING_V2_ARTIFACT.pt'
        bindings[path.name] = digest(path)
        writes = torch.load(path, weights_only=True, map_location='cpu')['write_vertices'].double()
        assert writes.shape == (len(rows), 5, 1152)
        delta = writes - writes[:, :1]
        for family in [None, 0, 1, 2, 3]:
            ids = [i for i, row in enumerate(rows) if family is None or row['family'] == family]
            ref = delta[ids, 1].flatten()
            for arm in [2, 3, 4]:
                vec = delta[ids, arm].flatten()
                assert ref.norm() > 0 and vec.norm() > 0
                cosine = torch.dot(vec, ref) / (vec.norm() * ref.norm())
                gain = torch.dot(vec, ref) / torch.dot(vec, vec)
                error = (gain * vec - ref).norm() / ref.norm()
                # Orthogonal-projection identity independently checks the diagnostic.
                expected = (1 - cosine.square()).clamp_min(0).sqrt()
                assert abs(float(error - expected)) < 1e-12
                cells.append(dict(method=name, family=family, arm=arm,
                                  cosine=float(cosine), norm_ratio=float(vec.norm()/ref.norm()),
                                  diagnostic_best_scalar=float(gain), minimum_scalar_error=float(error),
                                  meets_original_10pct_write_bar=bool(error <= .1)))
    panels = []
    for stem in ['REGIONAL_SOURCE_BLOCK_V2', 'REGIONAL_SOURCE_BLOCK_OOD_V1', 'REGIONAL_BEHAVIOR_CONTROLS_V2']:
        path = P / (stem + '_ROWS.json')
        panel = json.loads(path.read_text())['rows']
        bindings[path.name] = digest(path)
        prefixes = Counter()
        for i in range(0, len(panel), 2):
            left, right = panel[i]['ids'], panel[i+1]['ids']
            assert len(left) == len(right)
            changed = [j for j, (a, b) in enumerate(zip(left, right)) if a != b]
            assert len(changed) == 1
            pos = changed[0]
            prefixes[tuple(left[:pos+1])] += 1
            prefixes[tuple(right[:pos+1])] += 1
        panels.append(dict(panel=stem, rows=len(panel), unique_cue_prefixes=len(prefixes),
                           prefixes=[dict(ids=list(k), length=len(k), count=v) for k,v in sorted(prefixes.items())]))
    result = dict(cells=cells, panels=panels, source_sha=digest(Path(__file__)), bindings=bindings,
                  scope='Diagnostic scalar optimum only; no fitted replacement. Prefix equivalence follows from causal attention, but native key replay under prefix truncation is not tested here. Full prompt remains necessary for queries/downstream context.')
    out = P / 'REGIONAL_ROUTING_V2_GEOMETRY_RESULT.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(global_cells=[c for c in cells if c['family'] is None],
                          panels=[{k:v for k,v in p.items() if k != 'prefixes'} for p in panels]), indent=2))


if __name__ == '__main__':
    main()
