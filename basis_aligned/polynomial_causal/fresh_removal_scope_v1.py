"""Descriptive cached intervention scope; ratios already inspected, not preregistered."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import torch

P = Path(__file__).resolve().parent


def main():
    start = time.monotonic()
    out = P / 'FRESH_REMOVAL_SCOPE_V1_RESULT.json'
    assert not out.exists()
    path = P / 'MINIMAX_FRESH_CACHE_V1_ARTIFACT.pt'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha == 'd26812841706bad2addc6d845948b0c7ba9415a6f372739d986b6bfa08382965'
    data = torch.load(path, weights_only=True)['measures'][24:].double()
    rows = json.loads((P / 'MINIMAX_FRESH_CACHE_V1_ROWS.json').read_text())['rows'][24:]
    families = []
    for family in range(4):
        part = rows[24*family:24*(family+1)]
        for a, b in zip(part[::2], part[1::2]):
            assert a['cue'] == 'British' and b['cue'] == 'American'
            assert (a['family'], a['concept'], a['variant']) == (b['family'], b['concept'], b['variant'])
        values = data[24*family:24*(family+1), :, 0]
        cue = values[::2] - values[1::2]
        native = cue[:, 0]
        assert native.norm() > 0
        effects = {}
        for column, name in [(1, 'child'), (2, 'remainder')]:
            change = native - cue[:, column]
            effects[name] = dict(cue_change=change.tolist(),
                                 norm_ratio=float(change.norm()/native.norm()),
                                 signed_projection=float(change.dot(native)/native.square().sum()),
                                 aligned_pairs=int((change*native > 0).sum()))
        families.append(dict(family=family+1, native_cue=native.tolist(), effects=effects))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), artifact_sha=sha,
                  families=families, seconds=time.monotonic()-start,
                  scope='Post hoc descriptive audit; values previously inspected. Native minus single-removal cue changes, not additive attribution fractions. No joint-parent removal is present. Existing controlled panels are development evidence, not corpus OOD. No fitting or new forwards.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
