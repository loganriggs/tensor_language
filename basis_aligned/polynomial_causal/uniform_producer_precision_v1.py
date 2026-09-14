"""Preregistered board precision check: two original16-row batches, <=1% bars.

Compare FP64 suffix and native attention-output injection order independently.
No fitting, law change or substitution of this audit for the original ranking.
"""
import json
import os
import signal
import time
from datetime import datetime, timezone
import torch
from uniform_producer_context_v1 import UniformProducer
from retained_objective_context_v1 import P, branch_errors
from pair_ranking_v1 import NAMES


@torch.no_grad()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    out = P/'UNIFORM_PRODUCER_PRECISION_V1_RESULT.json'
    assert not out.exists()
    torch.set_num_threads(2)
    signal.alarm(120)
    start = time.monotonic()
    model = UniformProducer()
    operators = [model.tensor] + [model.decode(n)-model.tensor for n in NAMES]
    rows = []
    for seed in (170224000, 170224001):
        baseline = None
        for arm, kwargs in [('baseline', {}), ('fp64_suffix', {'suffix_fp64': True}),
                            ('native_injection_site', {'native_site': True})]:
            z, terms, denominator, diagnostics = model.sample(seed, **kwargs)
            vectors = [branch_errors(op, z, terms, denominator).sum(1) for op in operators]
            if baseline is None:
                baseline = vectors
            errors = [float((v-b).norm()/b.norm().clamp_min(1e-30)) for v, b in zip(vectors, baseline)]
            rows.append(dict(seed=seed, arm=arm,
                             vector_relative_changes=dict(zip(['mixed_reference']+NAMES, errors)),
                             vector_norms=[float(v.norm()) for v in vectors],
                             passes=all(e<=.01 for e in errors), diagnostics=diagnostics))
    result = dict(utc=datetime.now(timezone.utc).isoformat(), rows=rows,
                  pred_fp64=all(r['passes'] for r in rows if r['arm']=='fp64_suffix'),
                  pred_site=all(r['passes'] for r in rows if r['arm']=='native_injection_site'),
                  seconds=time.monotonic()-start,
                  scope='Two original batches; suffix precision and injection-order audit only, upstream hierarchy remains FP32.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    signal.alarm(0)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
