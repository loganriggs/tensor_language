"""ER: FRESH-DATA REPLICATIONS of the readout family (Logan spec update two;
fresh single-epoch batch-16 protocol, see qk_e_common).

The V10/V11/V13 readout results were measured under the 6-epoch cooc
convention; this runner replicates the four informative arms on the fresh
single-epoch stream so their CE ordering can be compared against the fresh
controls without any memorization confound:

  V10    slots + group-lasso + N=6 window for blocks; readout reads ALL 24
         module writes (no embedding at the readout)
  V11    V10 + full affine decoder per module into the readout, decoder lasso
  V11nl  V11 without the decoder lasso
  V13r1  slots + window-6 + rank-1 per-edge read adapters (114 edges)

Skipped as redundant per the directive: V11lr rank-32 (sits between V11nl and
V13) and V13 rank-4 (tied rank-1 in the 6-epoch run). Model classes are reused
VERBATIM from qk_v10v11_common / qk_v13_common -- only the data / batch / lr
plumbing changes (family lr from the E0b sweep, identical epoch_order(0) data
order). Deltas paired vs E0a and E0b on the fresh held set, plus the old cooc
held evaluation per arm.

Positive controls: the full qk_v10v11_common.run_controls() block (matched
control == vanilla-A at init, penalty vectorization, V10 visibility table,
V10/V11 identity controls, V11 init == V10 init) and qk_v13_common
.v13_controls() (114 edges, V13 init == slots+window init, zero adapter
penalty at init). Results -> qk_er.json. Idempotent."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import qk_e_common as E                          # must import FIRST (smoke patch)
from qk_e_common import Q, W, torch
import qk_v13_common as X

JP = E.jpath('qk_er.json')

ARMS = (
    ('qk_er_v10', 'V10', W.make_v10, 'v10'),
    ('qk_er_v11', 'V11',
     lambda: W.make_v11('V11', dec_lasso=True, cls=W.V11Route), 'v11'),
    ('qk_er_v11nl', 'V11nl',
     lambda: W.make_v11('V11nl', dec_lasso=False, cls=W.V11Route), 'v11'),
    ('qk_er_v13r1', 'V13r1', lambda: X.make_v13('V13r1', 1), 'v13'),
)


if __name__ == '__main__':
    E.setup()
    W.run_controls()
    X.v13_controls()

    counts = {}
    for stem, key, factory, _ in ARMS:
        m = factory()
        pc = W.param_counts(m)
        if hasattr(m, 'ad_U'):
            pc['adapter_params'] = sum(p.numel() for p in m.ad_U) \
                + sum(p.numel() for p in m.ad_V)
        counts[key] = pc
        del m
        torch.cuda.empty_cache()
    E.merge(JP, 'param_counts', counts)

    for stem, key, factory, _ in ARMS:
        E.train_arm(stem, JP, key, factory, W.GC,
                    extra={'replication_of': f'{key} (6-epoch cooc original)'})
        E.oldheld_record(stem, factory, JP, f'{key}_oldheld')
        E.paired_fresh(stem, JP, key)
    print('er readout replication run done', flush=True)
