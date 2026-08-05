"""Scale-box launcher for the handed-off E12 funnel chain (local -> scale,
2026-08-05). Two box-specific patches, then runs qk_e12_funnel_run.py as
__main__ unchanged:

1. Q.gpu_guard neutered BEFORE the qk_e_common import chain -- the probe
   module calls it at import time and reads nvidia-smi's first line (physical
   GPU 0) regardless of CUDA_VISIBLE_DEVICES, which deadlocks when GPU 0 is
   busy with a scale arm (bug already documented in qk_s_gate_run).
2. E.oldheld_record wrapped to stamp every old-held record with the
   substitute-corpus caveat: data_fineweb_cooc_tokens.npy on this box is
   fresh34k rows [0:6000] (pure eval, so the numbers are valid fresh-data
   evals) but they are NOT comparable to the original session's cooc numbers.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import runpy

import qk_tokenline_train as Q
Q.gpu_guard = lambda *a, **k: None

import qk_e_common as E

_orig_oldheld = E.oldheld_record


def oldheld_flagged(stem, factory, jp, key):
    rec = _orig_oldheld(stem, factory, jp, key)
    if rec is not None:
        rec = dict(rec)
        rec['CAVEAT'] = ('scale box: old-held corpus is a SUBSTITUTE '
                         '(fresh34k[0:6000], pure eval) -- valid fresh eval, '
                         'NOT comparable to cooc old-held numbers')
        E.merge(jp, key, rec)
    return rec


E.oldheld_record = oldheld_flagged

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'qk_e12_funnel_run.py'), run_name='__main__')
