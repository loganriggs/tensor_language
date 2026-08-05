"""E12b runner (scale box, GPU 0) -- the gated fourth funnel arm, launched
after the E12b gate passed on this box: no divergence and no
token_starvation_flag among E12L (held100@2000 5.735), E12Lv (5.645), E12a
(5.789), all far under the 6.5 flag.

Same pattern as qk_s_e12a_gpu0.py: records go to qk_e12_b_gpu0.json so the
GPU-1 chain's read-modify-write merges on qk_e12.json can never race; keys
are merged into qk_e12.json after both processes exit. Same box patches
(guard + oldheld caveat).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import importlib

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

M = importlib.import_module('qk_e12_funnel_run')
M.JP = E.jpath('qk_e12_b_gpu0.json')
E.setup()
M.run_arm('qk_e12_b', 'E12b', M.make_e12b,
          'old shared-values narrowing arm (264 wide -> 208 narrow), '
          'kept because the GPU is free after everything',
          extra_pairs=(('qk_e12_a', 'e12a'),))
print('e12b gpu0 done', flush=True)
