"""Parallel E12a runner (scale box, GPU 0) -- splits the handed-off E12
funnel chain across both GPUs per Logan's exploration-first directive.

Runs ONLY the E12a arm (wide 264 -> narrow 208 true-narrowing price), with
records going to qk_e12_a_gpu0.json so the GPU-1 chain's read-modify-write
merges on qk_e12.json can never race; keys are merged into qk_e12.json after
both processes exit. Checkpoint stem stays qk_e12_a (shared namespace, no
collision -- the GPU-1 chain is killed after E12Lv, before its own E12a).
Same box patches as qk_s_e12_launch (guard + oldheld caveat).
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
M.JP = E.jpath('qk_e12_a_gpu0.json')
E.setup()
M.run_arm('qk_e12_a', 'E12a', M.make_e12a,
          'funnel narrowing arm: wide 264, narrow 208 = 26 x 8 (heads '
          '4x52) -- the additional price of true narrowing',
          extra_pairs=(('qk_e12_L', 'e12L'),))
print('e12a gpu0 done', flush=True)
