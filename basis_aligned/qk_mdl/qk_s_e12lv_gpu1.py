"""GPU-1 restart runner for the E12 funnel chain (scale box).

The original chain (qk_s_e12_launch) crashed after E12L's training finished
and its checkpoint saved: funnel_light_probe/_ce_with pre-sliced Q.HELD to
Q.T columns and then took targets b[:, 1:Q.T+1], leaving Q.T-1 targets vs
Q.T logit positions (now fixed in qk_e12_funnel_run.py). This runner resumes:
E12L skips training via its checkpoint and completes the probe/pair/oldheld
records, then E12Lv runs in full. E12a is NOT run here -- it trains in
parallel on GPU 0 (qk_s_e12a_gpu0.py, own JSON); E12b is gated and launched
separately after both GPUs finish. Same box patches as qk_s_e12_launch.
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
E.setup()
M.run_arm('qk_e12_L', 'E12L', M.make_e12L,
          'funnel PRIMARY: wide 384 (6x64) detokenization, narrow 286 = '
          '26 slots x 11 (E9a-matched message bandwidth), heads 11x26')
M.run_arm('qk_e12_Lv', 'E12Lv', M.make_e12Lv,
          'E12L + shared values from block-0 wide c_v via P_sv (384->286, '
          '11x26)', extra_pairs=(('qk_e12_L', 'e12L'),))
print('e12 L+Lv gpu1 done', flush=True)
