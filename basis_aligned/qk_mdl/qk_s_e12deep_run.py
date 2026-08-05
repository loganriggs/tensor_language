"""Deeper-narrowing exploration with shared values (scale box follow-up to
the E12 funnel family, per Logan's exploration-first directive).

E12b showed shared values recover most of the true-narrowing cost (208-dim
narrow: E12a +0.120 vs E12L, E12b only +0.036). This sweep pushes the
narrow stream further to find where the cost comes back: Dn = 156 = 26x6
(heads 3x52) and Dn = 104 = 26x4 (heads 2x52), both wide 264 (6x44) and
shared_values=True -- the winning ingredient held fixed, only sub_n moves
(8 -> 6 -> 4 dims per slot group).

argv[1] in {156, 104}; one per GPU, each writing its own JSON
(qk_e12_deep{N}.json) so nothing races; merged into qk_e12.json after.
Same box patches as qk_s_e12b_gpu0 (guard + oldheld caveat).
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import importlib
import sys

import torch

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

DN = int(sys.argv[1]) if __name__ == '__main__' and len(sys.argv) > 1 else 156
NHN = {156: 3, 104: 2}[DN]
SUB = DN // 26


def cfg_deep():
    return dict(Dw=264, NHw=6, HDw=44, Gw=24, Dn=DN, NHn=NHN, HDn=52,
                Gn=26, sub_n=SUB, control=False)


def make_deep():
    torch.manual_seed(Q.SEED)
    return M.FunnelRoute(f'E12b{DN}', cfg_deep(),
                         shared_values=True).to(E.DEV)


if __name__ == '__main__':
    M.JP = E.jpath(f'qk_e12_deep{DN}.json')
    E.setup()
    M.run_arm(f'qk_e12_b{DN}', f'E12b{DN}', make_deep,
              f'deeper shared-values narrowing: wide 264, narrow {DN} = '
              f'26 x {SUB} (heads {NHN}x52) -- where does the recovered '
              f'narrowing cost come back?',
              extra_pairs=(('qk_e12_b', 'e12b'), ('qk_e12_a', 'e12a')))
    print(f'e12 deep{DN} done', flush=True)
