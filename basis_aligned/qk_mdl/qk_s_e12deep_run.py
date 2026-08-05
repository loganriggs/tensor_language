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

ARM = sys.argv[1] if __name__ == '__main__' and len(sys.argv) > 1 else '156'
if ARM == 'w480n208':
    # wide-axis extension: 264 -> 384 bought -0.0345 at 208 narrow; does 480
    # keep buying or saturate? (480 = 24 x 20 wide slot groups, heads 6x80)
    DW, NHW, HDW = 480, 6, 80
    DN, NHN = 208, 4
    NAME, STEM = 'E12bw480', 'qk_e12_bw480'
    DESIGN = ('wide-axis extension: wide 480 (6x80) -> narrow 208 = 26 x 8 '
              '(heads 4x52), shared values -- vs E12bw384 (384-wide) tests '
              'wide-width saturation')
elif ARM == 'w384n156':
    # grid completion: does the wider wide block also rescue the deeper
    # 156 narrowing? (w384n208 recovered -0.0345 of E12b's cost)
    DW, NHW, HDW = 384, 6, 64
    DN, NHN = 156, 3
    NAME, STEM = 'E12bw384n156', 'qk_e12_bw384n156'
    DESIGN = ('wide-width x narrowing grid: wide 384 (6x64) -> narrow 156 = '
              '26 x 6 (heads 3x52), shared values -- vs E12b156 (264-wide) '
              'and E12bw384 (208-narrow)')
elif ARM == 'w384':
    # wide-width axis: E12L's 384-wide detokenization + the 208 narrow with
    # shared values -- does a wider wide block absorb the narrowing cost?
    DW, NHW, HDW = 384, 6, 64
    DN, NHN = 208, 4
    NAME, STEM = 'E12bw384', 'qk_e12_bw384'
    DESIGN = ('wide-width axis: wide 384 (6x64, E12L detok) -> narrow 208 = '
              '26 x 8 (heads 4x52), shared values -- vs E12b (264-wide) '
              'isolates the wide-block width term in the narrowing cost')
else:
    DW, NHW, HDW = 264, 6, 44
    DN = int(ARM)
    NHN = {156: 3, 104: 2}[DN]
    NAME, STEM = f'E12b{DN}', f'qk_e12_b{DN}'
    DESIGN = (f'deeper shared-values narrowing: wide 264, narrow {DN} = '
              f'26 x {DN // 26} (heads {NHN}x52) -- where does the '
              f'recovered narrowing cost come back?')
SUB = DN // 26


def cfg_deep():
    return dict(Dw=DW, NHw=NHW, HDw=HDW, Gw=24, Dn=DN, NHn=NHN, HDn=52,
                Gn=26, sub_n=SUB, control=False)


def make_deep():
    torch.manual_seed(Q.SEED)
    return M.FunnelRoute(NAME, cfg_deep(), shared_values=True).to(E.DEV)


if __name__ == '__main__':
    M.JP = E.jpath(f'qk_e12_deep{ARM}.json')
    E.setup()
    M.run_arm(STEM, NAME, make_deep, DESIGN,
              extra_pairs=(('qk_e12_b', 'e12b'), ('qk_e12_a', 'e12a')))
    print(f'e12 deep{ARM} done', flush=True)
