# IS THE SPIKE A PROPERTY OF LAYER 4, OR OF mlp4?
#
# §1988 mapped the lone-compiled-MLP profile: 4.81 / 6.57 / 10.67 / 2.02 / 0.20 / 0.07 for layers 2–7. It
# is a peak at layer 4 with a ramp below and a cliff above, and nothing in the path rule predicts a peak.
# Every one of those arms compiled an MLP, so the profile cannot distinguish two readings:
#
#   (i)  the RESIDUAL STREAM at layer 4 is where attention 6 gets what it needs, and any site there is
#        expensive to freeze -- so a lone compiled ATTENTION 4 should be expensive too;
#   (ii) it is mlp4 specifically, and the lone attention layers below 6 are all cheap.
#
# This measures the attention profile the same way, over the same layers.
#
# ARMS. attention 2, 3, 4, 5, 7 alone; mlp4 alone as the §1985 anchor; the full 36-site program; and one
# fallback variant of it so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1988's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
A = {L: f'attn{L}' for L in (2, 3, 4, 5, 7)}
MLP4, FULL = 'mlp4', 'full_program'

PLAN = [(ARM, A384, A[L], [('attn', L)]) for L in (2, 3, 4, 5, 7)] + [
    (ARM, A384, MLP4, [('mlp', 4)]),      # §1985: 10.669 / 10.937 / 10.580
    (ARM, A384, FULL),                    # §1985: 2.808 / 2.979 / 2.702
    ('map512', A384, 'full_fb_control')]  # same rank, same sites, other fallback: INERT


def _attention_below_six_is_expensive(x):
    """a lone compiled attention 4 costs more than 1 nat on >=2 roles. If TRUE the spike is a property of
    the residual stream at layer 4 and not of mlp4; if FALSE it belongs to mlp4 specifically"""
    return sum(1 for r in x.roles if x.penalty(C, r, A[4]) > 1.0) >= 2


def _attention_peaks_at_four_too(x):
    """and the attention profile peaks in the same place: attention 4 costs strictly more than both
    attention 2 and attention 5 on all three roles, as mlp4 does among its neighbours"""
    return all(x.penalty(C, r, A[4]) > max(x.penalty(C, r, A[2]), x.penalty(C, r, A[5]))
               for r in x.roles)


def _attention_above_six_is_free(x):
    """and attention 7, above the boundary, costs under 0.5 nats on >=2 roles -- as mlp7 did at 0.072.
    The boundary should not care which kind of site sits above it"""
    return sum(1 for r in x.roles if x.penalty(C, r, A[7]) < 0.5) >= 2


B.run(
    name='does_attention_spike_too',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_attention_below_six_is_expensive',
         'a lone compiled attention 4 costs more than 1 nat (>=2 roles) -- the spike is layer 4, not mlp4',
         _attention_below_six_is_expensive),
        ('pred_b_attention_peaks_at_four_too',
         'and the attention profile peaks at layer 4 as the MLP one does (3/3 roles)',
         _attention_peaks_at_four_too),
        ('pred_c_attention_above_six_is_free',
         'and attention 7, above the boundary, costs under 0.5 nats -- as mlp7 did',
         _attention_above_six_is_free),
    ],
    refs=[(MLP4, B.PT + 'ops/where_the_cliff_is_results.json', 'mlp4', C, 0.0005),
          (FULL, B.PT + 'ops/where_the_cliff_is_results.json', FULL, C, 0.0005)],
    paired_pairs=[(A[4], A[5]), (A[4], A[2]), (A[4], MLP4)],
)
