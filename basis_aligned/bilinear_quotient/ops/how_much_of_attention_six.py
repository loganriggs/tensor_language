# HOW MUCH OF ATTENTION 6 DOES THE THRESHOLD NEED?
#
# §1992 and §1993 established a threshold that survived three attempts to turn it into a rule: an arm
# compiling both attention 5 and attention 6 costs 1.50-2.15 nats, one omitting either costs 2.56-10.94.
# §1994 placed 90-95% of the gap at covered inputs and §1995 killed its frequency gradient as generic.
#
# Every question asked of the threshold so far has been about MEMBERSHIP -- which sites are in the set --
# and all three answers were wrong. This asks one of DEGREE instead. Compiling attention 6 means replacing
# its output with a rank-384 context-free table; if the requirement is really "attention 6 must stop
# varying with context", a rank-16 table should satisfy it as well as a rank-384 one. If instead the
# threshold needs attention 6's table to carry real capacity, the curve will rise as the rank falls.
#
# ARMS. mlp2 + attention 5,6 with attention 6's table at rank 16 / 32 / 64 / 128 / 384, everything else at
# {mlp 768, attn 384}; the rank-384 arm is §1992's mlp2_a56 exactly. Plus mlp2 + attention 5 alone, the
# arm on the wrong side of the threshold, and one differing-table-rank arm so neither half of the derived
# control is vacuous.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1995's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
SITES2 = [('mlp', 2), ('attn', 5), ('attn', 6)]
RANKS = (16, 32, 64, 128, 384)
LAB = [f'r{r}' for r in RANKS]
BAD = 'mlp2_a5'
A256 = {'mlp': 768, 'attn': 256}


def _spec(r):
    return {'mlp': 768, 'attn': 384, ('attn', 6): r}


PLAN = [(ARM, _spec(r), f'r{r}', SITES2) for r in RANKS] + [
    (ARM, {'mlp': 768, 'attn': 384}, BAD, [('mlp', 2), ('attn', 5)]),   # §1992: 2.556 / 2.730 / 2.558
    (ARM, {'mlp': 768, 'attn': 384}, 'full_program', None),             # §1985: 2.808 / 2.979 / 2.702
    ('map512', {'mlp': 768, 'attn': 384}, 'full_fb_control', None),     # the INERT pair: same rank, ALL
    #                                                                     36 sites, different fallback.
    #                                                                     Run 1 of this script had no
    #                                                                     same-spec pair at all; run 2
    #                                                                     put the fallback partner at
    #                                                                     THREE sites, where attention is
    #                                                                     still live and inertness does
    #                                                                     not hold -- and it failed,
    #                                                                     correctly. bqlib now refuses to
    #                                                                     call a partial-site pair inert.
    (ARM, A256, 'rank_control', None)]                                  # differing rank, all 36 sites


def _full_rank_reproduces(x):
    """the rank-384 arm reproduces §1992's mlp2_a56 to 0.005 nats on all three roles -- naming attention 6
    explicitly at the same rank its kind already carried must be a no-op, and if it is not, the per-site
    rank plumbing is wrong and nothing else in this run means anything"""
    want = (1.971, 2.090, 1.952)
    return all(abs(x.penalty(C, r, 'r384') - v) < 0.005 for r, v in zip(x.roles, want))


def _low_rank_is_enough(x):
    """and rank 32 at attention 6 stays on the good side of the threshold -- under 2.5 nats on >=2 roles,
    against the wrong-side arm's 2.556. If TRUE the requirement is that attention 6 stop varying with
    context and almost none of its table's capacity is needed"""
    return sum(1 for r in x.roles if x.penalty(C, r, 'r32') < 2.5) >= 2


def _the_axis_is_not_flat(x):
    """and the axis is not trivially flat: rank 16 costs at least 0.02 nats more than rank 384 on >=2
    roles. If FALSE the table at attention 6 could be rank 16 for free and the whole capacity question is
    empty -- which would itself be worth knowing"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, 'r16') - x.penalty(C, r, 'r384') >= 0.02) >= 2


B.run(
    name='how_much_of_attention_six',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_rank_reproduces',
         'naming attention 6 at rank 384 reproduces §1992\'s mlp2_a56 to 0.005 nats on 3/3 roles',
         _full_rank_reproduces),
        ('pred_b_low_rank_is_enough',
         'and rank 32 at attention 6 stays under 2.5 nats -- on the good side of the threshold',
         _low_rank_is_enough),
        ('pred_c_the_axis_is_not_flat',
         'and rank 16 costs at least 0.02 nats more than rank 384 (>=2 roles)', _the_axis_is_not_flat),
    ],
    refs=[(BAD, B.PT + 'ops/where_the_threshold_gap_lives_results.json', BAD, C, 0.0005)],
    paired_pairs=[('r16', 'r384'), ('r32', 'r384'), ('r384', BAD)],
)
