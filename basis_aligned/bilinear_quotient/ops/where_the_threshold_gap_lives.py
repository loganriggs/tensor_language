# WHERE DO THE THRESHOLD'S 0.55 NATS LIVE?
#
# §1993 replicated the one surviving statement of the §1984–§1992 line at both coverages: compiling
# attention 5 and attention 6 together is what makes a compiled MLP below them affordable, and omitting
# either costs 0.54–0.61 nats. Three attempts to say WHICH sites are required have been falsified, so this
# asks a different kind of question — not which sites, but which POSITIONS pay.
#
# Both instruments are already in every artifact and need no new plumbing: the §1789 target-frequency
# buckets and the §1936 covered/uncovered input axis. The program is weakest on rare targets and on
# uncovered input tokens, so the deflationary reading is that the whole threshold is an artefact of those
# cells. If instead the gap is present everywhere, it is a fact about the computation rather than about
# the fallback.
#
# ARMS. mlp2 + attention 5,6 against mlp2 + attention 5 — the same compiled MLP, differing by exactly one
# attention layer, one on each side of the threshold. Plus mlp4 + attention 5,6 and the full 36-site
# program as anchors, and one fallback variant so the inert half of the derived control is real.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §1993's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

A384 = {'mlp': 768, 'attn': 384}
ARM = 'mix30m640'
C = 'c5419'
A56 = [('attn', 5), ('attn', 6)]
GOOD, BAD = 'mlp2_a56', 'mlp2_a5'
ANCHOR, FULL = 'mlp4_a56', 'full_program'

PLAN = [(ARM, A384, GOOD, [('mlp', 2)] + A56),                # §1992: 1.971 / 2.090 / 1.952
        (ARM, A384, BAD, [('mlp', 2), ('attn', 5)]),          # §1992: 2.556 / 2.730 / 2.558
        (ARM, A384, ANCHOR, [('mlp', 4)] + A56),              # §1990: 1.555 / 1.640 / 1.498
        (ARM, A384, FULL),                                    # §1985: 2.808 / 2.979 / 2.702
        ('map512', A384, 'full_fb_control')]                  # other fallback, same sites: INERT


def _gap(x, role, cls='pooled', bucket='overall'):
    return x.ce(C, role, BAD, cls, bucket) - x.ce(C, role, GOOD, cls, bucket)


def _not_a_fallback_artefact(x):
    """the gap at COVERED inputs is at least half the pooled gap, on >=2 roles. The uncovered arm is where
    the program guesses, and if the threshold lived there it would be a fact about the fallback rather
    than about the computation"""
    return sum(1 for r in x.roles
               if _gap(x, r, 'covered_input') >= 0.5 * _gap(x, r)) >= 2


def _not_only_rare_targets(x):
    """and the gap in the most frequent target bucket is at least a quarter of the gap in the unseen
    bucket, on >=2 roles -- if the threshold were only about rare targets, the frequent cells would show
    almost none of it"""
    return sum(1 for r in x.roles
               if _gap(x, r, 'pooled', x.top) >= 0.25 * _gap(x, r, 'pooled', x.bot)) >= 2


def _present_in_every_bucket(x):
    """and the gap is strictly positive in all five frequency buckets on >=2 roles -- the threshold is a
    property of every part of the distribution, not a sum over a few cells"""
    return sum(1 for r in x.roles
               if all(_gap(x, r, 'pooled', b) > 0 for b in x.buckets)) >= 2


B.run(
    name='where_the_threshold_gap_lives',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_not_a_fallback_artefact',
         'the gap at covered inputs is at least half the pooled gap (>=2 roles)',
         _not_a_fallback_artefact),
        ('pred_b_not_only_rare_targets',
         'and the most frequent bucket carries at least a quarter of the unseen bucket\'s gap',
         _not_only_rare_targets),
        ('pred_c_present_in_every_bucket',
         'and the gap is strictly positive in all five frequency buckets (>=2 roles)',
         _present_in_every_bucket),
    ],
    refs=[(GOOD, B.PT + 'ops/is_it_just_attention_five_and_six_results.json', 'mlp2_a56', C, 0.0005),
          (BAD, B.PT + 'ops/is_it_just_attention_five_and_six_results.json', 'mlp2_a5', C, 0.0005),
          (ANCHOR, B.PT + 'ops/the_minimal_path_results.json', 'mlp4_a56', C, 0.0005)],
    paired_pairs=[(GOOD, BAD), (GOOD, ANCHOR), (GOOD, FULL)],
)
