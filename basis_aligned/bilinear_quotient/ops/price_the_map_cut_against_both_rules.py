# PRICE THE MAP CUT UNDER BOTH RULES, SO THE DECISION CAN BE MADE UNDER EITHER.
#
# §2030 and §2031 established that §2024's map cut is a uniform, mechanism-exact tax on the uncovered arm:
# -0.868 to -1.791 milli-nats there and EXACTLY 0.000 at covered inputs, across six cells and two
# coverages. §1947's rule sanctioned it on a POOLED average of +0.47 milli-nats of budget, and the pooled
# average cannot see a cost concentrated in a quarter of positions.
#
# Whether the rule should be changed is Logan's. What is available without him is the exposure: the cut is
# reversible, and §2024's own ladder showed its steps are not equal -- layers 0-5 cost 0.24x what they
# release, layers 6-7 cost 0.63x. This prices each cut depth under BOTH rules at once, so the decision is
# a lookup rather than another run.
#
# ARMS. §2020's build (tables raised, NO map cut); the cut at MLP layers 0-3, 0-5 and 0-7 (the last is
# §2024's, and the converged build); §1959's build as the common baseline; a fallback variant for the
# inert half of the control; and one differing-table-rank arm for the other half.
#
# Each cut site releases 0.885M values, worth 0.0000885 nats at §1947's 0.010-per-100M price.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2031's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

LO, HI = 'c5419', 'c16110'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
TABLES = {('mlp', L): 1152 for L in range(10, 18)}      # §2020
S1959, NOCUT = 'build_1959', 'cut0'
DEPTHS = (4, 6, 8)                                       # cut MLP layers 0..n-1
WORTH_PER_SITE = 0.0000885                               # §1947: 0.885M values

# §2030 at 5,419: §2020's build (no cut) against §1959's, at uncovered inputs, milli-nats
NOCUT_UNC = (0.316, 0.130, -3.837)


def _cut(n):
    poor = ','.join(f'mlp{L}' for L in range(n))
    rich = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18)
                    if not (k == 'mlp' and L < n))
    return f'mix30m256@{poor}+mix30m640@{rich}'


PLAN = [('mix30m640', BASE, S1959, None),
        ('mix30m640', {**BASE, **TABLES}, NOCUT, None)] + \
       [(_cut(n), {**BASE, **TABLES}, f'cut{n}', B.SITES) for n in DEPTHS] + [
    ('map512', BASE, 'shipped_fb_control', None),        # all 36 sites: the INERT pair
    ('mix30m640', A256, 'rank_control', None)]           # differing table rank: the other half


def _unc(x, cov, role, lab):
    """milli-nats `lab` beats the NO-CUT build by at uncovered inputs -- negative is the tax"""
    return 1000.0 * (x.ce(cov, role, NOCUT, 'uncovered_input')
                     - x.ce(cov, role, lab, 'uncovered_input'))


def _pooled(x, cov, lab):
    """nats `lab` beats the NO-CUT build by, pooled across roles"""
    return -x.tpool_full(cov, lab, NOCUT)['mean']


def _the_nocut_anchor_reproduces(x):
    """§2020's build with no map cut reproduces §2030's uncovered figures against §1959 -- +0.316 /
    +0.130 / -3.837 milli-nats within 0.05. Every tax below is measured from it"""
    return all(abs(1000.0 * (x.ce(LO, r, S1959, 'uncovered_input')
                             - x.ce(LO, r, NOCUT, 'uncovered_input')) - v) < 0.05
               for r, v in zip(x.roles, NOCUT_UNC))


def _the_tax_is_monotone_in_depth(x):
    """and the uncovered tax deepens with every cut depth, on >=2 roles -- cut0-3 shallower than cut0-5
    shallower than cut0-7. If FALSE the tax is not a per-site property and the partial reversal below
    cannot be priced by interpolation"""
    return sum(1 for r in x.roles
               if _unc(x, LO, r, 'cut4') > _unc(x, LO, r, 'cut6') > _unc(x, LO, r, 'cut8')) >= 2


def _the_tax_per_site_is_roughly_flat(x):
    """and the tax per cut site varies by under a factor of three across the three depths, on >=2 roles.
    §2031 found it uniform across roles and coverages; this asks whether it is uniform across DEPTH, which
    is what makes a partial reversal predictable"""
    def per(r, lab, n):
        return abs(_unc(x, LO, r, lab)) / n
    return sum(1 for r in x.roles
               if max(per(r, 'cut4', 4), per(r, 'cut6', 6), per(r, 'cut8', 8))
               < 3.0 * min(per(r, 'cut4', 4), per(r, 'cut6', 6), per(r, 'cut8', 8))) >= 2


def _the_shallow_cut_keeps_most_of_the_budget(x):
    """and cutting only layers 0-5 keeps most of the budget gain: its pooled CE cost is under half the
    full cut's, while releasing three quarters of the values. That is the partial reversal, priced --
    if TRUE it is available as a middle option under either rule"""
    return abs(_pooled(x, LO, 'cut6')) < 0.5 * abs(_pooled(x, LO, 'cut8'))


B.run(
    name='price_the_map_cut_against_both_rules',
    plan=PLAN,
    coverages=[(LO, B.FIT_5419, 5419), (HI, B.FIT_16110, 16110)],
    predicates=[
        ('pred_a_the_nocut_anchor_reproduces',
         '§2020\'s no-cut build reproduces §2030\'s +0.316/+0.130/−3.837 uncovered figures within 0.05',
         _the_nocut_anchor_reproduces),
        ('pred_b_the_tax_is_monotone_in_depth',
         'and the uncovered tax deepens with every cut depth (>=2 roles)', _the_tax_is_monotone_in_depth),
        ('pred_c_the_tax_per_site_is_roughly_flat',
         'and the tax per cut site varies by under 3x across depths (>=2 roles)',
         _the_tax_per_site_is_roughly_flat),
        ('pred_d_the_shallow_cut_keeps_most_of_the_budget',
         'and cutting only layers 0-5 costs under half the full cut\'s pooled CE',
         _the_shallow_cut_keeps_most_of_the_budget),
    ],
    refs=[(S1959, B.PT + 'ops/the_converged_build_end_to_end_results.json', 'build_1959', LO, 0.0005)],
    paired_pairs=[('cut8', NOCUT), ('cut6', NOCUT), ('cut4', NOCUT), (NOCUT, S1959)],
)
