# DOES §2012'S FINDING REACH THE SHIPPED PROGRAM AT ALL?
#
# §2012 found a compiled mlp2's content worth 1.408 nats inside the three-site repaired arm — six times
# attention 6's and two hundred times what §1983 measured for a lone compiled mlp4. That is the largest
# term the localisation line has produced, and it was measured in a probe, not a program.
#
# The shipped build (§1765, §1959) compiles all 36 sites at {mlp 768, attn 384}. In it, attention is
# DELETED — there is no live attention 6 and so no interface to break or repair. §2012's mechanism has no
# purchase there, and the honest prediction is that mlp2's content is much smaller in the full program
# than in the repaired arm. If instead it transfers, the whole §1983–§2012 line bears on the shipped
# allocation; if it does not, the line is about partial compilation only and should say so.
#
# The allocation question follows either way and is asked here directly: §1947's price rule is 0.010 nats
# per 100M parameters, and the build buys rank 768 at every MLP site.
#
# ARMS. the full 36-site program with mlp2's table at rank 1 / 16 / 128 / 768, and the same with a MEAN
# ROW at mlp2 as the zero-content baseline; a fallback variant of the full program for the inert half of
# the control; and one differing-table-rank arm for the other half.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3 -- §2012's open question.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bqlib as B                                                          # noqa: E402

ARM = 'mix30m640'
C = 'c5419'
BASE = {'mlp': 768, 'attn': 384}
A256 = {'mlp': 768, 'attn': 256}
RANKS = (1, 16, 128, 768)
MEAN = 'm2_mean'

FULL_PROGRAM = (2.808, 2.979, 2.702)       # §1985 / §1990
REPAIRED_CONTENT = (1.408, 1.469, 1.436)   # §2012: mlp2's content inside mlp2 + attention 5,6
REPAIRED_AT_16 = (0.440, 0.451, 0.437)     # §2012: its rank-16 content share there

_OTHERS = ','.join(f'{k}{L}' for k in ('mlp', 'attn') for L in range(18) if not (k == 'mlp' and L == 2))
MEAN_ARM = f'meanrow@mlp2+{ARM}@{_OTHERS}'


def _spec2(r):
    return {'mlp': 768, 'attn': 384, ('mlp', 2): r}


PLAN = [(ARM, _spec2(r), f'r{r}', None) for r in RANKS] + [
    (MEAN_ARM, BASE, MEAN, B.SITES),
    ('map512', _spec2(768), 'full_fb_control', None),         # SAME rank spec, all 36 sites, other
    #                                                           fallback: the INERT pair. Run 1 gave the
    #                                                           partner the plain {mlp,attn} spec, which
    #                                                           is a different _rk_key from _spec2(768),
    #                                                           so there was no same-spec pair at all and
    #                                                           control_is_two_sided FAILED.
    (ARM, A256, 'rank_control', None)]                        # differing table rank: the other half


def _content(x, role, lab):
    hi, lo = x.penalty(C, role, MEAN), x.penalty(C, role, 'r768')
    return (hi - x.penalty(C, role, lab)) / (hi - lo)


def _full_rank_reproduces(x):
    """the rank-768 arm reproduces the shipped program's 2.808 / 2.979 / 2.702 within 0.005 nats on all
    three roles -- naming mlp2 at the rank its kind already carried must be a no-op"""
    return all(abs(x.penalty(C, r, 'r768') - v) < 0.005 for r, v in zip(x.roles, FULL_PROGRAM))


def _content_is_much_smaller_here(x):
    """and mlp2's content in the FULL program is under half what it was in the repaired arm, on >=2 roles
    -- under 0.704 / 0.735 / 0.718 nats. Registered directionally: attention is DELETED in the full
    program, so there is no interface and §2012's mechanism has no purchase. If FALSE the §1983-§2012 line
    bears directly on the shipped allocation and not only on partial compilation"""
    return sum(1 for r, v in zip(x.roles, REPAIRED_CONTENT)
               if x.penalty(C, r, MEAN) - x.penalty(C, r, 'r768') < 0.5 * v) >= 2


def _the_shipped_rank_is_not_over_bought(x):
    """and the shipped allocation is not over-buying at this site: dropping mlp2 from rank 768 to rank 128
    costs under 0.010 nats on >=2 roles, which is §1947's price of 100M parameters. If FALSE the build is
    paying for capacity at mlp2 that a cheaper rank would supply, and the allocation set in §1959 wants
    revisiting"""
    return sum(1 for r in x.roles
               if x.penalty(C, r, 'r128') - x.penalty(C, r, 'r768') < 0.010) >= 2


B.run(
    name='does_it_transfer_to_the_shipped_program',
    plan=PLAN,
    coverages=[(C, B.FIT_5419, 5419)],
    predicates=[
        ('pred_a_full_rank_reproduces',
         'naming mlp2 at rank 768 reproduces the shipped program to 0.005 nats on 3/3 roles',
         _full_rank_reproduces),
        ('pred_b_content_is_much_smaller_here',
         'and mlp2\'s content in the full program is under half its repaired-arm value (>=2 roles)',
         _content_is_much_smaller_here),
        ('pred_c_the_shipped_rank_is_not_over_bought',
         'and dropping mlp2 to rank 128 costs under 0.010 nats -- §1947\'s price of 100M (>=2 roles)',
         _the_shipped_rank_is_not_over_bought),
    ],
    refs=[('r768', B.PT + 'ops/the_minimal_path_results.json', 'full_program', C, 0.0005)],
    paired_pairs=[('r1', MEAN), ('r128', 'r768'), ('r16', 'r768')],
)
