"""WINDOW CERTIFICATION OF THE THREE-HEAD ENVELOPE (rung 58).

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

S2150 certified attn16 as a three-head object on FR only. This runs the two configs with the zeroing active
THROUGHOUT (fits and window evals; bases excluded - the reference stays the real model): skip-a16 with all nine
heads vs skip-a16 with heads 16.1/2/5/6/7/8 zeroed. PRICE: the three-head description drops 6/9 of attn16.
NULL: window heterogeneity (some window relies on a "free" head).

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) THE FR RESULT HOLDS AT WINDOW GRAIN: median over the 8 windows of [L2_F_w(three) - L2_F_w(nine)] <= +0.005.
  (b) NOT ONE WINDOW: that difference is <= +0.01 on >= 6 of 8 windows.
  (c) REPRODUCTION: the nine-head arm reproduces S2146's per-window values (median |delta| <= 0.005).

Writes attn16_three_heads_w8_results.json. Self-reviewed. RUNG-57 HEADER FOLLOWS.

attn16 AS THREE HEADS (rung 57): the S2149 additive prediction, tested as a number.

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

S2149: heads 16.3/16.4/16.0 carry +0.113 of attn16's +0.119; the other six sum to -0.0012 and the head lattice
is additive to 0.0072. Zero heads {1,2,5,6,7,8} TOGETHER on the S2146 skip-a16 config and eval FR. If the
additive number holds, block-16 attention reduces to a THREE-HEAD object for the program's purposes.
PRICE: none here (attribution); the implied object drops 6/9 of attn16's description. NULL: head interactions
(the S2142 cross-kind precedent).

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) THE ADDITIVE NUMBER: |[L2_F(six zeroed) - L2_F(skip16)] - (-0.0012)| <= 0.01.
  (b) NEARLY FREE: that difference is <= +0.01.
  (c) REPRODUCTION: the skip-a16 config reproduces S2146 (FR L2_F 2.5091 within 0.01).

Writes attn16_three_heads_results.json. Self-reviewed. RUNG-55 HEADER FOLLOWS.

PER-HEAD MARGINALS OF THE REAL attn16 (rung 55): attribution inside the block-16 bottleneck.

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER; d_h = damage added on FR by zeroing
head h of the REAL attn16 (c_proj input slice) under the S2146 skip-a16 config.

S2145-S2148: a16L is the costliest replacement, its residual is damaging per unit energy, and class-linear
upgrades cannot touch it. Before any new grammar is designed for block 16, name where the computation sits:
zero each of attn16's nine heads in turn (and all nine together) and price them. PRICE: none (attribution).
NULL: uniform heads (d_h ~ d_all/9 each).

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) CONCENTRATION: max_h d_h >= 2 x median_h d_h (S2113/S2117 precedent: head grain is where structure lives).
  (b) ADDITIVITY: |sum_h d_h - d_all| <= 0.05 (S2139/S2143 precedent; heads may interact - a failure is
      informative, not fatal).
  (c) REPRODUCTION: the skip-a16 config reproduces S2146 (FR L2_F 2.5091 within 0.01).

Writes attn16_head_marginals_results.json. Self-reviewed. RUNG-52 HEADER FOLLOWS.

LEAVE attn16 REAL (rung 52): the coverage-vs-damage envelope point that S2145 licenses.

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

S2145: a16L's class dictionary is the assembly's single most expensive replacement (+0.1572 of the +0.353 tail
increment). This refits the tail sequence with block 16's attention left REAL - a10L..a15L as before, a16L
skipped, a17L refit under a16-real (grammar-consistent: fits stay sequential) - on the S2144 best middles
(mlp45-2304 + c6/c7-576 + c8/c9-288). HONEST FRAMING: the resulting config replaces ONE FEWER component; this is
an envelope point (less coverage, less damage), not a free improvement. NULL: the a17L refit under a16-real
reabsorbs the marginal and the gap shrinks well below 0.157.

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) THE SAVING SURVIVES REFITTING: median over the 8 windows of [L2_F_w(skip16) - L2_F_w(full)] <= -0.10.
  (b) NOT ONE WINDOW: that difference is <= -0.05 on >= 7 of 8 windows.
  (c) REPRODUCTION: the full arm reproduces S2144 (FR L2_F 2.6662 within 0.01).

Writes frontier_skip_a16_results.json. Self-reviewed. RUNG-51 HEADER FOLLOWS.

WHO PAYS THE TAIL-ATTENTION INCREMENT (rung 51): the grammar-consistent prefix curve over a10L-a17L.

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

The frontier pays +0.37 of its damage for replacing the eight tail attentions (blocks 10-17) with class
dictionaries. Because the aXL pieces are FIT SEQUENTIALLY (each under the stack with the previous ones
installed), the prefix configs a10L..a(10+k)L are matched-context by construction - evaluating them gives a
per-layer marginal cost attribution with no refits and no grammar violation. Run once on the current best
middles (mlp45-2304 + c69-576), eval the nine prefixes on FR. PRICE: none (attribution rung).
NULL: uniform marginals (~0.046 each).

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) REPRODUCTION: the full config reproduces S2140's best FR L2_F 2.6691 within 0.01.
  (b) EVERY REPLACEMENT COSTS: all eight marginals [L2_F(prefix k+1) - L2_F(prefix k)] >= -0.005. If some
      marginal is < -0.005, that block's real attention was HURTING the assembly - notable on its own.
  (c) CONCENTRATION: the largest marginal >= 2 x the median marginal (the increment is not uniform; §2143
      showed the middles are not interchangeable - this asks the same of the tail).

Writes frontier_tail_prefix_results.json. Self-reviewed. RUNG-50 HEADER FOLLOWS.

c8+c9 TO 288 TOGETHER (rung 50): the additive prediction, tested as a number.

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

S2143: c8@288 and c9@288 are individually free-or-better (-0.0076, -0.0036) and the per-layer costs add. The
registered additive prediction: together they land at -0.0112 vs the S2140 best - a NEW best config at -14.4M
values vs S312. Arms: norm-2304 anchor | best (c69-576, S2140 reproduction) | c8+c9@288 | c8+c9@144
(descriptive floor). NULL: S2142 showed composition CAN fail when fixes share a budget.

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) A REAL IMPROVEMENT: median over the 8 windows of [L2_F_w(c89_288) - L2_F_w(best)] <= -0.005.
  (b) THE ADDITIVE NUMBER: that median is within 0.01 of -0.0112.
  (c) REPRODUCTION: the best arm reproduces S2140 (median vs norm-2304 of -0.0290, within 0.015).

Writes frontier_c89_288_results.json. Self-reviewed. RUNG-49 HEADER FOLLOWS.

PER-LAYER c6-c9 FLOORS ON THE FRONTIER BEST (rung 49).

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

S2140: c6-c9 uniformly at 288 rebounds by +0.0217 vs the 576 optimum. Which layer carries the rebound? Arms:
norm-2304 anchor | best (c69-576, S2140 reproduction) | c6@288 | c7@288 | c8@288 | c9@288 (the others at 576).
Price: -0.55M further values per layer at 288. NULL: S2139's additivity precedent.

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) SOMEONE GOES TO 288 FREE: min over layers of median_w[L2_F_w(cX@288) - L2_F_w(best)] <= +0.005.
  (b) ADDITIVITY: | sum over the four layers of those medians - 0.0217 | <= 0.015 (0.0217 = the S2140 uniform-288
      rebound). If FAILED, per-layer interactions exist.
  (c) REPRODUCTION: the best arm reproduces S2140 (median vs norm-2304 of -0.0290, within 0.015).

Writes frontier_c69_perlayer_results.json. Self-reviewed. RUNG-47 HEADER FOLLOWS.

CAN mlp4/5 SHED UNITS ON TOP OF THE NEW BEST (rung 47).

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

S2140: the best measured config is mlp4/5 at norm-2304 + c6-c9 at norm-576 (2.6445 fresh, -13.3M values vs S312,
better on 8/8 windows). S2139: HALVING mlp4/5 costs +0.047 - but a quarter-trim (2304 -> 1728) may be much
cheaper (S2137: the marginal price per dropped unit falls steeply). Arms: norm-2304 everywhere (anchor) | best
(mlp45-2304 + c69-576; S2140 reproduction) | mlp45-1728 on top | mlp45-1152 on top.

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) THE QUARTER-TRIM IS CHEAP: median over the 8 windows of [L2_F_w(mlp45_1728) - L2_F_w(best)] <= +0.015.
  (b) NOT ONE WINDOW: that difference is <= +0.03 on >= 6 of 8 windows.
  (c) REPRODUCTION: the best arm reproduces S2140 (median vs norm-2304 of -0.0290, within 0.015).
  mlp45-1152 on top is descriptive (S2139 predicts ~ +0.047 there), no bar.

Writes frontier_mlp45_trim_results.json. Self-reviewed. RUNG-46 HEADER FOLLOWS.

HOW FAR DOWN DOES c6-c9 GO FOR FREE (rung 46).

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

S2139: with mlp4/5 kept at 2304, halving c6-c9 is free-or-better (-0.0118 median). Arms: norm-2304 everywhere
(S312 reproduction) | c69-1152 (S2139 reproduction) | c69-576 | c69-288. Price: -8.9M / -13.3M / -15.6M values.
NULL: S2137 showed norm-576-everywhere added +0.038, but that included the expensive mlp4/5 halving; the c6-c9
share must appear somewhere below 1152.

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) QUARTERING c6-c9 IS NEARLY FREE: median over the 8 windows of [L2_F_w(c69_576) - L2_F_w(norm2304)] <= +0.01.
  (b) NOT ONE WINDOW: that difference is <= +0.02 on >= 6 of 8 windows.
  (c) REPRODUCTION: the c69-1152 arm reproduces S2139 (median vs norm-2304 of -0.0118, within 0.015).
  c69-288 is descriptive, no bar.

Writes frontier_c69_floor_results.json. Self-reviewed. RUNG-45 HEADER FOLLOWS.

WHO PAYS FOR THE HALVING (rung 45): per-group attribution of S2137's +0.0292.

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER.

S2137: halving all six CP middles adds +0.0292 median damage. S2136 hinted c6-c9 pruning is nearly free while
S2113 put blocks 6-9 on the price cliff - genuine tension. Arms: norm-2304 everywhere (S312 reproduction) |
halve mlp4/mlp5 only (c6-c9 at 2304) | halve c6-c9 only (mlp4/5 at 2304). Price: -4.4M / -8.9M values.

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) ADDITIVITY: | median_w[dmg(halve45)] + median_w[dmg(halve6789)] - 0.0292 | <= 0.01 (dmg(arm) =
      L2_F_w(arm) - L2_F_w(norm2304)).
  (b) mlp4/5 PAY MORE: median_w[dmg(halve45)] >= median_w[dmg(halve6789)] (per S2136's near-free c6-c9 pruning;
      if FAILED, the S2113 cliff reading wins and the cheap units are at mlp4/5).
  (c) REPRODUCTION: the norm-2304 arm reproduces 2.6735 within 0.01.

Writes frontier_halving_attrib_results.json. Self-reviewed. RUNG-43 HEADER FOLLOWS.

NORM-RANK K REDUCTION ON THE INTACT S312 FRONTIER (rung 43).

CONVENTION (S2135): L2 is CE ADDED ABOVE THE REAL MODEL - LOWER IS BETTER. Every pred below is stated in that
convention.

S2136: norm-rank pruning of c6-c9 roughly broke even on a damaged base; the honest question - pure norm-rank K
reduction on the intact S312 frontier - has never been measured on the eight windows. Arms: norm-2304 everywhere
(S312 reproduction) | norm-1152 at all six CP middles | norm-576 at all six. Price: -13.3M / -20M stored values.
NULL (S2118 family): K reduction adds damage; the question is how little.

REGISTERED PREDICTIONS (arm-named; damage convention):
  (a) HALF THE UNITS FOR <= 0.02 NAT: median over the 8 windows of [L2_F_w(norm1152) - L2_F_w(norm2304)] <= +0.02.
  (b) NOT ONE WINDOW: that difference is <= +0.04 on >= 6 of 8 windows.
  (c) REPRODUCTION: the norm-2304 arm reproduces 2.6735 within 0.01.
  norm-576 is descriptive (the price curve's next point), no bar.

Writes frontier_norm_ksweep_results.json. Self-reviewed. RUNG-42 HEADER FOLLOWS.

PRUNE c6-c9 UNDER THE CONDITIONED RANKING (rung 42).

S2131 showed REORDERING c6-c9 at K = 2304 adds nothing. S2134 reframed the mlp4/5 win: it came from DROPPING the
harmful tail of the conditioned ranking, not from reordering the top. So the open question at c6-c9 is pruning:
with mlp4/5 fixed at cond-576 (the S2134 best), cut c6-c9 from 2304 to 576 each - ranked by the conditioned
Fisher at their sites vs by norm. Saves 4 x 1728 units (~23.9M values) if free. The S2131 null (conditioned
ranking is worthless at c6-c9) predicts pred_b ~ 0.

ARMS: norm-2304 (Fisher collection, sites 5-10) | cond576@45 (S2134 reproduction; c6-c9 norm-2304) |
c69norm576 (c6-c9 pruned by NORM rank) | c69cond576 (c6-c9 pruned by conditioned rank).

REGISTERED PREDICTIONS (arm-named):
  (a) PRUNING c6-c9 IS AT LEAST FREE: median over the 8 windows of [L2_F_w(c69cond576) - L2_F_w(cond576@45)] >= 0.
  (b) THE CONDITIONED RANK BEATS NORM FOR WHAT SURVIVES: median [L2_F_w(c69cond576) - L2_F_w(c69norm576)] >= 0.
  (c) REPRODUCTION: the cond576@45 arm reproduces S2134 (median over norm +0.1277 within 0.015).

Writes frontier_c69_prune_results.json. Self-reviewed. RUNG-41 HEADER FOLLOWS.

THE PRICE FLOOR (rung 41): K = 144, and K = 0 as the "do the units matter at all" null.

S2134: the conditioned curve peaks near K = 576 (+2.8372 fresh, quarter price) and cond-288 barely turns down.
Arms: norm-2304 (Fisher collection) | cond-576 (S2134 reproduction) | cond-144 | K-0 (Down-bias only).

REGISTERED PREDICTIONS (arm-named):
  (a) SIXTEENTH PRICE AT NO LOSS: median over the 8 windows of [L2_F_w(cond144) - L2_F_w(cond576)] >= -0.01.
  (b) THE UNITS CARRY REAL CE: median [L2_F_w(K0) - L2_F_w(cond576)] <= -0.05. If FAILED, mlp4/mlp5 CP content
      is nearly all harmful-or-inert in the deployed assembly - a major finding on its own.
  (c) REPRODUCTION: the cond-576 arm reproduces S2134 (median over norm +0.1277 within 0.015).

Writes frontier_cond_floor_results.json. Self-reviewed. RUNG-40 HEADER FOLLOWS.

THE CONDITIONED PRICE CURVE, DOWNWARD (rung 40): does the half-price win extend to quarter and eighth price?

S2133: cond-1152 beats norm-2304 by +0.0861 median and cond-2304 by ~+0.04 on every window - the bottom half of
the conditioned ranking is net harmful when deployed. S2107 (capacity vs selection) says the curve must turn
somewhere below; this finds where. Arms: norm-2304 (Fisher collection) | cond-1152 (S2133 reproduction) |
cond-576 | cond-288, each on the eight document-disjoint windows.

REGISTERED PREDICTIONS (arm-named):
  (a) QUARTER PRICE AT NO LOSS: median over the 8 windows of [L2_F_w(cond576) - L2_F_w(cond1152)] >= -0.01.
  (b) NOT ONE WINDOW: that difference is >= -0.02 on >= 6 of 8 windows.
  (c) REPRODUCTION: the cond-1152 arm reproduces S2133 (median over norm +0.0861 within 0.015).
  cond-288 is descriptive (the curve's next point), no bar.

Writes frontier_cond_ksweep_results.json. Self-reviewed. RUNG-39 HEADER FOLLOWS.

HALF-PRICE UNDER THE CONDITIONED METRIC (rung 39): does the assembly-conditioned selector let mlp4/mlp5 keep
HALF the units at no CE loss in the deployed frontier?

S2107 found metric-1152 ~ norm-2304 on two windows; S2118 WITHDREW half-price on the eight fresh windows under
the real-model metric on cfgE (median -0.028, worse on six of eight) - that is the stated null. But S2129 showed
the conditioned metric is the right one for the frontier (+0.0481 certified), so the price question reopens
exactly once, here: arms norm-2304 | conditioned-2304 (S2129 reproduction) | conditioned-1152 at mlp4/mlp5, each
on the eight document-disjoint windows. LITERAL PRICE: conditioned-1152 stores 2 x 1152 fewer CP units
(~7.96M fewer values) than either 2304 arm, plus the same 8 x 1152 selector per site.

REGISTERED PREDICTIONS (arm-named):
  (a) HALF PRICE AT NO LOSS: median over the 8 windows of [L2_F_w(cond1152) - L2_F_w(norm2304)] >= 0.
      If FAILED the S2118 withdrawal extends to the conditioned metric and the price axis is closed at mlp4/5.
  (b) NOT ONE WINDOW: that difference is >= -0.01 on >= 6 of 8 windows.
  (c) REPRODUCTION: the conditioned-2304 arm reproduces S2129 (median gain over norm +0.0481 within 0.015).

Writes frontier_cond_halfprice_results.json. Self-reviewed. RUNG-37 HEADER FOLLOWS.

EXTEND THE CONDITIONED SELECTION TO ALL SIX CP MIDDLES (rung 37).

S2129 certified the assembly-conditioned selector at mlp4/mlp5 (+0.0481 median over eight windows). The other four
CP middles (c6-c9) still use norm selection - and S2106/S2107 found that REAL-model metric selection there HURTS
on cfgE, so the null expectation is genuine. This collects the assembly-conditioned Fisher top-8 at sites 5-10
during the norm arm and runs THREE arms: norm | conditioned selection at mlp4/mlp5 (S2129 reproduction) |
conditioned selection at mlp4-mlp9, each scored on the eight document-disjoint windows.

REGISTERED PREDICTIONS (arm-named, per the S2128 process rule):
  (a) THE OTHER FOUR LAYERS ADD: median over the 8 windows of [L2_F_w(all) - L2_F_w(mlp45)] >= +0.02.
      If FAILED, conditioned selection is an mlp4/mlp5 story like the real-model metric was (S2106) and the
      frontier keeps the S2129 config.
  (b) NOT ONE WINDOW: that increment is >= 0 on >= 6 of 8 windows.
  (c) REPRODUCTION: the mlp45 arm reproduces S2129 (median gain over norm +0.0481 within 0.015; the Fisher
      collection has ~0.003 CUDA-atomics wobble, S2129).

Writes frontier_asm_alllayers_results.json. Self-reviewed. RUNG-35 HEADER FOLLOWS.

CERTIFY THE CONDITIONED FRONTIER AT THE RUNG-6 STANDARD (rung 35).

S2128 (correcting the sign-inverted S2125): assembly-conditioned true-Fisher top-8 selection at mlp4/mlp5 lifts
the S312 frontier from +2.6735 to +2.7682 fresh — but that fresh number is ONE 120-row window set (FR). Before
+2.7682 is called the frontier best it must meet the certification standard every other frontier claim met: the
eight document-disjoint pile-10k windows (built exactly as ops/metric_units_fresh8.py builds them). This reruns
both arms and evaluates each L2 config on all eight windows; fits touch FW rows only, so every window is held out
for both arms.

REGISTERED PREDICTIONS (formulas name the arms; rung 30's sign error is the reason):
  (a) THE GAIN CERTIFIES: median over the 8 windows of [L2_F_w(asm) - L2_F_w(norm)] >= 0.04 (S2128 saw +0.095 on
      FR; half survives averaging is the bar, matching rung 22's standard).
  (b) NOT ONE WINDOW'S STORY: the gain is positive on >= 7 of 8 windows.
  (c) REPRODUCTION: the norm arm's FR L2_F reproduces 2.6735 within 0.01 (pipeline unchanged).

Writes frontier_asm_fresh8_results.json. Self-reviewed. RUNG-30 HEADER FOLLOWS.

INSTALL THE CERTIFIED SELECTOR INTO THE FRONTIER (rung 30). The observability arc's certified, label-free
gain (§2116/§2119/§2124: mlp4/mlp5 CP units ranked by the true-Fisher top-8 at blocks 5/6, +0.082-0.086 median on
eight fresh windows) was shown on cfgE, the all-attention-real arm. The quotable frontier is §312's empirical-L2
config (+2.6735 fresh: empirical base + 38 motif heads + tail-attention dictionaries), whose middle MLPs use the
SAME norm-selected top-2304 CP construction. This reruns §312's pipeline twice — norm selection vs true-Fisher
top-8 selection at mlp4/mlp5 (c6-c9 stay norm, §2106) — and re-measures the frontier.

REGISTERED PREDICTIONS:
  (a) REPRODUCTION GATE: the norm arm's L2_F reproduces the published 2.6735 within 0.05.
  (b) THE GAIN INSTALLS: L2_F(norm) - L2_F(fisher8) >= 0.04 — half of cfgE's fresh median 0.086, since the
      frontier config carries motif-head and tail-dictionary error the selector does not touch. If FAILED, the
      certified gain does not survive composition with the head hybrids and tail refits, and the frontier keeps
      norm selection.
  (c) NO IN-DISTRIBUTION HARM: L2_C(norm) - L2_C(fisher8) >= -0.01.

Writes frontier_fisher8_results.json. Self-reviewed. ORIGINAL §312 HEADER FOLLOWS.

HIGH-COVERAGE VIA LOCAL PRICING -- the user's question sharpened the
objective: coverage is the goal; the pricing principle is the tool for
the HIGH-coverage end. Prediction from 311: at full-band coverage the
EMPIRICAL base wins (its block 2-9 streams are faithful where the
motif heads read), even though the fold base wins at low coverage.
Config: empirical matched-context base (20 comps) + 38 motif heads
(+2.287 fresh, measured) + tail-attention dictionaries a10-17 refit
under that exact stack -> 28 components + 38 heads.
REGISTERED PREDICTIONS:
  (a) total <= +2.75 fresh (beats fold-L2's +2.84 AND the old 34-comp
      +2.93 at comparable coverage);
  (b) the tail-attn increment lands in [0.30, 0.55] (pricing: empirical
      late streams are mid-faithful; fold's increment was 0.34 on
      better late streams);
  (c) window-C total reported."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['attn16_three_heads_results.json','empirical_L2_results.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    _p=json.load(open(_bq+'empirical_L2_results.json'))
    print(f"DRYRUN OK: published L2_F {_p['L2_F']}; nine-head vs three-head skip-a16 arms x eight windows (damage convention)")
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from circuit_dictionary import classify, COMPS as TAILC, CLS
D=1152; V=50257
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'attn16_three_heads_w8_results.json'
CA,CB=300,512; R0,R1=120,300
CONSTN={'digit','bclose','sentend','comma','name','rep'}
CONSTK=[k for k,nm in enumerate(CLS) if nm in CONSTN]
LINK=[k for k in range(10) if k not in CONSTK]
ATTM=[2,3,4,5,6,7,8,9]; ATTT=[10,11,12,13,14,15,16,17]
MIDL=(4,5,6,7,8,9)
TRI=torch.triu_indices(32,32)
SEL={'mode':'norm','P8':{}}
SKIP8=64


def select_units(li,L,Rw,Dw):
    if SEL['mode']=='fisher8' and li in (4,5):
        Pk=SEL['P8'][li+1]
        imp=(Pk.T@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
    elif SEL['mode']=='fisher8all' and li in (4,5,6,7,8,9):
        Pk=SEL['P8'][li+1]
        imp=(Pk.T@Dw).norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
    else:
        imp=Dw.norm(dim=0)*L.norm(dim=1)*Rw.norm(dim=1)
    return imp.argsort(descending=True)[:(SEL.get('K',2304) if li in (4,5)
                                          else SEL.get('K69MAP',{}).get(li,SEL.get('K69',2304)))]


def fisher_top8(site):
    """true-Fisher (y~p, 2 samples) top-8 at a block input, on the FIT rows; §2124's construction, seed 29."""
    genF=torch.Generator(device=DEV).manual_seed(29)
    TOKS=torch.cat([FW[i:i+4,:257] for i in range(CA,CB,4)]).to(DEV)
    G=torch.zeros(D,D,device=DEV,dtype=torch.float64)
    for b0 in range(0,TOKS.shape[0],4):
        idx=TOKS[b0:b0+4,:-1]
        with torch.no_grad():
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for _li,blk in enumerate(m.transformer.h): x,v1=blk(x,v1,x0)
            p=torch.softmax((30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30))[:,:-1].float(),-1)
        for _s in range(2):
            y=torch.multinomial(p.reshape(-1,p.shape[-1]),1,generator=genF).view(p.shape[0],p.shape[1])
            with torch.enable_grad():
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None; leaf=None
                for _li,blk in enumerate(m.transformer.h):
                    if _li==site:
                        x=x.detach().requires_grad_(True); leaf=x
                    x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                lp=F.log_softmax(lg[:,:-1],-1)
                (-lp.gather(-1,y[...,None]).squeeze(-1))[:,SKIP8:].sum().backward()
            g=leaf.grad[:,SKIP8:-1].reshape(-1,D).double(); G+=g.T@g
            m.zero_grad(set_to_none=True)
    _e,Q=torch.linalg.eigh(G)
    return Q.flip(1)[:,:8].float().contiguous()

@torch.no_grad()
def main():
    t0=time.time()
    S={}; cur={}
    vdir={}
    for li in (0,2,3):
        mlp=m.transformer.h[li].mlp
        _,_,Vh=torch.linalg.svd(
            torch.cat([mlp.Left.weight.detach().float(),
                       mlp.Right.weight.detach().float()]),
            full_matrices=False)
        vdir[li]=Vh[:32].T.contiguous()
    def quadfeat(X,li):
        Z=X@vdir[li]
        iu,il=TRI
        return torch.cat([X,Z[:,iu]*Z[:,il]],1)
    tab=torch.zeros(V,D,device=DEV,dtype=torch.float16)
    for s0 in range(0,V,4096):
        idx=torch.arange(s0,min(s0+4096,V),device=DEV)[:,None]
        x=F.rms_norm(m.transformer.wte(idx),(D,))
        hc=F.rms_norm((m.transformer.h[0].lambdas[0]
                       +m.transformer.h[0].lambdas[1])*x,(D,))
        tab[s0:s0+idx.shape[0]]=m.transformer.h[0].attn.c_v(hc)[:,0]\
            .to(torch.float16)
    S['a0']=('cv',0,tab)
    spans={}
    for li in TAILC:
        accs=[]
        for i in range(0,120,6):
            acc=[]; fwd(FW[i:i+6,:513].to(DEV), collect=li, acc=acc)
            accs.append(acc[0])
        Y=torch.cat(accs); Yb=Y.mean(0)
        _,_,Vh=torch.linalg.svd((Y-Yb).float(), full_matrices=False)
        spans[li]=(orth(Vh[:8].T),Yb.float())
    clsA=classify(CA,CB).to(DEV).reshape(CB-CA,256)
    clsC=classify(R0,R1).to(DEV).reshape(R1-R0,256)
    flatA=clsA.reshape(-1)
    for li in MIDL:
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float()
        Rw=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        db=mlp.Down_bias.detach().float()
        keep=select_units(li,L,Rw,Dw)
        S[f'c{li}']=('cp',li,L[keep].contiguous(),Rw[keep].contiguous(),
                     Dw[:,keep].contiguous(),db)
    def install(active):
        alist=[nm for nm in active if S[nm][0]=='attnd']
        hs=[]
        for nm in active:
            kind=S[nm][0]
            if kind=='cv':
                _,li,tb=S[nm]
                def h(mod,i_,o_,tb=tb):
                    return tb[cur['idx']].to(o_.dtype)
                hs.append(m.transformer.h[li].attn.c_v
                          .register_forward_hook(h))
            elif kind=='tableres':
                _,li,tb,A,P=S[nm]
                def h(mod,i_,o_,tb=tb,A=A,P=P,li=li):
                    x=i_[0].float().reshape(-1,D)
                    ft=quadfeat(x,li)
                    new=tb[cur['idx']].float().reshape(-1,D)+(ft@A)@P.T
                    return new.view(o_.shape).to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h))
            elif kind=='linear':
                _,li,W,b=S[nm]
                def h(mod,i_,o_,W=W,b=b):
                    x=i_[0].float().reshape(-1,D)
                    return (x@W+b).view(o_.shape).to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h))
            elif kind=='cp':
                _,li,Lk,Rk,Dk,db=S[nm]
                def h(mod,i_,o_,Lk=Lk,Rk=Rk,Dk=Dk,db=db):
                    x=i_[0].float()
                    return (((x@Lk.T)*(x@Rk.T))@Dk.T+db).to(o_.dtype)
                hs.append(m.transformer.h[li].mlp
                          .register_forward_hook(h))
            elif kind=='attnd':
                _,li,CV,LW,Wp2=S[nm]
                first=(len(alist)>0 and nm==alist[0])
                def h(mod,i_,o_,CV=CV,LW=LW,Wp2=Wp2,first=first):
                    y,v1=o_
                    x=i_[0].float().reshape(-1,D)
                    if first and cur['mode']=='probe':
                        cur['lab']=(x@Wp2).argmax(1)
                    c=cur['lab']
                    new=CV[c].clone()
                    for k in LINK:
                        sel=c==k
                        if sel.any(): new[sel]=x[sel]@LW[k]
                    return (new.view(y.shape).to(y.dtype),v1)
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(h))
            elif kind=='attnz':
                _,li=S[nm]
                def h(mod,i_,o_):
                    y,v1=o_
                    return (torch.zeros_like(y),v1)
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(h))
            elif kind=='attnm':
                _,li,mu=S[nm]
                def h(mod,i_,o_,mu=mu):
                    y,v1=o_
                    return (mu.expand_as(y).to(y.dtype),v1)
                hs.append(m.transformer.h[li].attn
                          .register_forward_hook(h))
            elif kind=='tail':
                _,Wp,DICT,LIN=S[nm]
                for li in TAILC:
                    Q,_=spans[li]
                    def h(mod,i_,o_,li=li,Q=Q,first=(li==TAILC[0]),
                          Wp=Wp,DICT=DICT,LIN=LIN):
                        B,T,_=o_.shape
                        x=i_[0].float().reshape(-1,D)
                        c=o_.float().reshape(-1,D)@Q
                        if first: cur['kk']=(x@Wp).argmax(1)
                        kk=cur['kk']
                        tgt=DICT[li][kk].clone()
                        for k in (8,9):
                            sel=kk==k
                            if sel.any(): tgt[sel]=x[sel]@LIN[li][k]
                        delta=((c-tgt)@Q.T).view(B,T,D)
                        return o_-delta.to(o_.dtype)
                    hs.append(m.transformer.h[li].mlp
                              .register_forward_hook(h))
        return hs
    def runA(active, cap_mod):
        Ys=[]; Xs=[]; Ids=[]
        hs=install(active)
        def capf(mod,i_,o_):
            Ys.append((o_[0] if isinstance(o_,tuple) else o_)
                      .detach().reshape(-1,D).float())
            Xs.append(i_[0].detach().reshape(-1,D).float())
        hs.append(cap_mod.register_forward_hook(capf))
        for i in range(CA,CB,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous()
            cur['mode']='oracle'
            cur['lab']=clsA[i-CA:i-CA+4].reshape(-1)
            m(cur['idx'], bb[:,1:].contiguous())
            Ids.append(cur['idx'].reshape(-1))
        for h in hs: h.remove()
        return torch.cat(Ys),torch.cat(Xs),torch.cat(Ids)
    def fit_table(Y,ids):
        sums=torch.zeros(V,D,device=DEV); cnt=torch.zeros(V,device=DEV)
        cnt.index_add_(0,ids,torch.ones_like(ids,dtype=torch.float))
        sums.index_add_(0,ids,Y)
        t=sums/cnt.clamp_min(1)[:,None]; t[cnt==0]=Y.mean(0)
        return t.to(torch.float16)
    def fit_tableres(Y,X,ids,li):
        tb=fit_table(Y,ids)
        Rr=Y-tb[ids].float()
        _,_,Vh=torch.linalg.svd(Rr[:30000], full_matrices=False)
        P=orth(Vh[:64].T)
        ft=quadfeat(X,li)
        lam=1e-2*len(X)
        A=torch.linalg.solve(ft.T@ft+lam*torch.eye(ft.shape[1],device=DEV),
                             ft.T@(Rr@P))
        return ('tableres',li,tb,A,P)
    Yoh=torch.zeros(len(flatA),10,device=DEV)
    Yoh[torch.arange(len(flatA)),flatA]=1.0
    Wp2=None
    def fit_attnd(li,active):
        nonlocal Wp2
        Y,X,ids=runA(active,m.transformer.h[li].attn)
        if Wp2 is None:
            lam=1e-2*len(X)
            Wp2=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),
                                   X.T@Yoh)
            acc=float(((X@Wp2).argmax(1)==flatA).float().mean())
            print(f'a{li}-input probe acc {acc:.2f}',flush=True)
        CV=torch.stack([Y[flatA==k].mean(0) if (flatA==k).sum()>0
                        else Y.mean(0) for k in range(10)])
        LW={}
        for k in LINK:
            mk=flatA==k
            Xk=X[mk]; Yk=Y[mk]
            l2=1e-2*max(len(Xk),1)
            LW[k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),
                                     Xk.T@Yk)
        print(f'fit a{li}',flush=True)
        return ('attnd',li,CV,LW,Wp2)
    order=['a0']
    Y,X,ids=runA(order,m.transformer.h[0].mlp)
    S['m0']=fit_tableres(Y,X,ids,0); order.append('m0')
    Y,X,ids=runA(order,m.transformer.h[1].attn.c_v)
    S['a1v']=('cv',1,fit_table(Y,ids)); order.append('a1v')
    Y,X,ids=runA(order,m.transformer.h[1].mlp)
    lam=1e-2*len(X)
    W1=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),X.T@Y)
    b1=Y.mean(0)-X.mean(0)@W1
    S['m1']=('linear',1,W1,b1); order.append('m1')
    S['a2']=fit_attnd(2,order); order.append('a2')
    Y,X,ids=runA(order,m.transformer.h[2].mlp)
    S['m2']=fit_tableres(Y,X,ids,2); order.append('m2')
    S['a3']=fit_attnd(3,order); order.append('a3')
    Y,X,ids=runA(order,m.transformer.h[3].mlp)
    S['m3']=fit_tableres(Y,X,ids,3); order.append('m3')
    for li in MIDL:
        S[f'a{li}']=fit_attnd(li,order)
        order.append(f'a{li}'); order.append(f'c{li}')
    for li in ATTT:
        S[f'a{li}']=fit_attnd(li,order); order.append(f'a{li}')
    capsT={li:[] for li in TAILC}; capsI={li:[] for li in TAILC}
    hs=install(order)
    for li in TAILC:
        def mk(li=li):
            def h(mod,i_,o_):
                capsT[li].append(o_.detach().reshape(-1,D).float())
                capsI[li].append(i_[0].detach().reshape(-1,D).float())
            return h
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mk()))
    for i in range(CA,CB,4):
        bb=FW[i:i+4,:257].to(DEV)
        cur['idx']=bb[:,:-1].contiguous()
        cur['mode']='oracle'; cur['lab']=clsA[i-CA:i-CA+4].reshape(-1)
        m(cur['idx'], bb[:,1:].contiguous())
    for h in hs: h.remove()
    X10=torch.cat(capsI[TAILC[0]])
    lam=1e-2*len(X10)
    Wp=torch.linalg.solve(X10.T@X10+lam*torch.eye(D,device=DEV),X10.T@Yoh)
    DICT={}; LIN={}
    for li in TAILC:
        Q,_=spans[li]; C=torch.cat(capsT[li])@Q
        DICT[li]=torch.stack([C[flatA==k].mean(0) if (flatA==k).sum()>0
                              else C.mean(0) for k in range(10)])
        Xl=torch.cat(capsI[li]); LIN[li]={}
        for k in (8,9):
            mk_=flatA==k
            Xk=Xl[mk_]; Ck=C[mk_]
            l2=1e-2*len(Xk)
            LIN[li][k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),
                                          Xk.T@Ck)
        capsT[li]=None; capsI[li]=None
    S['tail']=('tail',Wp,DICT,LIN); order.append('tail')
    def evalC(active, mode):
        hs=install(active)
        ces=[]
        for i in range(R0,R1,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['mode']=mode
            if mode=='oracle':
                cur['lab']=clsC[i-R0:i-R0+4].reshape(-1)
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,)); x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    # ---- fold tables (weights-only) ----
    FOLD={}
    capsF={0:[],2:[],3:[]}
    hsF=[]
    for li in (0,2,3):
        def mkf(li=li):
            def h(mo,i_,o_):
                capsF[li].append(o_.detach()[:,0].float())
            return h
        hsF.append(m.transformer.h[li].mlp.register_forward_hook(mkf()))
    for s0 in range(0,V,2048):
        idx=torch.arange(s0,min(s0+2048,V),device=DEV)[:,None]
        xF=F.rms_norm(m.transformer.wte(idx),(D,)); x0F=xF; v1F=None
        for blk in m.transformer.h[:4]:
            xF,v1F=blk(xF,v1F,x0F)
    for h in hsF: h.remove()
    for li in (0,2,3):
        FOLD[li]=torch.cat(capsF[li]).to(torch.float16); capsF[li]=None
    # ---- CP middles (weights-only) ----
    for li in (4,5,6,7,8,9):
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float()
        Rw=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        db=mlp.Down_bias.detach().float()
        keep=select_units(li,L,Rw,Dw)
        S[f'c{li}']=('cp',li,L[keep].contiguous(),Rw[keep].contiguous(),
                     Dw[:,keep].contiguous(),db)
    # ---- matched-context sequential fits, both arms ----
    def build_arm(fold):
        S3={}
        def fit_res(li,tb,active):
            Y,X,ids=runA(active,m.transformer.h[li].mlp)
            if tb is None:
                tb=fit_table(Y,ids)
            Rr=Y-tb[ids].float()
            _,_,Vh2=torch.linalg.svd(Rr[:30000],full_matrices=False)
            P=orth(Vh2[:64].T)
            ft=quadfeat(X,li)
            lam=1e-2*len(X)
            A=torch.linalg.solve(ft.T@ft+lam*torch.eye(ft.shape[1],
                                                       device=DEV),
                                 ft.T@(Rr@P))
            return ('tableres',li,tb,A,P)
        pre='F' if fold else 'E'
        S[f'm0{pre}']=fit_res(0,FOLD[0] if fold else None,['a0'])
        act=['a0',f'm0{pre}','a1v','m1']
        S[f'm2{pre}']=fit_res(2,FOLD[2] if fold else None,act)
        act=act+[f'm2{pre}']
        S[f'm3{pre}']=fit_res(3,FOLD[3] if fold else None,act)
        front=['a0',f'm0{pre}','a1v','m1',f'm2{pre}',f'm3{pre}']
        stack=front+[f'c{li}' for li in (4,5,6,7,8,9)]
        # tail refit under this exact stack (attention real)
        capsT={li:[] for li in TAILC}; capsI={li:[] for li in TAILC}
        hs=install(stack)
        for li in TAILC:
            def mk2(li=li):
                def h(mo,i_,o_):
                    capsT[li].append(o_.detach().reshape(-1,D).float())
                    capsI[li].append(i_[0].detach().reshape(-1,D)
                                     .float())
                return h
            hs.append(m.transformer.h[li].mlp
                      .register_forward_hook(mk2()))
        for i in range(CA,CB,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous()
            cur['mode']='oracle'
            cur['lab']=clsA.reshape(CB-CA,256)[i-CA:i-CA+4].reshape(-1)
            m(cur['idx'], bb[:,1:].contiguous())
        for h in hs: h.remove()
        X10=torch.cat(capsI[TAILC[0]])
        Yoh=torch.zeros(len(flatA),10,device=DEV)
        Yoh[torch.arange(len(flatA)),flatA]=1.0
        lam=1e-2*len(X10)
        Wp=torch.linalg.solve(X10.T@X10+lam*torch.eye(D,device=DEV),
                              X10.T@Yoh)
        DICT={}; LIN={}
        for li in TAILC:
            Q,_=spans[li]; C=torch.cat(capsT[li])@Q
            DICT[li]=torch.stack([C[flatA==k].mean(0)
                                  if (flatA==k).sum()>0 else C.mean(0)
                                  for k in range(10)])
            Xl=torch.cat(capsI[li]); LIN[li]={}
            for k in (8,9):
                mk_=flatA==k
                Xk=Xl[mk_]; Ck=C[mk_]
                l2=1e-2*len(Xk)
                LIN[li][k]=torch.linalg.solve(
                    Xk.T@Xk+l2*torch.eye(D,device=DEV),Xk.T@Ck)
            capsT[li]=None; capsI[li]=None
        S[f'tail{pre}']=('tail',Wp,DICT,LIN)
        return stack+[f'tail{pre}']
    cfgF=build_arm(False)  # EMPIRICAL twin base
    def evalT(TOK,N,active):
        hs=install(active)
        ces=[]
        for i in range(0,N,4):
            bb=TOK[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['mode']='oracle'
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,))
            x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    import tiktoken
    from datasets import load_dataset
    enc3=tiktoken.get_encoding('gpt2')
    dsf=load_dataset('NeelNanda/pile-10k',split='train')
    seen={tuple(FW[r,:32].tolist()) for r in range(FW.shape[0])}
    rows=[]
    for di in range(3000,10000):
        tk=enc3.encode_ordinary(dsf[di]['text'])
        for st0 in range(0,len(tk)-513,513):
            row=tk[st0:st0+513]
            if tuple(row[:32]) in seen: continue
            rows.append(row)
            if len(rows)>=120: break
        if len(rows)>=120: break
    FR=torch.tensor(rows,dtype=torch.long)
    baseC=evalT(FW[R0:R1],R1-R0,[])
    baseF=evalT(FR,120,[])
    if SEL.get('zero6'): SEL['zh_on']=True
    # ---- L1: motif gains fit under cfgF ----
    mt=json.load(open(PT+'attn_motifs3_results.json'))['motif_table']
    prevh={}; selfh={}
    for li,hd,mo,fr in mt:
        if 2<=li<=9:
            if mo=='prev': prevh.setdefault(li,[]).append(hd)
            if mo=='self': selfh.setdefault(li,[]).append(hd)
    mod2=sys.modules[type(m.transformer.h[0].attn).__module__]
    are=mod2.apply_rotary_emb
    T=256
    def head_z(at,X2,v1):
        B=X2.shape[0]
        q=at.c_q(X2).view(B,T,9,128); k=at.c_k(X2).view(B,T,9,128)
        q2=at.c_q2(X2).view(B,T,9,128); k2=at.c_k2(X2).view(B,T,9,128)
        v=at.c_v(X2).view(B,T,9,128)
        if v1 is None: v1=v
        vm=(1-at.lamb)*v+at.lamb*v1.view_as(v)
        cos,sin=at.rotary(q)
        qn=F.rms_norm(q,(128,)); kn=F.rms_norm(k,(128,))
        qn,kn=are(qn,cos,sin),are(kn,cos,sin)
        q2n=F.rms_norm(q2,(128,)); k2n=F.rms_norm(k2,(128,))
        q2n,k2n=are(q2n,cos,sin),are(k2n,cos,sin)
        sc=torch.einsum('bqhd,bkhd->bhqk',qn.float(),kn.float())/128
        sc2=torch.einsum('bqhd,bkhd->bhqk',q2n.float(),k2n.float())/128
        pat=(sc*sc2)*torch.tril(torch.ones(T,T,device=DEV))
        z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
        return z,vm.float()
    caps={li:{'x':[],'v1':[]} for li in range(2,10)}
    hs=install(cfgF)
    for li in range(2,10):
        def mkc(li=li):
            def h(mo_,args):
                caps[li]['x'].append(args[0].detach())
                caps[li]['v1'].append(args[1].detach()
                                      if args[1] is not None else None)
            return h
        hs.append(m.transformer.h[li].attn
                  .register_forward_pre_hook(mkc()))
    for i in range(CA,CA+32,4):
        bb=FW[i:i+4,:257].to(DEV)
        cur['idx']=bb[:,:-1].contiguous()
        m(cur['idx'], bb[:,1:].contiguous())
    for h in hs: h.remove()
    ALPHA={}
    for li in range(2,10):
        at=m.transformer.h[li].attn
        num=torch.zeros(9,device=DEV); den=torch.zeros(9,device=DEV)
        nums=torch.zeros(9,device=DEV); dens=torch.zeros(9,device=DEV)
        for X2,v1 in zip(caps[li]['x'],caps[li]['v1']):
            z,vm=head_z(at,X2,v1)
            vp=torch.zeros_like(vm); vp[:,1:]=vm[:,:-1]
            vp=vp.permute(0,2,1,3); vs=vm.permute(0,2,1,3)
            num+=(z*vp).sum((0,2,3)); den+=(vp*vp).sum((0,2,3))
            nums+=(z*vs).sum((0,2,3)); dens+=(vs*vs).sum((0,2,3))
        ALPHA[li]=(num/den.clamp_min(1e-9),nums/dens.clamp_min(1e-9))
        caps[li]=None
    def motif_hooks(layers):
        hs2=[]
        for li in layers:
            if li not in set(list(prevh)+list(selfh)): continue
            at=m.transformer.h[li].attn
            ap,asf=ALPHA[li]
            ph=prevh.get(li,[]); sh=selfh.get(li,[])
            def h(mo_,args,out,at=at,ph=ph,sh=sh,ap=ap,asf=asf):
                y,v1r=out
                X2=args[0]; v1=args[1] if args[1] is not None else v1r
                z,vm=head_z(at,X2,v1)
                vp=torch.zeros_like(vm); vp[:,1:]=vm[:,:-1]
                vp=vp.permute(0,2,1,3); vs=vm.permute(0,2,1,3)
                for hd in ph: z[:,hd]=ap[hd]*vp[:,hd]
                for hd in sh: z[:,hd]=asf[hd]*vs[:,hd]
                B=X2.shape[0]
                ynew=at.c_proj(z.transpose(1,2).contiguous()
                               .view(B,T,-1).to(X2.dtype))
                return (ynew,v1r)
            hs2.append(at.register_forward_hook(h))
        return hs2
    def evalM(TOK,N,active,mlayers):
        hs=install(active)+motif_hooks(mlayers)
        ces=[]
        for i in range(0,N,4):
            bb=TOK[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            cur['mode']='oracle'
            if 'clsmap' in cur:
                cur['lab']=cur['clsmap'][i:i+4].reshape(-1)
            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,))
            x0=x; v1=None
            for blk in m.transformer.h:
                x,v1=blk(x,v1,x0)
            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                       reduction='none'))
        for h in hs: h.remove()
        return float(torch.cat(ces).mean())
    ML=list(range(2,10))
    L1C=evalM(FW[R0:R1],R1-R0,cfgF,ML)-baseC
    L1F=evalM(FR,120,cfgF,ML)-baseF
    print(f'L1 (+38 heads): C {L1C:+.4f} | fresh {L1F:+.4f}',flush=True)
    # tail-attention dicts refit under (empirical base + motifs)
    Yoh=torch.zeros(len(flatA),10,device=DEV)
    Yoh[torch.arange(len(flatA)),flatA]=1.0
    order2=list(cfgF)
    Wp2=None
    ML=list(range(2,10))
    for li in range(10,18):
        if SEL.get('skip16') and li==16: continue
        Ys=[]; Xs=[]
        hs=install(order2)+motif_hooks(ML)
        def cap2(mo_,i_,o_):
            Ys.append((o_[0]).detach().reshape(-1,D).float())
            Xs.append(i_[0].detach().reshape(-1,D).float())
        hs.append(m.transformer.h[li].attn.register_forward_hook(cap2))
        for i in range(CA,CB,4):
            bb=FW[i:i+4,:257].to(DEV)
            cur['idx']=bb[:,:-1].contiguous()
            cur['mode']='oracle'
            cur['lab']=clsA.reshape(CB-CA,256)[i-CA:i-CA+4].reshape(-1)
            m(cur['idx'], bb[:,1:].contiguous())
        for h in hs: h.remove()
        Y=torch.cat(Ys); X2=torch.cat(Xs)
        if Wp2 is None:
            lam=1e-2*len(X2)
            Wp2=torch.linalg.solve(X2.T@X2+lam*torch.eye(D,device=DEV),
                                   X2.T@Yoh)
        CV=torch.stack([Y[flatA==k].mean(0) if (flatA==k).sum()>0
                        else Y.mean(0) for k in range(10)])
        LW={}
        for k in LINK:
            mk_=flatA==k
            Xk=X2[mk_]; Yk=Y[mk_]
            l2=1e-2*max(len(Xk),1)
            LW[k]=torch.linalg.solve(Xk.T@Xk+l2*torch.eye(D,device=DEV),
                                     Xk.T@Yk)
        S[f'a{li}L']=('attnd',li,CV,LW,Wp2)
        order2.append(f'a{li}L')
        print(f'fit a{li}L',flush=True)
    cur['clsmap']=clsC.reshape(R1-R0,256)
    L2C=evalM(FW[R0:R1],R1-R0,order2,ML)-baseC
    del cur['clsmap']
    import tiktoken as tk2
    enc4=tk2.get_encoding('gpt2')
    def classify2(Tk):
        n=Tk.shape[0]
        Mid=torch.zeros(n,256,dtype=torch.long)
        for r in range(n):
            toks=Tk[r,:257].tolist()
            for pos in range(256):
                t=toks[pos+1]; p=toks[pos]
                tg=enc4.decode([t]); pv=enc4.decode([p]); st=tg.strip()
                if st.isdigit() and not tg.startswith(' '): k=0
                elif st in (')',']') and any(b in enc4.decode(
                    toks[max(0,pos-60):pos+1]) for b in ('(','[')): k=1
                elif chr(10) in tg: k=2
                elif tg in ('.','!','?'): k=3
                elif tg==',': k=4
                elif (tg.startswith(' ') and st[:1].isupper() and
                      (pv.strip()[:1].isupper() if pv.strip()
                       else False)): k=5
                elif t==p: k=6
                elif (not tg.startswith(' ')) and st.isalpha(): k=7
                elif t in toks[:pos+1]: k=8
                else: k=9
                Mid[r,pos]=k
        return Mid
    cur['clsmap']=classify2(FR).to(DEV)
    L2F=evalM(FR,120,order2,ML)-baseF
    if SEL.get('head16'):
        HD16=D//9
        def _zh(hh):
            def pre(mod,args):
                x=args[0].clone(); x[...,hh*HD16:(hh+1)*HD16]=0
                return (x,)+tuple(args[1:])
            return m.transformer.h[16].attn.c_proj.register_forward_pre_hook(pre)
        hks=[_zh(k2) for k2 in (1,2,5,6,7,8)]
        v=evalM(FR,120,order2,ML)-baseF
        for k3 in hks: k3.remove()
        SEL['head16_result']={'six_zeroed_L2F':round(v,4),'d':round(v-L2F,4)}
        print(f'  heads 1,2,5,6,7,8 zeroed: L2 fresh {v:+.4f}  (d={v-L2F:+.4f})',flush=True)
    del cur['clsmap']
    if SEL.get('prefix_tail'):
        cur['clsmap']=classify2(FR).to(DEV)
        _pl=[]
        for _kk in range(0,9):
            _act=order2[:len(order2)-8+_kk]
            _v=evalM(FR,120,_act,ML)-baseF
            _pl.append(round(_v,4))
            print(f'  prefix +{_kk} tail-attn dicts: L2 fresh {_v:+.4f}',flush=True)
        del cur['clsmap']
        SEL['prefix_result']={'prefix':_pl,'marginals':[round(_pl[_i+1]-_pl[_i],4) for _i in range(8)]}
    W8banned=set(); W8RES=[]
    for wi in range(8):
        rws=[]; used=set()
        for di in range(3000,10000):
            if di in W8banned: continue
            tkr=enc3.encode_ordinary(dsf[di]['text'])
            for st0 in range(0,len(tkr)-513,513):
                row=tkr[st0:st0+513]
                if tuple(row[:32]) in seen: continue
                rws.append(row); used.add(di)
                if len(rws)>=120: break
            if len(rws)>=120: break
        assert len(rws)==120, f'window {wi} short: {len(rws)}'
        W8banned|=used
        for row in rws: seen.add(tuple(row[:32]))
        Wt=torch.tensor(rws,dtype=torch.long)
        bW=evalT(Wt,120,[])
        cur['clsmap']=classify2(Wt).to(DEV)
        fW=evalM(Wt,120,order2,ML)-bW
        del cur['clsmap']
        W8RES.append(round(fW,4))
        print(f'  window {wi} ({len(used)} docs): L2 fresh {fW:+.4f}',flush=True)
    inc=L2F-L1F
    print(f'L2 empirical: C {L2C:+.4f} | fresh {L2F:+.4f} | tail-attn '
          f'increment {inc:+.4f}',flush=True)
    if SEL.get('collect_asm'):
        genA=torch.Generator(device=DEV).manual_seed(32)
        TOKS=torch.cat([FW[i:i+4,:257] for i in range(CA,CB,4)]).to(DEV)
        for site in (5,6,7,8,9,10):
            Gm=torch.zeros(D,D,device=DEV,dtype=torch.float64)
            for b0 in range(0,TOKS.shape[0],4):
                bb=TOKS[b0:b0+4]
                cur['idx']=bb[:,:-1].contiguous(); cur['mode']='oracle'
                cur['lab']=clsA.reshape(CB-CA,256)[b0:b0+4].reshape(-1)
                hs=install(order2)+motif_hooks(ML)
                try:
                    with torch.no_grad():
                        x=F.rms_norm(m.transformer.wte(cur['idx']),(D,)); x0=x; v1=None
                        for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                        pdist=torch.softmax((30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30))[:,:-1].float(),-1)
                    for _sm in range(2):
                        y=torch.multinomial(pdist.reshape(-1,pdist.shape[-1]),1,generator=genA).view(pdist.shape[0],pdist.shape[1])
                        with torch.enable_grad():
                            x=F.rms_norm(m.transformer.wte(cur['idx']),(D,)); x0=x; v1=None; leaf=None
                            for _li,blk in enumerate(m.transformer.h):
                                if _li==site:
                                    x=x.detach().requires_grad_(True); leaf=x
                                x,v1=blk(x,v1,x0)
                            lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
                            lp=F.log_softmax(lg[:,:-1],-1)
                            (-lp.gather(-1,y[...,None]).squeeze(-1))[:,SKIP8:].sum().backward()
                        g=leaf.grad[:,SKIP8:-1].reshape(-1,D).double(); Gm+=g.T@g
                        m.zero_grad(set_to_none=True)
                finally:
                    for h in hs: h.remove()
            _e,Qm=torch.linalg.eigh(Gm)
            SEL['P8'][site]=Qm.flip(1)[:,:8].float().contiguous()
            print(f'assembly-conditioned Fisher top-8 collected at site {site}',flush=True)
    pa=L2F<=2.75; pb=0.30<=inc<=0.55
    out={'L1_F':round(L1F,4),'L2_C':round(L2C,4),'L2_F':round(L2F,4),
         'increment':round(inc,4),
         'orig_s312_a':bool(pa),'orig_s312_b':bool(pb)}
    print(f"(a) L2 <= +2.75 fresh: {'HELD' if pa else 'FAILED'}")
    print(f"(b) increment in [0.30,0.55]: {'HELD' if pb else 'FAILED'}")
    out['fresh8']=W8RES
    out['runtime_s']=time.time()-t0
    return out

if __name__=='__main__':
    T00=time.time()
    HD16=D//9
    def _mkzh(hh):
        def pre(mod,args):
            if not SEL.get('zh_on'): return None
            x=args[0].clone(); x[...,hh*HD16:(hh+1)*HD16]=0
            return (x,)+tuple(args[1:])
        return m.transformer.h[16].attn.c_proj.register_forward_pre_hook(pre)
    ZH=[_mkzh(k2) for k2 in (1,2,5,6,7,8)]
    SEL['mode']='norm'; SEL['K']=2304; SEL['K69']=576; SEL['K69MAP']={8:288,9:288}
    SEL['skip16']=True; SEL['zero6']=False; SEL['zh_on']=False
    print('ARM 1/2: skip-a16, all nine heads (S2146 reproduction)',flush=True)
    r_n=main()
    SEL['zero6']=True
    print('ARM 2/2: skip-a16, heads 16.1/2/5/6/7/8 zeroed throughout',flush=True)
    r_t=main()
    SEL['zh_on']=False
    for k3 in ZH: k3.remove()
    import statistics as stt
    S2146=[2.5091-0.1571+2.6735-2.6735,0,0,0,0,0,0,0]
    prev=json.load(open(PT+'frontier_skip_a16_results.json'))['skip_a16']['fresh8']
    d=[round(a-b,4) for a,b in zip(r_t['fresh8'],r_n['fresh8'])]
    rp=[abs(round(a-b,4)) for a,b in zip(r_n['fresh8'],prev)]
    md=stt.median(d)
    pa=md<=0.005
    pb=sum(g<=0.01 for g in d)>=6
    pc=stt.median(rp)<=0.005
    res={'nine':r_n,'three':r_t,'per_window_three_minus_nine':d,
         'per_window_nine_vs_s2146_absdelta':rp,
         'convention':'L2 = CE above the real model; lower is better',
         'pred_a_window_grain':bool(pa),'pred_b_windows':bool(pb),'pred_c_reproduces_s2146':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'three - nine per window: {d}  median {md:+.4f}')
    print(f"(a) median {md:+.4f} <= +0.005: {'HELD' if pa else 'FAILED'}")
    print(f"(b) <= +0.01 on {sum(g<=0.01 for g in d)}/8 (bar 6): {'HELD' if pb else 'FAILED'}")
    print(f"(c) nine-head vs S2146 per-window median |delta| {stt.median(rp):.4f} <= 0.005: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({res["runtime_s"]:.0f}s)')
