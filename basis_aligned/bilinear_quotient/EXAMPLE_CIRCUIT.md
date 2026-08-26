# Worked example: the pronoun circuit (head 12.4) — full trace for criticism

Everything below is reproducible from the named scripts/results in this directory.

## 1. Discovery (circuit_screen4.py)
Class definition: positions whose TARGET token ∈ {' he',' she',' they',' He',' She',' They'}
(regex over the 50,257-token vocabulary). Discovery score per head (weights-only, no
forward passes): ||u_class · W_proj[:, head_slice]|| where u_class = normalized mean
unembedding row of the class tokens and W_proj[:, head_slice] = that head's 128-column
output block in its layer's projection matrix. Top-5: {12.4, 9.5, 10.5, 13.2, 15.6}.
Graded by removal: 41.5x selectivity at NR=480 (498 class positions).

## 2. Construction (circuit_greedy5.py)
Greedy over the top-12 scored candidates; accept a head only if class damage grows
>= 15% AND selectivity stays >= 90% of the best so far. Result: singleton {12.4} —
every additional head diluted selectivity. (Known boundary: this recipe needs
>= ~300 class positions; below that it adds noise — S1559.)

## 3. Removal certification (circuit_verify_g5.py, NR=1920)
- Rows: 1,920 held-out (skip=7000); scored positions/row: 192 → 368,640 total.
- Class positions: ~2,100. Removal = head 12.4's 128-dim projection-input slice
  replaced by its optimal constant (learned constant baseline from the
  198-component sweep, frozen).
- Class CE: 3.29 -> 3.51 (rise +.2162). Global CE rise: +.0006.
- Selectivity 347x. Screen-scale estimate was 275x; the direction of the
  correction was UP with more data.

## 4. What it is NOT (circuit_h124.py / h124b.py)
- NOT gender/number-specific: damage he .268 / she .269 / they .130 (within 3x).
- NOT a coreference tracker: with a powered control (623 pronoun positions lacking
  a capitalized token in the prior 16), antecedent-present vs -absent damage is
  1.05x. The antecedent-choice computation is elsewhere (distributed state — the
  same result as agreement and quote parity, S1550-51).
- It is an ANNOUNCER: raises pronoun probability at pronoun-appropriate positions.

## 5. Generalization status
- Row-set replication: verified (screen skip=7000 NR=480 -> NR=1920, rise within
  35% band; the certified ship work shows row-set spreads of ~.04 at this scale).
- Member-level: pronouns has 6 member tokens — too few for the per-token
  classifier leg (that leg needs large open classes; its best result is
  capitalization: Spearman .74, FN .62, FP .26; its worst: digits .09/.79/.47).

## 6. Known criticisms / open holes
1. Circuits are ATTENTION-HEAD ensembles only. No MLP or MLP-unit membership has
   been tested for any certified circuit (queued: unit-grain screen).
2. Removal uses optimal-constant substitution; mean-ablation or resampling
   ablation could give different selectivities (not cross-checked per circuit).
3. Class masks are regex-defined; boundary cases (capitalized mid-sentence 'He'
   after quotes etc.) are not audited.
4. Selectivity denominators for small classes are noisy (the S1523 >50x rule
   exists because one screen number inflated 4.4x); circuits with < ~300 class
   positions (titles, days, however) carry screen-scale error bars.
5. "Announcer" is certified by what removal does NOT damage; the mechanism by
   which 12.4 detects pronoun-appropriateness (its input side) is uncharacterized.
