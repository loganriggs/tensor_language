# PENDING RETRACTION — §1612 membership claim (NOT APPLIED)

**Status: awaiting Logan. Nothing here has been applied to the ledger or the
registry.** Retracting a published claim is outside delegated autonomy (wake_prompt
AUTONOMY section). This file states exactly what would change so the decision is a
yes/no, not a research task.

## The claim to be withdrawn

In `theseus-bench/registry/circuits.json`, entry `S1612`:

- `membership_at_question`: "lambda top-4 = mlp17, mlp11, attn10, mlp9; random
  top-4 = mlp17, mlp10, mlp11, mlp16. FLOOR = {mlp17, mlp11}. RULE-SPECIFIC =
  {attn10, mlp9}, absent from random in 3/3."
- `WITHDRAWS`: "S1610's claim that attn10 is 'not distinguishable from floor'.
  That used the positive-only statistic. Under the correct absolute-mass statistic
  attn10 is ABSENT from the random top-4 in 3/3 — S1597's headline head SURVIVES
  the control."

## Why it is wrong (§1628, 3-for-3, 60 trials)

1. **Refuted at power.** attn10 sits in a random top-4 in **32/60 = 53.3%** of
   trials (20 independent rank-2 bases x 3 disjoint 160-row chunks); attn9 in
   **53/60 = 88.3%**. Six of twenty seeds never place attn10, seven place it 3/3 —
   membership is a property of the draw.
2. **Refuted on its own seed.** Under the corrected quantity, §1612's own basis
   (seed 1729) places attn10 in the top-4 in chunks [0, 2] — 2 of 3, not 0 of 3.
3. **Wrong component set entirely.** §1612's λ top-4 was
   `{mlp17, mlp11, attn10, mlp9}`, which contains **mlp17, downstream of the site**.
   Under the corrected quantity the λ top-4 is `{attn10, attn9, mlp9, mlp10}` —
   §1597's published set, stable 3/3. This was not a near-miss measurement.

## What the correction would say

- §1610's original claim — attn10 is **not distinguishable from floor by
  membership** — is CORRECT and is restored.
- §1612's `membership_at_question` and `WITHDRAWS` are withdrawn as artifacts of
  (a) the pre-§1623 wrong quantity and (b) a single random basis.
- **§1597 is NOT retracted.** Its share replicates exactly (.7179 vs .718, §1623)
  and separates **60/60** against a proper multi-seed null (λ min .7257 > random max
  .7185). Its head-grain claim (attn10 = head 10.5 at 20:1 within-layer) is a
  different, stronger statistic that §1628 does not test and does not touch.
- §1612's WITHIN-cell share conclusions are unaffected.

## Scope of the blast radius (checked, not assumed)

Sections citing the membership claim rather than the share claim would need the
same note. §1613/§1614/§1616 rest on the null SHARE spread, not on membership, and
are unaffected by this. §1624's pronouns verdict rests on share and is separately
under test (`pronouns_multiseed_null`) for the unrelated single-seed problem.

## SECOND PENDING ITEM (added 2026-08-27, §1634) — the pronoun STRUCTURAL READING

Separate from the membership claim above, and also NOT applied.

The S1612 entry contains `S1598_verdict_strengthened`: "Against the null the pronoun
slice is DRAMATICALLY less concentrated than a meaningless basis -- a positive
structural claim about distributed writing, not a negative result."

§1634 measured five ordinary function-word classes at pronouns' OWN site (mlp17,
rank-8, TOP-6, 20-seed null, 60 trials each). Below-null is GENERIC there: `at`
separates 60/60 with gap −.1094 against pronouns' −.1320, and four of five fresh
classes exceed 52/60. Pronouns is the most diffuse of the six but not qualitatively
distinct, so "dramatically less concentrated than a meaningless basis" describes
ordinary prepositions at mlp17 too and licenses no structural claim about pronouns.

Note this also WITHDRAWS §1630's restoration of §1612, which I made on mlp11 data —
the wrong site for a claim about mlp17. §1598's .482 and §1624's .4823 are
unaffected; this concerns interpretation, not measurement.

## WITHDRAWN MECHANISM (was added §1636, removed §1637)

A mechanism paragraph stood here claiming question@mlp11 is distinctive because mlp11
is the separation minimum. §1637 withdrew it: the mlp11 minimum is a FUNCTION-WORD
property, and `question` is punctuation. Against its same-type control `period`
(58/60 at the same cell and configuration) question's 60/60 is a margin of 2, not the
13 measured against function words.

This does NOT change what is being asked of Logan. The second pending item — that
§1612's pronoun structural reading is unsupported because below-null is generic at
mlp17 — rests on §1634's direct measurement at mlp17 and is unaffected. Both
withdrawals remain drafted and unapplied.

## STATUS 2026-08-27 12:06 — BOTH ITEMS FULLY EVIDENCED, NEITHER APPLIED

Item 1 (membership): §1628, 20 independent random bases x 3 disjoint chunks. attn10
appears in the random top-4 in 32/60 = 53.3% of trials, attn9 in 53/60 = 88.3%, and
§1612's OWN seed 1729 places attn10 in 2 of 3 chunks, contradicting "absent 3/3".

Item 2 (pronoun structural reading): confirmed TWICE on independent control sets.
AMENDED BY §1639 -- read the gap evidence, not the count evidence. At rank-8 TOP-6
the separation COUNT saturates (six unrelated cells all land at 57-60/60), so §1638's
"margin 0 in count" is close to uninformative on its own. The argument rests instead
on the mean GAPS, which are not saturated: pronouns -.1320 against it -.1190, we
-.0940, you -.0965 and ` I` -.1795. ` I` being MORE below-null than the certified
class is what carries it. The conclusion is unchanged; one leg of its support is not
load-bearing.
§1634 used five function-word classes at mlp17 (margin 0). §1638 answered the
subtype worry §1637 raised by using four OTHER PERSONAL PRONOUNS at the identical
cell -- ` it`, ` we`, ` I` all reach 60/60 and ` I` is MORE below-null (−.1795) than
the certified class (−.1320). Margin 0 again. The control is now matched on cell,
rank, TOP, rows, seeds, statistic and exact subtype.

Neither item requires further measurement. Both await a decision.

## The rule this earns either way

A top-K membership test against ONE random basis is a sample of size one dressed as
a control. **Report a fraction over many bases, or make no membership claim.**

AMENDED BY §1629: do NOT generalise this to "membership is worthless". At
pronouns@mlp17, `mlp16` is in 60/60 random top-6s (no information) while `x0` is in
only 9/60 = 15% (real information). Membership is uninformative by default and
occasionally informative, and only the measured fraction tells you which. §1628's
specific findings — attn10 at 53%, attn9 at 88% — are unaffected.
