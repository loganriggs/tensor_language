# Circuit campaign update — 2026-08-30 05:55 UTC

## UPDATE: the MLP0 section is closed

The two-hour cutoff has passed.  We are not reopening the MLP0 decomposition branch.
Its final useful conclusion is:

- the folded MLP0 map splits exactly into token-token, token-context, and
  context-context tensor terms;
- the split reconstructs the original map to relative MSE `3.11e-13` in full
  precision (`5.48e-6` in bf16);
- the downstream CE contributions interact strongly, so the three terms cannot be
  compressed and priced independently;
- the best current structural picture is a shared lexical/token DAG coupled to a
  continuous contextual tensor.

The detailed cutoff record is
[`TWO_HOUR_SECTION_DEADLINE_2026-08-30.md`](../TWO_HOUR_SECTION_DEADLINE_2026-08-30.md).

## UPDATE: what “high-quality circuit” means here

A behavior label plus an important head is not enough.  Each campaign circuit is
tracked on two separate axes.

The **mechanistic tier** asks how far its computation has been reduced:

1. causal localization;
2. a precise behavior and matched controls;
3. upstream writers/readers;
4. an executable tensor algebra that reproduces the behavior;
5. recursive reduction to tokens, positions, and other terminal primitives.

The **terminal certificate** asks whether that program is useful:

- **extraction:** can the tensor program restore the behavior after the native owner
  is deleted?
- **selective removal:** can we remove the behavior while leaving matched unrelated
  behavior intact?
- **OOD transport:** do the frozen effect and extraction survive new tokens,
  structures, or domains?
- **collateral:** is off-target CE small, not merely smaller than target CE?

Cross-entropy (CE) is the model's average next-token surprise in natural-log units.
Positive `dCE` after removal means the removed computation helped prediction.  A
document bootstrap resamples whole documents, preserving correlations among tokens;
the reported simultaneous 95% bounds cover every registered coordinate together.

The complete rubric is
[`TIER_RUBRIC.md`](../../bilinear_quotient/circuits/TIER_RUBRIC.md).  The ten separate
circuit dossiers are in
[`circuits/campaign_2026_08_30/`](../../bilinear_quotient/circuits/campaign_2026_08_30/README.md).

## UPDATE: induction equality fetching is real and OOD-predictive, but not surgically removable

The induction program uses four fixed attention heads: `L5H5`, `L7H3`, `L8H3`, and
`L8H4`.  Its key computation is an equality-fetch tensor.  In simplified notation,

\[
s(q,k) \propto
\langle e_{t_q}, e_{t_{k-1}}\rangle^2,
\]

so a query token scores earlier locations whose preceding token matches it.  The value
path then fetches the following token.  This is a fixed squared-bilinear attention
program; there is no TopK switch, decoded label, or target-position router.

The terminal run used 192 held-out natural documents and 192 held-out code documents.
For each document it ran six fixed arms:

1. native model;
2. exact full tensor replay;
3. all four heads deleted;
4. equality tensor added back to that deletion background;
5. an equal-price deranged-equality null;
6. equality contribution removed from the native computation.

The extraction-recovery computation is

\[
R = \frac{\mathrm{CE}_{\rm deleted}-\mathrm{CE}_{\rm extracted}}
         {\mathrm{CE}_{\rm deleted}-\mathrm{CE}_{\rm native}}.
\]

`R=1` means the extracted tensor restores the entire effect of deleting the four
heads; `R=0` means it restores none.

### What passed

| quantity | natural FINAL | held-out code OOD |
|---|---:|---:|
| target CE increase when equality service is removed | `+0.46856` | `+1.50166` |
| simultaneous 95% lower bound | `+0.25900` | `+1.29210` |
| target-minus-matched-negative specificity | `+0.48880` | `+1.28176` |
| extraction recovery `R` | `0.90851` | `1.01041` |
| simultaneous lower bound for `R` | `0.69895` | `0.80086` |
| deranged-null recovery | `-0.00302` | `-0.00090` |

The exact full replay has numerically zero KL divergence from the native model.  The
program therefore reconstructs the owner exactly, restores about 91% of its natural
induction effect, and correctly predicts an even larger effect in code.  This is strong
extraction and OOD-prediction evidence.

### What failed

The preregistered overall verdict is **NO-GO** because collateral damage did not pass.
On natural text, off-target CE increased by only `0.003455` nat, but the simultaneous
upper confidence bound was `0.19516`, too uncertain for the frozen `0.01`-nat
guarantee.  On code, off-target CE increased by `0.13831` nat directly.

The likely interpretation is not “the equality tensor is wrong.”  It is that the
registered repeated-bigram mask is too narrow a name for the service.  Equality
fetching also supports other copying opportunities, particularly program text.  Those
uses are related to the tensor program but were counted as off-target by this assay.

So the honest quality label is:

> Mechanistic Tier 4; exact replay, extraction, and OOD prediction pass; an
> induction-only selective-removal certificate fails.

The next principled move is to factor a broad equality matcher from the different
payload/use branches it serves, or prospectively enumerate all equality-copy
affordances before defining unrelated controls.  We will not weaken or relabel this
completed verdict after seeing it.

The readable circuit dossier is
[`02_induction_copy.md`](../../bilinear_quotient/circuits/campaign_2026_08_30/02_induction_copy.md),
and the sealed numerical result is
[`induction_equality_tensor_final_ood_v2_retry1_result.json`](../induction_equality_tensor_final_ood_v2_retry1_result.json).

## UPDATE: learned low-rank subspaces are enriched, not complete circuits

A separate discovery run learned rank-1 and rank-4 subspaces by gradient descent on
600 rows, then evaluated on 400 held-out rows.  The optimizer-health tests passed for
all 20 fits.  Rank-4 subspaces recovered only `8%` to `23%` of their full component's
target effect, despite occupying just `4/1152` of the residual width.  They were often
much more target-concentrated than random directions, and one newline/layout rank-4
fit reached about `15.4x` target-versus-off-target concentration.

This is useful but not a circuit certificate.  It says behavior is strongly enriched
in small directions inside shared components; it does not say the whole computation
is low rank.  A rank sweep also shows that raising one candidate from rank 4 to rank 64
recovers only about `25%` to `35%` for representative cases.  Consequently we are
prioritizing tensor-factor composition and behavior-specific use branches over plain
SVD truncation.

## UPDATE: the ten-circuit queue

Each candidate has its own file and current tier:

| circuit | current mechanistic tier | present status |
|---|---:|---|
| previous-token/bigram lookup | 5 | exact/extractable; earlier selective-removal assay failed |
| induction/equality copying | 4 | extraction + OOD pass; narrow collateral gate fails |
| ordered successor | 2 | tensor owner built; row-budget protocol needs amendment |
| matched bracket closure | 4 | dense exact adapter built; external audit found lifecycle defects to repair |
| article choice | 3 | rank-16 writer promising; composed terminal test pending |
| newline boundary | 4 | exact five-head algebra; fresh L12H6 canary being built |
| copied entity | 2 | behavior large but overlaps general copying |
| novel capitalization | 2 | diffuse shared late-layer service |
| quote parity/closure | 2 | shares L13H8 with brackets; decoded state is not yet causal |
| numeric/unit formatting | 1 | needs a powered fresh screen before tensor compilation |

The successor row freezer was correctly rejected after `28.06` seconds because the
frozen demand for all nine digit transitions plus powered controls required more than
192 documents.  No model was loaded and no outcome was seen.  We are measuring the
exact minimum and will prospectively raise the default document budget with margin,
rather than dropping rare transitions or changing support after inspection.

The bracket adapter currently remains NO-GO for execution until its independent audit
defects—source closure, race-proof terminal state, lineage replay, and full model/call
rebinding—are repaired.  This is infrastructure correctness, not a negative bracket
result.  Newline is being advanced independently using the shared exact-attention and
statistics machinery.

## Why this direction can expose common upstream structure

The point of ten circuits is not ten disconnected stories.  Every circuit is expressed
as an executable tensor program with explicit owners and shared-service accounting.
Once several pass, we can compare their upstream interfaces:

- equality matching shared by induction and copied entities;
- L13H8 shared by bracket and quote closure;
- layer-0 token/bigram lookup shared by previous-token and article choice;
- late structural writers shared by newline, punctuation, capitalization, and quotes.

Repeated common factors become candidate reusable library functions or DAG parents.
Behavior-specific residual factors become the editable leaves.  This gives a concrete
route from circuit discovery to a smaller whole-model program: store a shared tensor
once, attach sparse use-specific branches, and validate each branch by extraction,
selective removal, and OOD transport.

