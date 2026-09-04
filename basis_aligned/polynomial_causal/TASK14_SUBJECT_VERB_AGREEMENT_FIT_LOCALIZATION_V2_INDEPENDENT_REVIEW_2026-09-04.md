# Independent review: task14 FIT localization v2 authority

**Reviewed:** 2026-09-04 UTC

**Exact candidate:** `8f41f51cdf7e073063201cc48760622607ce91b9`

**Verdict:** **APPROVE** these exact bytes for construction of a separate CPU-only, outcome-blind physical-call
compiler. This does not approve an intervention implementation, model/checkpoint/GPU/activation access, execution,
enqueue, publication, or any SELECT/TEST/OOD or later-phase work.

The v2 authority repairs every blocking issue in the immutable v1 rejection. Its serialized partition and donor
objects are materialized, canonical, reconstructible, and validator-bound. Most importantly, its absolute C alignment,
bidirectional coordinated-plural C ↔ ordinary-singular transfers, and matched coordinated ↔ ordinary-plural controls
make the old local-head-morphology construction logically unable to reach a success terminal. Seed aggregation,
site/pair selection, higher-rank falsification, necessity/redundancy, ordered reset/rescue, and exhaustive terminal
precedence are now fixed without a decision-changing ambiguity.

I inspected exact Git objects and the frozen FIT authority only. I did not inspect or create a localization
implementation, read a model/checkpoint/activation/localization outcome, use a GPU, touch queue/runner state, or open
SELECT, TEST, OOD, or any later phase.

## Exact-object and ancestry closure

The candidate resolves to the full SHA above, with parent
`1bff99a757736320401fd7f7a8bac19629ce124f` and tree
`8da1aac1fb172c33eaecdeaa8b1534a062e334dc`. It changes only the board plus the five reviewed files. The rejected v1
authority (`7986557ece6ee117cd40842fc02c9cf8d21149a5`), its independent rejection
(`52884a4691c3f388c4b0ba0c1327a39f1c0ef411`), and the valid task14 FIT capability package
(`90c5b1606f6eb309ea9fca0042414c9146d8c455`) are strict ancestors.

| Exact candidate object | Git blob | Raw SHA-256 |
|---|---|---|
| builder | `18a27086ca6feacb8843407e67c5db388ae8c705` | `ac6cc964065204193a1c119c721b37dabd9f026ec56b4a4d3b0c0ce837f27d49` |
| partition JSON | `9bec6286b01c78a1ab9a4bc875d6b83890b46f02` | `1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3` |
| donor JSON | `2e7359007a2f7e9ef9a0f139f220f6b9e42a48c4` | `ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a` |
| focused tests | `ae76cbf60e1a31836701891ccf12f1c0d0d30aef` | `bd2623ebe8aafc28a59990c615abd2919591ac9b062cd57ce7ed49fc99374ccf` |
| v2 preregistration | `4d74c90ae7c2911db38690d377f3406fa307ee8d` | `3ea31387f611d0d095895dec6ed0859e1d99b2ad91a5d5adfb7be178bf127f59` |

The builder independently rechecks the frozen FIT file
`e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f`, FIT logical rows
`3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1`, complete logical authority
`1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1`, and generator
`33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94` before deriving either artifact.
Exact candidate blobs and current reviewed worktree bytes agreed.

## Independent reconstruction

I reconstructed both artifacts from the FIT authority rather than trusting their metadata. Canonical serialization is
sorted-key compact ASCII JSON, with exactly one trailing newline for files. Recomputed logical hashes are:

- partition records: `285092178ef25e5aee923a2b02ec791c6b2df83e7c47f185626cd5cfa507d08c`;
- donor records: `6e1fc1fef2715e0c87f0e494646057957bad284f7b69b1e52dcc4ec0f3e6f905`;
- endpoint table: `1b0deab978dbd3126ac09b22818609177b1b1da461eaa1812aa2d05bbb9d8438`; and
- unchanged original-704 core envelope: `25a1f09d5947301f573b223abfbcae1699555ddf809f2b137eabffcbe776f3dc`.

The partition has 32 records: 16 DISCOVERY groups and 16 VALIDATION groups, exactly the printed memberships. Each
half has 64 FIT rows, 128 endpoint prompts, 128 unique prompt strings, 16 rows per A1/A2/P/C family, eight head-noun
pairs, and four mirror-coherent groups in each ordinary subject-number × attractor-number cell. Cross-half overlap is
zero for group IDs, row IDs, prompt strings, and head-noun pairs. Both halves use the same four FIT templates, and
nouns can recur across semantic roles; the preregistration accurately limits the claim to held-out prompts,
head-role/head-pair transfer rather than held-out syntax or globally disjoint vocabulary.

The donor object contains 256 canonical endpoints and exactly 1,088 ordered relations: the unchanged 704-record v1
core plus 384 new Q-only complete-subject relations. Per partition, the new census is 64
`C_to_ordinary_singular`, 32 `ordinary_singular_to_C`, 64 `C_to_ordinary_plural_control`, and 32
`ordinary_plural_to_C_control`. Within each arm, A1/A2 family, base/donor side, and both attractor-number values are
balanced as specified. Every new pair is partition-local, attractor-number matched, non-self, from different groups
and different head pairs. Every C endpoint is target of two answer-changing donors (one A1, one A2) and two
answer-preserving plural donors. No correctness filter or lexical exclusion exists.

The builder reconstructs the complete expected Python objects before requiring structural equality and materialized
byte equality. My additional in-memory attacks—partition flip, partition reorder, Q-only escape, semantic-relation
relabel, replacement of a C→singular donor by a plural morphology donor, and duplicate record—were all rejected.
The checked-in suite additionally covers schema, literal, ID, ordinal, seed, source, authority, symlink, create-only,
and coherent endpoint/donor mutations.

## Shortcut and feasibility audit

### Local morphology, syntax, lexical identity, and length

The v1 counterexample encoded ordinary local head morphology at both H and Q and assigned C the singular value. It
cannot pass v2. At Q, C must lie absolutely on the plural side of the DISCOVERY ordinary midpoint (separate
base/donor medians, sign fractions, and pooled lower quartile), accept donor-directed C→singular and singular→C
effects separately for A1 and A2, and remain inert under the matched C↔ordinary-plural controls. A coordinate that
sets C to singular fails alignment and both answer-changing directions; a construction flag independent of number
fails the matched plural controls or the singular transfers. H is explicitly only local head morphology, excludes C
from fitting/success, and cannot be promoted to a complete-subject claim.

Length is not a number shortcut: within each ordinary family, singular and plural endpoints have identical token
length and aligned H/Q positions (A1/P length 5; A2 length 8), while C length 8 is checked against both A2 and A1 in
the fixed cross-construction arms. Template and lexical memorization cannot select held-out groups or head-role pairs;
all selection and fitting are DISCOVERY-only. Shared syntax and cross-role noun reuse remain real scope limitations,
not hidden independence claims. A role-sensitive rule that computes coordination as plural and transfers causally
across the registered arms is operationally the complete-subject state being tested, rather than the rejected local
morphology surrogate.

### Output direction and generic low-rank reduction

Output effects and coordinate distances are normalized in their own units; neither can inflate the other. A generic
rank-one variance or construction coordinate cannot pass merely by reducing dimension: it must clear cellwise
donor-directed full-state-relative effects, absolute C sign/alignment, four complete-subject causal cells, ordinary
number × attractor cells, same-state output and coordinate controls, syntax cross-fits, necessity, and five-seed
rules on held-out prompts. Rank two/four are matched-opportunity falsifiers and cannot become an alternate success
route; rescuing a failed rank-one arm, alignment, or control fires the higher-rank rejection terminal.

A Q direction may be geometrically aligned with the fixed `are`−`is` readout, especially at a late boundary. The
authority does not claim semantic purity or identify a named component, and FIT alone cannot distinguish every
equivalent basis for the same causal answer state. Such a direction reaches a success terminal only if it also carries
the correct coordinated-subject value, transfers in both directions, passes same-state controls and cell gates, is
necessary, and participates in the frozen H→Q reset/rescue relation. Under the document's operational claim this is
not a false terminal. The document correctly closes component localization and bilinear translation behind later,
separate preregistration.

### Selection, denominators, reset/rescue, and interactions

The 38-site screen uses DISCOVERY-only full-state ceilings and gradient scores. It keeps at most three eligible H
sites and all eligible Q sites; no empty class can be papered over. H, Q, medoid seed, top-two-Q pair, rank, and
calibration are all frozen from DISCOVERY before VALIDATION. Site ties, Q-formation fallback, multi-site medoids,
cell weighting, deterministic initialization, exact minibatch streams, and the universal medoid/median/four-of-five
rule are explicit. Validation never reselects or recalibrates.

Every quotient has a frozen unit, baseline, and failure boundary. Nonfinite values and unhealthy/missing seeds are
instrument-invalid. Finite absent causal ceilings, collapsed coordinates, nonpositive necessity baselines, and
nonpositive reader effects fail their scientific hypotheses without division. Small same-state normalizers become
infinite leakage and fail. These conventions prevent zero or sign-flipped denominators from manufacturing success.

Necessity uses unique validation endpoints and four separate family × state cells. Single-site necessity and the
top-two redundancy route are mutually exclusive: redundancy is considered only after selected-Q necessity fails and
requires both singleton ratios below 0.25 in every cell, joint ratio at least 0.50, interaction at least 0.20, and
direction support. The joint intervention is causally ordered and uses the later site's current activation.

Reader reset is measured against native target and rescue against upstream-neutralized target. Positive finite
upstream/downstream effects are mandatory, reset/rescue sign reversal and overshoot fail, H must strictly precede Q,
and every syntax, direction, and matching cell passes separately. A first rise in Q score cannot substitute for this
test. The nine terminal clauses are mutually exclusive and exhaustive: malformed instrument, absent ceiling,
higher-rank rescue, failed rank-one semantics, sufficiency-only, single/redundant necessity with supported reader,
and the two corresponding unresolved-reader outcomes have a fixed precedence.

## Implementation boundary and logical price

The authority freezes logical science, not physical compute: 38 sites, 1,088 donor relations, five fixed seeds,
ranks 1/2/4, objectives, selection, gates, and terminals. Its stated worst case of 60,000 optimizer steps is correct:
44,000 joint rank-one steps if all 19 Q sites are eligible plus 8,000 family-only cross-fit steps and 8,000 rank-2/4
falsifier steps at selected H/Q. This is deliberately not called a physical forward/backward count. A later compiler
must freeze every call, batch, cache, dtype, retained byte, runtime gate, create-only namespace, and hard time limit,
and must fail before model access if its approved budget is inadequate. No current artifact can be interpreted as
that compiler or as authorization.

## Reproduction record

- Deterministic `--check`: **PASS**, 32 partition records, 1,088 donor records, exact artifact hashes, `model_calls=0`.
- Focused builder plus frozen task14 authority tests: **43 passed in 3.51 s** with model-disabled, CUDA-hidden,
  bytecode/cache-disabled CPU settings.
- Independent canonical rebuild and byte comparison: **PASS**.
- Independent split, endpoint, arm, balance, prompt, token-length/position, and donor-semantics census: **PASS**.
- Six additional coherent in-memory mutations: **6/6 rejected**.
- Local-morphology countermodel: **cannot satisfy v2 Q gates**.
- Decision logic and logical feasibility: **PASS**, with no physical-price inference and no later-phase opening.

The exact candidate is therefore suitable only as prospective input to a separate, independently reviewed CPU
compiler construction.
