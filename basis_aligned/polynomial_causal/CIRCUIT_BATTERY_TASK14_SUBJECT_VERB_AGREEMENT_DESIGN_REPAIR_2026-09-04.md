# Task 14 subject–verb agreement: authority design repair

**Date:** 2026-09-04 UTC. **Status:** CPU-only repaired design and semantic generator candidate. This is not an execution
preregistration or authorization. It creates no model calls, outcomes, split artifacts, localization objects, queue
records, or task-21 changes.

## Repair record after rejected candidate `31b812b`

Commit `31b812b751fb5a39b7c7933294ca18213cb52b9f` must not be frozen. Independent review found two real design
defects: FIT P and TEST A1 used the same literal `The {head} behind the {attractor}` surface under different template
IDs, and the second half of each in-distribution C panel repeated the first half's endpoint pair in the reverse
direction. The latter left 32 row records but only 16 independent unordered endpoint pairs and made the base and
donor prompt sets identical.

This repaired successor changes TEST A1 to the natural and phase-unique
`The {head} in front of the {attractor}` surface. It assigns the coordinated second-head role with the same balanced
cyclic map plus an offset of three in groups 16–31. Every noun still occurs exactly twice in that role, but all four
phases now have 32 distinct C base prompts, 32 distinct C donor prompts, no base/donor prompt overlap, and 32 distinct
unordered intervention pairs. The validator checks literal template strings as well as template IDs and checks these
C endpoint-set properties directly. Adversarial tests reconstruct both rejected designs and require rejection. This
is an explicit repair with new hashes below, not a silent amendment of `31b812b`.

## Decision

The task is exact enough to implement. The CPU generator is
`basis_aligned/bilinear_quotient/ops/circuit_battery_task14.py`. Its semantic validator regenerates every row from
the declared split, seed, group number, noun roles, number features, and template. It rejects a row even when an
attacker coherently changes text and token IDs or recomputes the row hash. The default candidate contains 512 rows:
32 linked A1/A2/P/C panels in each of FIT, SELECT, TEST, and OOD.

The behavior ID is `subject_verb.number_agreement`; “task 14” is the stable behavior-bank ordinal, not a new claim
that the old component labels are correct.

## Exact high-level computation

For an ordinary noun phrase, let $n_H\in\{0,1\}$ be the grammatical number of the subject head noun, with 0 singular
and 1 plural. Let $n_A$ denote the number of a nearer noun inside a prepositional phrase or relative clause. The
registered answer is

$$
y = \begin{cases}
\texttt{ is}, & n_H=0,\\
\texttt{ are}, & n_H=1.
\end{cases}
$$

Thus changing $n_H$ must change $y$, while changing $n_A$ must not. For the active coordinated-subject control, two
singular conjuncts joined by “and” form a plural subject, so $y=\texttt{ are}$ even though neither visible head noun
is morphologically plural. The model is scored by the signed two-candidate margin

$$
m(x)=\ell_{y(x)}(x)-\ell_{\bar y(x)}(x),
$$

where $\bar y$ is the other copula. This is forced-choice subject-number agreement, not unrestricted next-token
accuracy and not a claim that “is” or “are” is the uniquely natural continuation.

## Linked A1/A2/P/C panel

Each generated group shares the same five prospectively assigned noun roles and base number state. Every base/donor
pair has equal token length, an unchanged final prediction position, and exactly one changed prompt token.

| Family | Base → donor change | Answer relation | What it distinguishes |
|---|---|---|---|
| A1 | Flip only the subject-head noun number in a prepositional-phrase template | ` is` ↔ ` are` | Head number controls the answer despite a nearer attractor. |
| A2 | Flip only the subject-head noun number in a relative-clause template | ` is` ↔ ` are` | The same abstract change on a structurally different surface. The embedded clause uses number-neutral “I placed/noticed/moved”, so no second agreement cue changes. |
| P | Replace the nearest attractor with a different noun of the same number | unchanged | Attractor lexical identity and surface vocabulary are irrelevant while its grammatical feature stays fixed. |
| C | Under two singular conjuncts joined by “and”, flip only the nearest attractor's number | ` are` on both sides | Attractor number itself is irrelevant, and a “first singular noun → is” shortcut is false. This is an active grammatical control, not a paraphrase or collective-noun convention. |

The two answer-preserving controls deliberately do different work. The post-task21 audit suggested lexical-only P;
the earlier draft considered an attractor-number P. Keeping only either one would leave the other confound. The final
design assigns lexical identity to P and attractor number to coordinated C, preserving one-variable edits and the
four-family contract without adding physical calls.

Examples from default FIT group 0 are:

- A1: `The road near the ship` → ` is`; `The roads near the ship` → ` are`.
- A2: `The road that I placed beside the ship` → ` is`; pluralizing only `road` changes the answer to ` are`.
- P: `The road behind the ship` and `The road behind the star` both → ` is`.
- C: `The road and the table near the ship` and the same prompt with `ships` both → ` are`.

## Split isolation, balance, and OOD

The 64 frozen noun pairs are partitioned into four disjoint 16-pair phase vocabularies. A SHA-256-derived permutation
is used per phase. Across 32 groups, every noun pair appears exactly twice in each of five semantic roles: subject
head, ordinary attractor, second OOD attractor, second coordinated head, and P's replacement attractor. Every
head-number × attractor-number cell occurs eight times. In every A1, A2, and P side, ` is` and ` are` each occur 16
times as answer and 16 times as foil. C has 32 ` are` answers and 32 ` is` foils by its declared coordination rule.

- FIT uses `near`, `that I placed beside`, `behind`, and a coordinated `near` surface.
- SELECT uses disjoint noun vocabulary and `beside`, `that I noticed behind`, `beyond`, and coordinated `behind`.
- TEST uses disjoint noun vocabulary and `in front of`, `that I moved beyond`, `across from`, and coordinated `under`.
- OOD uses the fourth noun vocabulary, two attractors, and longer/fronted structures. Its A1 is
  `Near the A1 beside the A2, the H`, putting the controlling head after both distractors; its A2 retains the head
  before two distractors. This is the prospective test against a fixed “token position 1 is the subject” rule.

Template IDs are disjoint between phases. Prompt strings, noun forms, and group IDs are pairwise disjoint between
phases. Literal template strings are also disjoint between phases; distinct IDs cannot disguise a reused surface.
Within every phase the C base and donor sets each contain 32 unique prompts, are disjoint from each other, and define
32 unique unordered intervention pairs. No phase file has been materialized by this design unit; future phase
artifacts must be separately frozen and opened only in FIT → SELECT → TEST → OOD order.

## Tokenization and intervention coordinates

Every singular/plural noun form and both answers are verified as one GPT-2 token with their actual leading-space
boundary. For each side the generator checks

$$
\operatorname{encode}(x+y)=\operatorname{encode}(x)\mathbin{\|}[\operatorname{id}(y)].
$$

It stores the exact prompt IDs, answer ID, head positions, attractor positions, changed-token position, and final
prediction position. The validator requires equal base/donor prompt length and identical semantic coordinates. A1/A2
must change exactly the head-token coordinate; P and C must change exactly the nearest-attractor coordinate. OOD rows
store both attractor coordinates and prove that the changed one is the nearest to the prediction point.

The default token lengths are 5 or 8 in FIT/SELECT, 5/6/8 in TEST, and 8/9/11 in OOD. Length is constant within every
base/donor pair, not pooled across distinct syntax templates. Later activation interchange must use stored semantic
coordinates rather than pretending that raw token index 1 has the same meaning in every family.

## Direct and local shortcuts versus the dependency

| Candidate rule | Expected result on this authority |
|---|---|
| Always predict one copula | 50% on balanced A1/A2/P ordinary cells; C alone cannot validate the task. |
| Follow the nearest noun's number | 50% on the balanced ordinary number cells and explicitly contradicted by C donor pairs. |
| Read noun identity rather than number/syntax | Cannot transport across the four disjoint noun vocabularies; P also changes identity at fixed number. |
| Read token position 1 as the head | Can fit the ordinary in-distribution templates, so it remains an explicit shortcut; fronted OOD A1 moves the true head after both attractors. |
| Read the first morphologically singular noun | Fails coordinated C, whose two singular conjuncts require ` are`. |
| Parse the subject and use its grammatical number | Predicts A1/A2 changes, P/C invariance, and fronted/long OOD rows. |

Capability alone cannot distinguish every implementation of the final rule. In particular, a template recognizer plus
a position-specific number lookup may pass FIT. That is why any later localization must test held-out template
transport and semantic-coordinate interchange before calling a component a syntactic circuit.

## Prospective FIT capability price and hard gate

If a later preregistration freezes these rows, the smallest native capability screen has four base calls followed by
four donor calls, one call per family and side. Every call is a batch of 32 prompts. The literal price is:

- 8 forward calls;
- 256 explicit row-side evaluations;
- 0 backward calls and 0 updates;
- two retained C-contiguous `float32[32]` arrays per call—answer logit and foil logit—for exactly
  $8\times2\times32\times4=2{,}048$ raw numeric bytes;
- no full logits, activations, component names, readers, writers, heads, MLPs, subspaces, or localization bytes.

Proposed opposing capability bars, to be frozen before any model access:

1. exact call/row/token/price coverage and finite arrays must pass;
2. pooled base accuracy and pooled donor accuracy across all four families must each be at least 0.85;
3. every A1/A2/P family × side cell must have accuracy at least 0.85 and positive mean signed margin;
4. each ordinary-family × side incongruent subset must have accuracy at least 0.85 and positive mean margin;
5. both coordinated-C side cells must have accuracy at least 0.75 and positive mean margin; and
6. all 32 donor directions for A1/A2 are scored according to the frozen donor answer, while P/C must retain the base
   answer exactly.

Because 32 rows give 1/32 resolution, a 0.85 cell bar requires at least 28/32 correct and the 16-row incongruent bar
requires at least 14/16. Any instrument, runtime, namespace, or scientific bar failure is a hard abort: every
scientific projection is null, no site is selected, no localization namespace is created, and SELECT/TEST/OOD remain
unopened. There is no threshold relaxation or row repair after seeing model values.

## Prior work reused, corrected, and not duplicated

Reused facts:

- `qk_algoverify_sv_agreement.py` supplied the original 5-head × 4-attractor × 2 × 2 generator and archived
  capability: 80/80 forced-choice correct, including 40/40 incongruent rows, mean signed margin 3.769.
- `qk_svagree_patch.py`, §42, and `redteam_svagree_2026-07-30.md` establish that attention is required by the prompt
  geometry and that L11H3 removes 1.403 of the 3.057 incongruent margin, but single-component removal flips no rows.
  The balanced all-attention-ablation mean of zero is a cancellation tautology and is not reused as evidence.
- `qk_svagree_locus.py` and §53 found a number-sensitive head-position residual present by layer 1, swappable through
  the middle, and no longer effective after layer 11. The same-number identity swap passed. This supports a future
  early-feature → late-reader hypothesis, not a preselected task14 circuit.
- `circuit_agreement.py/results.json` tested the natural-text copula ensemble `{L11H3,L15H5}`: clean is/are accuracy
  was 0.9568 and removal accuracy 0.9517, a drop of only 0.0052 against the registered 0.10 bar. That result rejects
  treating the two-head ensemble as the general agreement circuit.
- The behavior-bank preimplementation audit already required the A2 repair, lexical P, coordinated C, one-token
  copulas, and exclusion of collective nouns.

Genuinely new work in this unit:

- deterministic four-phase linked panels with disjoint noun and template authorities;
- separate lexical-identity P and attractor-number C controls;
- exact balancing of noun roles, head/attractor number cells, answers, and foils;
- OOD fronting and a second attractor to falsify fixed-position and short-template rules;
- semantic token-coordinate records and full regeneration validation; and
- adversarial tests for coherent text/ID mutation, resigned schema/template mutation, effect/coordinate mutation,
  answer mutation, missing panels, duplicate rows, process-randomization drift, literal surface aliases, and
  reverse-duplicated C endpoint pairs.

Repository organization check: `circuits/DOSSIER.md`, `circuits/experiment_index.json`, and
`CIRCUIT_EXPERIMENT_INDEX.md` currently contain no subject–verb-agreement event. The older files above therefore need
an explicit legacy dossier entry before later localization, but no existing registered task14 experiment would be
duplicated by this authority design.

## What this licenses

This unit licenses only independent CPU review of the generator and, if approved, construction of a separate
capability-only FIT preregistration/compiler. It does not license authority-file publication, model/checkpoint access,
GPU work, enqueue, native capability execution, component localization, reuse of L11H3 as a selected site, or any
later split.

## Frozen CPU verification record

The candidate source bytes and the logical authority they generate are:

- generator SHA-256: `33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94`;
- adversarial-test source SHA-256: `254fe3798efd8a4426f30e054fd8e5646a5bd6635df69815f376311ac2023694`;
- full 512-row logical authority SHA-256:
  `1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1`;
- FIT logical rows SHA-256: `3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1`;
- SELECT logical rows SHA-256: `d6d8a7e7cae24ac3e25e3bef11bde4b4b235e950a23c2842978e7fd2a91803b6`;
- TEST logical rows SHA-256: `d62dae278f66ae5a2e77aadf8b841fe9aecf4bf2fa7bb9378b8d59e9f5829b27`;
- OOD logical rows SHA-256: `f2e4a6fc68be3ff8a87efde056780996106b9fb10a532381588d3d47d9da40b6`.

The focused suite passes 16/16 tests, including deterministic regeneration under three Python hash seeds, coherent
adversarial mutations, literal-surface aliases, and the old reverse-pair construction. A broader CPU-only contract
suite passes 61/61 tests. The ordinary experiment gate is not a
validator for this data-only generator: it expects a runnable experiment with three registered prediction keys, so
its rejection of this module is expected and grants no execution status. Independent review of these exact bytes is
still required before any capability preregistration/compiler is constructed.
