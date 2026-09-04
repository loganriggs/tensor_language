# Independent review: task14 FIT localization preregistration

**Reviewed:** 2026-09-04 UTC

**Exact candidate:** `7986557ece6ee117cd40842fc02c9cf8d21149a5`

**Exact preregistration SHA-256:** `6fb4b00080d9bf4b1eaec5953b2806b4a8c2fcc7323a2f938ce7f53192734e6e`

**Verdict:** **BLOCK** this exact preregistration as the scientific authority for compiler construction. The proposed
experiment has several strong design choices, but two fail-closed defects permit either an unreproducible compiler or
a false positive for the central complete-subject-number claim:

1. the frozen partition and donor digests do not have fully specified serialized objects, so an independent compiler
   cannot reproduce or enforce either digest from the document; and
2. every registered causal gate can pass for a rank-one local head-token morphology coordinate which assigns the
   coordinated C subject the wrong number state.

Reset/rescue aggregation and terminal/redundancy rules also need exact clarification before a physical call compiler is
frozen. Because no localization outcomes or activations exist, these are prospective design repairs, not post-hoc bar
changes.

I inspected exact Git objects and the frozen FIT authority only. I did not create or inspect an implementation, access
a model/checkpoint/GPU/activation/localization result, touch a queue/runner, or open SELECT, TEST, or OOD.

## Exact-object and ancestry closure

The candidate resolves exactly to `7986557ece6ee117cd40842fc02c9cf8d21149a5`, with parent
`2cc222ce546347c6944bb8059e653dad2aec44e4` and tree `e1b649f859fbc5e15709be99c9d0865a8e79c4a6`.
Its only changes are the new 574-line preregistration and its board entry. The preregistration is a regular Git blob
(`450697c9885d7d5742eabc9a1dc07530568611bf`, mode `100644`), and its raw SHA-256 matches the requested
`6fb4b000...`. Current worktree bytes match the exact candidate object.

The following are strict ancestors:

| Required ancestor | Exact commit |
|---|---|
| repaired task14 authority | `e9686bc9bbb40f872d8e8320b30fab4f019e524d` |
| authority review | `ea7efad782c088ba91a2ce338a9f740563c4e7c1` |
| capability compiler | `fc586c1158ddeee7df8f4b502deec54189609c4c` |
| compiler review | `10afc5d6005d169879b07e92cb5fcb4e3a65f312` |
| producer build and review chain | `26d45e89797515240eec368bc313728925d5f48a`, `753afa27e05b594acc39b0c1d84d72272c26e640` |
| authorization and final review | `434f11a927669b86525bf6b9bdc050bd64de544b`, `117af1288b42c8928745842154e0248c5fa9da86` |
| valid capability outcome | `90c5b1606f6eb309ea9fca0042414c9146d8c455` |

The document binds the exact FIT artifact `e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f`,
FIT logical digest `3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1`, and complete authority
digest `1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1`. It correctly treats the capability pass
as an opener rather than localization evidence.

## Independent partition reconstruction

I recomputed the SHA ordering over the 16 indivisible `{g,g+16}` units separately for even and odd `g`. The first
four units in each stratum exactly reproduce the printed groups:

```text
DISCOVERY = [0, 1, 4, 6, 9, 10, 11, 15, 16, 17, 20, 22, 25, 26, 27, 31]
VALIDATION = [2, 3, 5, 7, 8, 12, 13, 14, 18, 19, 21, 23, 24, 28, 29, 30]
```

The substantive split census is sound:

| Check | DISCOVERY | VALIDATION |
|---|---:|---:|
| groups | 16 | 16 |
| FIT rows | 64 | 64 |
| endpoint prompts | 128 | 128 |
| unique endpoint prompts | 128 | 128 |
| head-noun pairs | 8 | 8 |
| groups per subject-number × attractor-number cell, on each side | 4 | 4 |

There is zero cross-half group-ID, row-ID, prompt, or head-pair overlap, and every mirror unit stays together. All four
literal FIT templates occur in both halves. Nouns are not globally held out: because of the Latin role schedule,
discovery head nouns recur in validation as other semantic roles (all eight discovery head pairs recur as validation
second-head pairs, with additional head/attractor cross-role intersections). This is disclosed in the document and is
not activation leakage. It means the validation claim is held-out head-role and prompt transfer, not held-out
vocabulary or held-out template transfer.

One rationale sentence is factually wrong for this repaired authority: none of the 16 `{g,g+16}` units has an exact
natural-prompt overlap. Pair coherence is still useful because it prevents the same noun-role pair from crossing the
split, but “splitting them would put exact natural prompts in both discovery and validation” is false and should be
corrected rather than carried into a validator.

### Frozen partition digest is not independently reproducible

The group membership algorithm is deterministic, but the claimed partition SHA
`125b744d311088b3b6a41b144be51bacd81478212c71b5b82d04fef3548612ec` is not. The document never defines the
bytes called the “canonical partition record”: it does not state whether the hashed object is a list or envelope,
whether it contains group numbers, group IDs, mirror-unit records, the seed, or counts, nor its field names and record
order. I tested 117 ordinary canonical encodings of the printed membership; none matched, but the core issue is not
guessing the hidden encoding—multiple encodings all satisfy the prose and necessarily hash differently.

A future compiler cannot independently “match the digest or stop” without receiving the missing record schema or
materialized bytes. This fails the exact-authority boundary.

## Independent donor reconstruction

Using endpoint identity `row_id:side`, the literal candidate filters, and the three stated SHA-ranking strings, I
reconstructed the selected donor relationships independently. The logical selection census does match the prose:

| Arm per partition | Records |
|---|---:|
| paired A1/A2 | 64 |
| cross-noun A1/A2 | 128 |
| cross-syntax A1/A2 | 64 |
| P positive transfer | 64 |
| paired P control | 16 |
| paired C control | 16 |
| **total** | **352** |

Both halves therefore yield 704 relationships. Every A1/A2 same-family candidate pool has exactly seven members,
every cross-syntax pool seven, and every P-positive pool eight. The selected cross-noun, cross-syntax, and P-positive
donors always use a different head pair from their target; there are no duplicate selected donor prompts for one
target. Every answer-changing/P-positive relationship flips subject number and holds attractor number fixed. Every
paired P/C control holds subject number fixed. No donor crosses the discovery/validation boundary.

### Frozen donor digest is not independently reproducible

The selected relationships are reconstructible, but the claimed donor-contract SHA
`25a1f09d5947301f573b223abfbcae1699555ddf809f2b137eabffcbe776f3dc` is not. “Keys sorted, compact JSON”
specifies JSON formatting but not the record fields. The document never enumerates the record schema, exact `arm` and
`matching` literal values, enclosing object, record-ID formula, semantic labels included in each record, or exact
partition/arm/family ordering vocabularies. Two compilers can choose the same 704 target/donor pairs and produce
different valid-looking byte strings and hashes.

This is a blocking reproducibility defect, not a request for an implementation detail. The digest is meant to be an
authority gate before calls exist; its exact preimage therefore must be frozen now.

## Sign, dimensions, and normalization

The main intervention algebra is dimensionally consistent. With declared residual width `D=1152`,
`U in R^(1152×k)`, `P=UU^T`, and the one-position residual delta are well typed. Nineteen residual boundaries times H/Q
give 38 initial sites. The fixed margin `m = logit(are) - logit(is)` and
`(s(d)-s(x))/2` make positive interchange effect mean donor-directed movement in both singular→plural and
plural→singular directions. The necessity sign is likewise correct for both states.

The document correctly keeps output leakage and coordinate leakage in their own units: output changes are normalized
by a median output change, while projected residual norms are normalized by a median projected residual norm. It does
not divide activations by logits. Recovery also has a reasonable fail-closed full-interchange denominator and direction
guard. A successor should explicitly make a zero gradient-score denominator and a zero natural-margin site-score
denominator instrument-invalid rather than relying on downstream nonfiniteness.

C-at-H handling is anatomically honest: a coordinated subject has two head tokens, so C is excluded from a fictitious
single H gate and both conjunct projections are merely descriptive. The scientific consequence below, however, is
that C-at-Q needs an affirmative state test, not only an invariance test.

## Blocking semantic shortcut: head morphology can pass as complete-subject state

For every one of the 192 ordinary A1/A2/P endpoints, the registered subject state is perfectly predicted by whether
the single surface head token is morphologically singular or plural. Cross-noun and cross-syntax transfer do not break
this confound: English singular/plural morphology generalizes across the held-out head nouns and both templates.

The registered C gates only ask whether a paired attractor edit changes the Q coordinate or its output. They never
require the absolute C coordinate to have the registered plural sign, never compare C to the DISCOVERY class midpoint,
and never use a coordinated-plural C endpoint in cross-construction sufficiency. Both H conjunct measurements are
descriptive. Consequently the following non-grammatical rank-one construction passes all registered bars:

1. Let `h(x)=+1` for an ordinary morphologically plural head token and `-1` for a singular one; set `h(C)=-1` because
   both coordinated conjunct heads are singular. Store `a=h` at the selected H and Q sites along a serial path.
2. Let ordinary A1/A2/P output depend causally on `h`, while C's plural output is supplied by an independent conjunction
   route. Opposite-state A1/A2 and P-positive swaps then have direction and recovery 1 across every donor and syntax.
3. P paired controls have `Delta h=0`; C paired controls also have `Delta h=0`. Both output and coordinate leakage are
   therefore zero.
4. Neutralizing `h` gives perfect A1/A2 necessity, every subject×attractor cell passes, both cross-fits pass, all five
   seeds can be stable, and ranks two/four need not improve anything.
5. Serial propagation makes downstream reset remove the upstream effect and downstream patching rescue it, satisfying
   the intended reader logic.

This construction can reach `fit_rank1_state_and_reader_supported` while assigning every C subject `-1` even though
the preregistered complete-subject state is `s(C)=+1`. It is exactly the local-morphology shortcut the coordinated
family was supposed to distinguish. Reporting all C rows and preserving native C errors does not close it.

The minimal semantic repair is a frozen Q-site C state-alignment and causal-transfer requirement. At minimum, with
rank-one sign fixed from DISCOVERY, all C endpoints should be tested against the DISCOVERY class midpoint and normalized
ordinary class separation, separately by partition/side/lexical strata. Stronger and preferable is a frozen
cross-construction arm: coordinated-plural C donors into ordinary singular targets and ordinary singular donors into C
targets at Q, with donor-directed effect and full-interchange-relative recovery bars. The donor schema/digest and price
must be updated prospectively. C paired leakage should remain as the attractor-invariance control; it cannot substitute
for affirmative plural-state alignment.

## Selection, ranks, reader ordering, and interactions

Several protections are sound:

- gradients and site scores nominate sites on DISCOVERY but cannot satisfy a scientific predicate;
- mirror units and all donors are partition-local;
- H and Q sites, medoid seeds, and the interaction pair are intended to be selected from DISCOVERY only;
- VALIDATION is one-shot and cannot choose another site, seed, rank, donor, checkpoint, or threshold;
- A1-only→A2 and A2-only→A1 cross-fits are mandatory;
- ranks two/four run only at rank-one-selected sites with matched opportunity and can only falsify the binary account;
  they are not alternate success routes; and
- upstream patch then downstream reset, and upstream neutralization then downstream rescue, are the correct causal
  temporal order.

The preregistration nevertheless leaves decision-changing details undefined:

1. It does not say whether sufficiency, necessity, leakage, cross-fit, and reader bars are applied to the DISCOVERY
   medoid, independently to all healthy seeds, or to which seed aggregate. Item 8 defines four-of-five direction and
   median recovery, but not the aggregation for the other bars.
2. Rescue's “resulting signed effect” lacks a baseline equation: it could mean change from the native target or change
   from the upstream-neutralized target. Only the latter isolates downstream rescue. Neither rescue nor mediated
   fraction has an explicit positive/finite denominator guard. A near-zero upstream effect or sign-reversing reset can
   make a ratio arbitrarily large and pass `>=0.70` without valid mediation.
3. “Top two DISCOVERY Q sites” does not specify the ranking statistic or tie rule used to freeze the interaction pair.
4. Necessity bar 5 requires removal of at least 0.25 of native oriented margin, while the redundant-pair classification
   requires each singleton to remove less than 0.25. The first success terminal requires “every validation bar” and
   either a reader or redundancy account, so the redundancy route is contradictory if bar 5 applies to those selected
   Q singletons and underspecified if it applies elsewhere.
5. `two_site_redundant_candidate` is called a result in the interaction section but is absent from the terminal list.
   Conversely, the listed `fit_rank1_state_and_reader_supported` terminal can be emitted for a redundancy account with
   no ordered reader, so its name overclaims. Necessity failure without a qualifying redundancy account also lacks an
   unambiguous terminal/precedence rule.

These choices cannot safely be invented by the later call compiler because they affect scientific success rather than
only batching or cost.

## Implementation and compute boundary

The implementation boundary itself is strong and passes review. The document authorizes only CPU review and a future
outcome-blind compiler/preregistration successor. It explicitly withholds intervention implementation, activation
caching, model/checkpoint/GPU access, enqueue, publication, component selection, weight extraction, and all later
phases. It requires the later compiler to freeze every physical call, batch, forward/backward/update, retained array,
dtype, numeric byte, cache policy, GPU-time maximum, runtime gate, and create-only namespace. Candidate commit
`7986557...` contains no implementation or outcome artifact.

That boundary is exactly why the defects should be repaired now. Freezing compute cannot make an ambiguous logical
manifest or incomplete scientific gate retrospective-proof.

## Required repair and re-review

Create a new immutable preregistration successor; do not edit or reinterpret this rejected object. The successor must:

1. materialize canonical partition and donor JSON artifacts, or enumerate their exact envelope/record schemas, literal
   field values, ordering, record-ID formula, and canonical byte rule; freeze both file and logical hashes and require
   validators to reject missing, duplicate, cross-partition, or semantically invalid records;
2. retain the verified split and donor censuses, but correct the false mirror-prompt statement;
3. add affirmative coordinated-subject Q-state alignment and preferably bidirectional C/ordinary causal transfer, with
   fixed donors, strata, aggregations, denominators, and bars, while retaining paired C leakage and all native errors;
4. freeze exactly which seed/aggregate/site each sufficiency, necessity, leakage, rank, reset, rescue, and interaction
   bar uses;
5. define reset/rescue effects relative to explicit baselines, require finite positive denominators, and prevent reset
   overshoot or sign reversal from masquerading as mediation;
6. specify the DISCOVERY-only top-two-Q ranking/ties and reconcile singleton necessity with the redundancy route; and
7. provide mutually exclusive, exhaustive terminal precedence, with distinct labels for a verified ordered reader and
   a two-site redundant state candidate.

After those prospective repairs, a fresh independent CPU review can decide whether an exact physical compiler may be
built. No compiler, producer, adapter, or execution should be derived from SHA `6fb4b000...`.

## Reproduction record

- Exact candidate/doc/worktree hash and ancestor checks: **PASS**.
- Frozen task14 authority tests: **16 passed** in 2.60 s.
- Independent split reconstruction: all printed memberships and censuses **PASS**; zero group/row/prompt/head-pair
  overlap; all four templates shared; zero actual mirror-prompt overlaps.
- Independent donor selector reconstruction: **704/704** logical relationships and every stated semantic constraint
  **PASS**; exact serialized digest **UNVERIFIABLE** because its schema is absent.
- Adversarial morphology construction: satisfies the stated transfer/control/necessity/rank/reader bars while violating
  `s(C)=+1`; central identification gate **FAIL**.

No model-facing or later-phase action occurred during any check.
