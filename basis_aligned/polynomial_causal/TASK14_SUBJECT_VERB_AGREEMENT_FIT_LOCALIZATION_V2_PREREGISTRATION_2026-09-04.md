# Task 14 FIT localization v2: local head-number carrier, complete-subject state, and readers

**Prospective freeze:** 2026-09-04 UTC. **Status:** CPU-only immutable successor candidate. It licenses only an
independent CPU review and, if approved, construction of a separate exact physical-call compiler. It does not license
an intervention implementation, model/checkpoint/GPU/activation access, queue/enqueue, result publication, or opening
SELECT, TEST, or OOD.

## 1. Why v2 exists

The v1 preregistration is preserved unchanged at commit
`7986557ece6ee117cd40842fc02c9cf8d21149a5`, document SHA-256
`6fb4b00080d9bf4b1eaec5953b2806b4a8c2fcc7323a2f938ce7f53192734e6e`. Independent review commit
`52884a4691c3f388c4b0ba0c1327a39f1c0ef411`, review SHA-256
`d4d7ac9b76d54eee73278a2af903c8c34472bcc27917b28c997475d50eab3da2`, **BLOCKED** v1 as compiler authority.

V1 had two central defects. First, its partition and donor digests named byte strings that were not materialized or
fully specified. Second, every affirmative causal example used an ordinary morphologically singular/plural head.
A rank-one “does this local token have an `s` suffix?” coordinate could pass all v1 gates while assigning coordinated
subjects the wrong singular state. Several aggregation, reader, interaction, and terminal choices were also left to a
future compiler even though they could change the scientific verdict.

V2 is a new authority, not an edit or reinterpretation of v1. It:

1. materializes and validates exact partition and donor JSON bytes;
2. preserves the verified 16/16 partition and all original 704 donor relations;
3. adds 384 Q-only relations that test coordinated plural subjects affirmatively against ordinary singular and plural
   subjects in both causal directions;
4. distinguishes a local head-number carrier at H from a complete-subject number state at Q; and
5. freezes every decision-changing aggregation, denominator, reader, redundancy, rank, and terminal rule.

The task-14 capability pass at `90c5b1606f6eb309ea9fca0042414c9146d8c455` is only the phase opener. Capability
values do not select a site, seed, subspace, reader, rank, or row.

The already-recorded opener summary is retained without re-reading model artifacts: all six ordinary A1/A2/P
side-cells were 32/32; C was 28/32 base and 29/32 donor, with the seven failed row-sides concentrated in `key`/`dog`
lexical cases; paired P and C own-answer margin mean-absolute differences were 0.750 and 0.614 nat. These native C
errors and nonzero natural control differences are part of the authority, not exclusion criteria. V2 neither
recomputes the outcome nor postselects rows from it.

## 2. Exact CPU authority closure

The only prompt authority is FIT:

| Object | SHA-256 |
|---|---|
| FIT authority JSON | `e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f` |
| FIT logical rows | `3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1` |
| complete logical authority | `1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1` |
| task generator | `33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94` |

The v2 CPU artifacts are:

| File | Schema | Artifact SHA-256 | Logical-record SHA-256 |
|---|---|---|---|
| `ops/circuit_battery_task14_fit_localization_partition_v2.json` | `task14_fit_localization_partition_v2` | `1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3` | `285092178ef25e5aee923a2b02ec791c6b2df83e7c47f185626cd5cfa507d08c` |
| `ops/circuit_battery_task14_fit_localization_donors_v2.json` | `task14_fit_localization_donors_v2` | `ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a` | `6e1fc1fef2715e0c87f0e494646057957bad284f7b69b1e52dcc4ec0f3e6f905` |

The donor artifact's canonical endpoint table has SHA-256
`1b0deab978dbd3126ac09b22818609177b1b1da461eaa1812aa2d05bbb9d8438`. The exact original 704-relation v1
core envelope remains
`25a1f09d5947301f573b223abfbcae1699555ddf809f2b137eabffcbe776f3dc`.

The deterministic builder is `ops/build_task14_fit_localization_v2.py`, SHA-256
`ac6cc964065204193a1c119c721b37dabd9f026ec56b4a4d3b0c0ce837f27d49`. Its focused tests are
`ops/test_build_task14_fit_localization_v2.py`, SHA-256
`bd2623ebe8aafc28a59990c615abd2919591ac9b062cd57ce7ed49fc99374ccf`.

Canonical JSON is UTF-8 `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)` followed by
one newline for artifact files and no newline for logical-object hashes. Both artifacts bind the exact FIT hashes,
v1 commit/document, and independent BLOCK review commit/document. The validator reconstructs the complete expected
objects and requires Python structural equality plus byte equality for the materialized files. It rejects altered
schemas, literals, fields, order, IDs, hashes, missing or duplicate records, cross-partition donors, changed semantic
relations, symlinks, and source mutation.

### Exact partition record

`partition_order` is `DISCOVERY`, then `VALIDATION`; records are in that partition order and increasing group number.
Each record has exactly `group_id`, `group_number`, `mirror_base_group_number`, `mirror_group_numbers`, `partition`,
and `stratum`. The seed literal is
`task14-fit-localization-v1|discovery-validation-pair-coherent`.

The indivisible units are $\{g,g+16\}$ for $g=0,\ldots,15$. Even and odd $g$ form two number-cycle strata. Within
each stratum, units are ordered by

```text
SHA256(seed_label + "|" + group_id(g) + "|" + group_id(g+16))
```

and the first four units enter DISCOVERY. The exact membership is:

```text
DISCOVERY = [0, 1, 4, 6, 9, 10, 11, 15, 16, 17, 20, 22, 25, 26, 27, 31]
VALIDATION = [2, 3, 5, 7, 8, 12, 13, 14, 18, 19, 21, 23, 24, 28, 29, 30]
```

Pair coherence preserves a complementary-number noun-role unit and keeps each head-noun pair on one side. It is
**not** justified by exact prompt duplication: all 16 mirror units have zero prompt overlap. Each half has 64 rows,
128 unique endpoint prompts, four groups per subject-number $\times$ attractor-number cell, and eight disjoint
head-noun pairs. All four FIT templates occur in both halves, and nouns recur across halves in different semantic
roles. Validation is held-out prompt, group, head-role, and head-pair transfer—not held-out syntax or globally disjoint
vocabulary. Cross-syntax evidence comes from A1-to-A2 and A2-to-A1 transfer inside each half.

### Exact donor record

The donor artifact stores a 256-row endpoint table once. An endpoint has exactly `attractor_plural`, `endpoint_id`,
`family`, `group_id`, `group_number`, `head_pair`, `prompt_sha256`, `row_id`, `side`, and `subject_state`. Endpoint ID
is `row_id:side`, with side literal `base` or `donor`, and subject state is integer $-1$ or $+1$.

Every donor record has exactly:

```text
arm, donor_endpoint_id, expected_relation, family, matching, ordinal,
partition, q_only, record_id, source_contract, target_endpoint_id
```

`record_id` is the SHA-256 of the canonical JSON object containing all those fields except `ordinal` and `record_id`.
Records sort by partition-order index; source-contract-order index (`v1_original_704`, then
`v2_complete_subject_Q`); then lexicographic `arm`, `family`, `target_endpoint_id`, `matching`, and
`donor_endpoint_id`. `ordinal` is the zero-based position after sorting. The materialized JSON is the exhaustive
authority for all literal values and relations; prose cannot add or remove a donor.

## 3. Frozen donor families

### Original 704, unchanged

Per partition, v2 preserves the original 352 relations exactly:

| Relation | Count | Meaning |
|---|---:|---|
| paired A1/A2 | 64 | One-token head-number edit, both causal directions. |
| cross-noun A1/A2 | 128 | Two different-group donors with opposite subject number and matched attractor number. |
| cross-syntax A1/A2 | 64 | A1 target with A2 donor and vice versa. |
| P positive transfer | 64 | Opposite subject-number transfer on P prompts across noun identity. |
| paired P control | 16 | Attractor lexical identity changes; subject number stays fixed. |
| paired C control | 16 | Attractor number changes; coordinated plural subject stays fixed. |

Original records retain their exact v1 `arm`, `family`, `matching`, target, donor, and seed-selected relationships.
All remain partition-local. C paired controls are Q-only; the others apply at the semantic sites stated below.

### New 384 complete-subject Q relations

Every new record has `source_contract="v2_complete_subject_Q"` and `q_only=true`. The seed literal is
`task14-fit-localization-v2|complete-subject-Q-donors`. Candidate donors are partition-local, have matched attractor
number, use a different group and head-noun pair, and are chosen by the lexicographically least

```text
SHA256(seed + "|" + relation_label + "|" + partition + "|" + ordinary_family
       + "|" + target_endpoint_id + "|" + candidate_endpoint_id)
```

The materialized records fix every candidate. Per partition:

| Arm | Count | Registered relation |
|---|---:|---|
| `C_to_ordinary_singular` | 64 | Each of 32 C endpoints receives one A1-singular and one A2-singular donor; answer should move `are` → `is`. |
| `ordinary_singular_to_C` | 32 | Every singular A1/A2 endpoint receives a coordinated-plural donor; answer should move `is` → `are`. |
| `C_to_ordinary_plural_control` | 64 | Each C endpoint receives one A1-plural and one A2-plural donor; both complete subjects are plural. |
| `ordinary_plural_to_C_control` | 32 | Every plural A1/A2 endpoint receives a coordinated-plural donor; both complete subjects are plural. |

Thus the added answer-changing arms are bidirectional, and the answer-preserving arms test the same complete-subject
state across very different surface constructions. Every C endpoint appears as a target in two answer-changing and
two same-state relations. All native C errors and every lexical case remain; no correctness filter exists.

## 4. Two different scientific variables: H and Q

The model residual width is $D=1152$. Boundaries $b=-1,0,\ldots,17$ mean normalized embedding input and residual
after complete blocks 0 through 17. Token coordinates come from the FIT authority.

- **H is a local head-number carrier.** On ordinary A1/A2/P prompts, $H$ is the stored subject-head token position.
  Its binary label is the local head noun's morphological number. H can help locate where this lexical feature enters
  or is transported. It is not called the grammatical number of a complete coordinated subject. C has two singular
  conjunct heads, so C never enters an H fit or H success gate; both C conjunct projections may be reported only as
  descriptive values.
- **Q is the candidate complete-subject state.** $Q$ is the stored final prompt position where the copula is predicted.
  Its label is grammatical subject number: ordinary singular is $-1$, ordinary plural is $+1$, and coordinated C is
  $+1$ although both head tokens are locally singular. Every complete-subject claim, every C relation, and every
  prospective reader gate is Q-only.

The fixed output contrast is

$$
m(x)=\ell_{\texttt{ are}}(x)-\ell_{\texttt{ is}}(x).
$$

For an orthonormal rank-$k$ matrix $U_{b,p}\in\mathbb R^{1152\times k}$, let $P=UU^{\mathsf T}$. A finite natural-donor
interchange changes one semantic residual position:

$$
r'_{b,p}(x\leftarrow d)
=r_{b,p}(x)+P\left(r_{b,p}(d)-r_{b,p}(x)\right).
$$

For opposite complete-subject states, define $\sigma(x,d)=(s(d)-s(x))/2\in\{-1,+1\}$ and

$$
E_P(x,d)=\sigma(x,d)\left[m(x\leftarrow_P d)-m(x)\right].
$$

Positive means movement toward the donor answer in either direction. $E_I$ uses the whole $1152$-dimensional state
at that one position and is the finite intervention ceiling.

For an arm $A$:

$$
\operatorname{Dir}_A(P)=|A|^{-1}\sum_{(x,d)\in A}\mathbf 1[E_P(x,d)>0],
\qquad
\operatorname{Rec}_A(P)=\frac{\operatorname{mean}_AE_P}{\operatorname{mean}_AE_I}.
$$

An answer-changing cell recovery is arithmetically invalid if either mean is nonfinite. A finite cell has an
**available causal ceiling** only when $\operatorname{mean}E_I>10^{-6}$ nat and its full-state direction rate is at
least 0.65. A finite low/negative effect or low direction rate is scientific evidence of no usable ceiling, not
instrument corruption. The 0.65 rule applies to paired, cross-noun, cross-syntax, P-positive, and each of the four
C↔ordinary-singular family/direction cells. Section 7 says exactly how unavailable DISCOVERY ceilings affect site
eligibility; on VALIDATION, an unavailable mandatory ceiling simply fails its scientific gate. Section 7's 0.80
paired-A1/A2 threshold is additional and stricter.

## 5. Absolute coordinated-plural alignment at Q

For each Q site and optimizer seed, a `joint` fit uses only natural DISCOVERY A1/A2 endpoints to define equally
family-weighted class means

$$
\mu_- = \tfrac12(\operatorname{mean}_{A1,s=-1}a+
                         \operatorname{mean}_{A2,s=-1}a),
\quad
\mu_+ = \tfrac12(\operatorname{mean}_{A1,s=+1}a+
                         \operatorname{mean}_{A2,s=+1}a),
$$

For rank one, $a(x)=u^{\mathsf T}r_{b,Q}(x)$. For a rank-$k$ falsifier, define the two ordinary class centroids
$\bar r_-$ and $\bar r_+$ after projection into $U$, and set
$q=UU^{\mathsf T}(\bar r_+-\bar r_-)/\|UU^{\mathsf T}(\bar r_+-\bar r_-)\|_2$ and
$a(x)=q^{\mathsf T}r_{b,Q}(x)$. A nonfinite projected-centroid norm is instrument-invalid. A finite norm at most
$10^{-6}$ is a failed state hypothesis: set $A_C=0$, do not evaluate a quotient by that norm, and fail every absolute
alignment and complete-subject coordinate gate. For rank one a noncollapsed construction fixes the sign of $u=q$.
Orient $q$ so
$\delta=\mu_+-\mu_->0$, and freeze

$$
c=\tfrac12(\mu_++\mu_-),
\qquad
z_C(x)=\frac{a(x)-c}{\delta/2}.
$$

For an `A1_only` or `A2_only` fit, $\mu_-$, $\mu_+$, $q$, $c$, and $\delta$ use only the named ordinary family.
The omitted family does not enter fitting or calibration and is evaluated later with those frozen quantities.

$\delta$ nonfinite is instrument-invalid. A finite $\delta\le10^{-6}$ follows the same registered failed-hypothesis
convention ($A_C=0$, coordinate leakage $=+\infty$, relevant gates fail) and is not mislabeled a runtime failure.
No midpoint, scale, sign, or threshold is recalibrated on
VALIDATION. A perfect ordinary binary code has singular $z=-1$ and plural $z=+1$. On VALIDATION, C base and donor
sides must separately have median $z_C\ge0.50$ and at least 0.80 of endpoints with $z_C>0$; pooled C must have lower
quartile $z_C>0$. Values are reported for every C head pair, second-head pair, attractor pair, side, and native-
correctness stratum, but no lexical stratum is dropped or given its own fitted offset.

This affirmative absolute gate plus bidirectional C-to-singular causal transfer is what rules out a local head-token
morphology coordinate. Paired C invariance alone cannot do so.

## 6. Same-state controls and normalization

For a same-state relation define

$$
L_P(x,d)=|m(x\leftarrow_Pd)-m(x)|,
\qquad
Z_P(x,d)=\|U^{\mathsf T}(r(d)-r(x))\|_2.
$$

Output and coordinate units remain separate.

- P paired and C paired output leakage is normalized by the median absolute projected output effect of paired A1/A2
  opposite-state relations at that site. Coordinate leakage is normalized by the median projected residual distance
  of those same opposite-state relations.
- `C_to_ordinary_plural_control` maps to `C_to_ordinary_singular` with the same donor ordinary family (A1 or A2),
  and `ordinary_plural_to_C_control` maps to `ordinary_singular_to_C` with the same ordinary target family. Each
  control cell's output leakage is normalized by the median absolute projected output effect of that one mapped
  answer-changing cell at Q; directions and families are never pooled for the normalizer. Its coordinate leakage is
  $|a(d)-a(x)|/\delta$.

Any nonfinite normalizer is instrument-invalid. A finite normalizer at most $10^{-6}$ assigns normalized leakage
$+\infty$ (clipped to 1 only inside the training objective) and fails the corresponding control gate. P at H and Q,
and C only at Q, retain
the v1 upper bars: mean normalized output leakage $\le0.20$, median normalized coordinate leakage $\le0.20$, and
90th percentile coordinate leakage $\le0.50$. Each coordinated↔ordinary-plural direction and ordinary family must
have mean normalized output leakage $\le0.25$, median normalized coordinate leakage $\le0.35$, and 90th percentile
coordinate leakage $\le0.75$.

Raw natural P/C logit differences are not required to be zero. Only the effect carried by the proposed coordinate is
controlled.

## 7. Discovery-only site screen

The initial grid has 19 boundaries $\times$ H/Q = 38 sites. At every site, paired DISCOVERY A1/A2 rows receive a
full-state finite ceiling and native gradient score

$$
G_{b,p}=\frac{\|\operatorname{mean}_x[s(x)\nabla_{r_{b,p}}m(x)]\|_2}
{\sqrt{\operatorname{mean}_x\|\nabla_{r_{b,p}}m(x)\|_2^2}}.
$$

The gradient denominator must be finite and greater than $10^{-12}$; otherwise the instrument is invalid. For each
family the natural-margin normalizer $\operatorname{mean}|m(d)-m(x)|$ must be finite and greater than $10^{-6}$ nat.
The screen score is

$$
S_{b,p}=G_{b,p}\min_{f\in\{A1,A2\}}
\operatorname{clip}_{[0,1]}
\frac{\operatorname{mean}_{f}E_I}{\operatorname{mean}_{f}|m(d)-m(x)|}.
$$

An H site is eligible only if every mandatory H answer-changing cell (A1/A2 paired and cross-noun, both cross-syntax
directions, and both P-positive matchings) has an available 0.65 ceiling, and paired A1 and A2 each additionally have
full-state direction at least 0.80. A Q site must satisfy those H conditions plus available ceilings for all four
C↔ordinary-singular family/direction cells. Keep the three eligible H sites with largest raw float64 $S$, ties
resolved by earlier boundary, and every eligible Q site. If either class is empty, the terminal is
`no_intervention_ceiling`. Gradient and screen scores can nominate sites only; they never satisfy a causal gate.

## 8. Rank-one DAS fitting

Rank one is the scientific primary because the proposed state is binary. The following fixes the complete objective,
including coefficients; a compiler may transcribe it but may not choose them.

Partition records into cells by exact `(arm, family, matching, target_subject_state)`. Within a nonempty
answer-changing cell $c$, let

$$
e_c(P)=\operatorname{clip}_{[-1,1]}
\frac{\operatorname{mean}_{(x,d)\in c}E_P(x,d)}
     {\operatorname{mean}_{(x,d)\in c}E_I(x,d)}.
$$

The denominator rules in section 4 apply. For a same-state cell, compute the section-6 normalized output and
coordinate leakage for every record, clip each to $[0,1]$, average each kind within the cell, and let $l_c(P)$ be
their equal mean. An effect or leakage aggregate below is the unweighted mean of its cells, so differing record
counts cannot silently set weight:

- $E_{A1},E_{A2}$ use `answer_change` cells for that family, including paired and both cross-noun matchings;
- $E_{X1},E_{X2}$ use the A1 and A2 `cross_syntax` cells;
- $E_P$ uses `P_positive_transfer` cells;
- $E_{CS}$ uses all four family/direction groups in `C_to_ordinary_singular` and
  `ordinary_singular_to_C`;
- $L_P,L_C,L_{CP}$ use `P_zero_coordinate_control`, `C_zero_coordinate_control`, and both
  coordinated↔ordinary-plural control arms, respectively.

For DISCOVERY C endpoints define an alignment score, with base and donor sides first averaged separately,

$$
A_C=1-\frac14\cdot\frac12\sum_{s\in\{base,donor\}}
       \operatorname{mean}_{x\in C,s}\operatorname{clip}_{[0,4]}\left((z_C(x)-1)^2\right).
$$

Thus $A_C\in[0,1]$, and the ordinary class midpoint and separation are always recomputed from DISCOVERY for the
current fitted subspace; they are never learned offsets. The exact joint objectives are

$$
J_H=\frac{E_{A1}+E_{A2}+\tfrac12E_{X1}+\tfrac12E_{X2}
                 +\tfrac12E_P-\tfrac12L_P}{4},
$$

$$
J_Q=\frac{E_{A1}+E_{A2}+\tfrac12E_{X1}+\tfrac12E_{X2}+\tfrac12E_P
                 +E_{CS}+A_C-\tfrac12L_P-\tfrac12L_C-\tfrac12L_{CP}}{7}.
$$

The divisor is the sum of absolute coefficients. For an `A1_only` or `A2_only` cross-fit, retain only the named
ordinary family's same-syntax `answer_change` records. Cross-syntax records are never training data for a family-only
fit because even a named-family target has an omitted-family donor. At H:

$$
J_H^{(f)}=\frac{E_f+\tfrac12E_P-\tfrac12L_P}{2}.
$$

At Q, let $E_{CS,f}$ contain exactly C-to-ordinary-singular records whose donor family is $f$ and
ordinary-singular-to-C records whose target family is $f$, and use

$$
J_Q^{(f)}=\frac{E_f+\tfrac12E_P+E_{CS,f}+A_C
                       -\tfrac12L_P-\tfrac12L_C-\tfrac12L_{CP}}{5}.
$$

Every cross-syntax record and every record whose ordinary endpoint belongs to the omitted family is evaluation-only.
Any change to a cell definition, clipping rule, normalizer, sign, coefficient, or divisor requires a new prospective
amendment.

Fit `joint` at the retained H sites and every eligible Q site. After DISCOVERY site selection, fit `A1_only` and
`A2_only` at the selected H and Q sites; the omitted syntax is evaluation-only. Use seeds
`14001,14002,14003,14004,14005`, 400 stratified minibatch steps, 32 logical relations per step, Adam at learning rate
0.03 with cosine decay to zero, QR orthonormalization every forward, and maximize the registered $J$ (equivalently
minimize $-J$). Adam has $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, and zero weight decay. At step
$t\in\{0,\ldots,399\}$ the learning rate is
$0.03[1+\cos(\pi t/399)]/2$. Model weights remain frozen.

Initialization is deterministic and library-independent at the logical level. For matrix entry $(d,j)$, form

```text
SHA256("task14-localization-v2-init" + "|" + seed + "|" + rank + "|" + site
       + "|" + objective_name + "|" + d + "|" + j)
```

and use $+1$ when the low bit of the first digest byte is zero and $-1$ otherwise. Apply reduced QR in increasing
column order and flip each column so the corresponding diagonal of $R$ is positive. A future physical compiler must
freeze the arithmetic dtype and prove its initialized projectors match its own deterministic replay before execution.

The minibatch estimator is also fixed. Multiply every absolute objective coefficient by two, producing an integer
slot multiplicity. The H-joint slot cycle in literal order is
`A1,A1,A2,A2,X1,X2,P,L_P`; Q-joint is
`A1,A1,A2,A2,X1,X2,P,CS,CS,A_C,A_C,L_P,L_C,L_CP`. A family-only cycle deletes the omitted family's slots and uses
neither cross-syntax slot: H-family is `f,f,P,L_P`, and Q-family is
`f,f,P,CS_f,CS_f,A_C,A_C,L_P,L_C,L_CP`. Concatenate the applicable cycle infinitely and assign step $t$ its 32
slots at stream offsets $32t$ through $32t+31$. Within an aggregate, cells are visited round-robin in
lexicographic `(arm,family,matching,target_subject_state)` order. Within a cell, records are ordered by ascending

```text
SHA256("task14-localization-v2-fit" + "|" + seed + "|" + rank + "|" + site
       + "|" + objective_name + "|" + record_id)
```

and consumed without replacement, restarting the same fixed order only after exhausting that cell. `A_C` uses C
endpoint IDs in place of record IDs, with literal cells `C_base` then `C_donor` alternating before the same rule.
The exact site literals are `H:-1,H:0,...,H:17` and `Q:-1,Q:0,...,Q:17`; objective literals are `joint`, `A1_only`,
and `A2_only`; rank is the decimal literal `1`, `2`, or `4`. The effect loss uses the fixed full-state,
full-DISCOVERY denominators and cell weights above, not a batch-dependent denominator. Section-6 leakage normalizers
are recomputed from the **current** projector $P$ using their complete DISCOVERY reference cells at every step, never
from only the minibatch; the nonfinite/finite-too-small rules in section 6 apply. Thus every
physical implementation must reproduce the same logical examples at every step; it cannot choose a convenient
sampler.

A seed is unhealthy if
$\|P_{final}-P_{init}\|_F/\sqrt{2k}<0.02$, if the mean of pre-update $J_t$ over steps 350–399 does not strictly exceed
the mean over steps 0–49, if any required aggregate is absent from the complete 400-step schedule, or if a
gradient/value is nonfinite. **All five seeds must be healthy**; otherwise the run is `instrument_invalid`.

### Frozen seed aggregation and medoid

For each seed and site, form a DISCOVERY causal-effect fingerprint from records with
`expected_relation="opposite_subject_toward_donor"` that are applicable to that site (`q_only=false` at H; all such
records at Q), preserving their exact donor-manifest order. Each coordinate is the record's signed projected output
change divided by its exact cell's valid full-state mean effect. Controls never enter the fingerprint because they
have no signed answer-changing ceiling. For seeds $i,j$, site distance is the mean absolute coordinate difference.
The medoid minimizes the sum of its four distances; an exact tie chooses the smaller integer seed.

For a gate involving multiple sites, including two-site necessity and H→Q reset/rescue, seed distance is the equal
mean of the participating site distances, and the medoid is recomputed from that multi-site distance. This prevents
the Q fingerprint's larger record count from silently outweighing H and fixes one seed identity for the composed
intervention. Single-site gates use that site's medoid.

Every scalar validation gate uses the same rule. A lower-bound gate $v\ge t$ passes only when:

1. the medoid seed is finite and $v_{medoid}\ge t$;
2. the median over all five finite seed values is at least $t$; and
3. at least four of five seeds individually reach $t$.

For an upper-bound gate replace every $\ge$ with $\le$. Boolean gates require the medoid and at least four of five
seeds. Quantiles and direction fractions are first computed within each seed, then passed through this rule. No gate
uses a pooled activation across seeds, a best seed, or a seed chosen on VALIDATION.

### Frozen H, Q, and top-two-Q choices

The H site maximizes median DISCOVERY joint objective across the five healthy seeds; an exact float64 tie chooses the
earlier boundary.

For Q boundary $b$, define

$$
T_b=\min\{\operatorname{medianSeeds}\operatorname{Rec}_{A1,paired},
          \operatorname{medianSeeds}\operatorname{Rec}_{A2,paired},
          \operatorname{medianSeeds}\operatorname{Rec}_{C\to A1_s},
          \operatorname{medianSeeds}\operatorname{Rec}_{C\to A2_s},
          \operatorname{medianSeeds}\operatorname{Rec}_{A1_s\to C},
          \operatorname{medianSeeds}\operatorname{Rec}_{A2_s\to C}\}.
$$

Each C term is exactly one direction × ordinary-family recovery aggregate from section 8. The minimum, not a pooled
mean, is taken after each term's seed median, so no failed direction can be hidden by the other three.

Let $T_{max}=\max_bT_b$. If $T_{max}\le10^{-6}$ or is nonfinite, choose the finite argmax (ties earlier), force reader
status unresolved, and the mandatory Q gates still determine scientific failure. Otherwise, the selected Q site is
the later boundary of the earliest consecutive pair whose later
$T_b\ge0.90T_{max}$ and earlier $T_{b-1}<0.50T_{max}$. If none exists, choose the boundary with largest raw float64
$T_b$, exact ties earlier, and force reader status unresolved. Every $T_b$ denominator must be valid. This curve
locates Q-state formation/transport; it is not itself reader evidence.

For two-site redundancy, sort eligible Q sites by descending raw float64 $T_b$, exact ties earlier, and freeze the
first two distinct boundaries. This is the only `top_two_Q` definition. If fewer than two Q sites are eligible, the
redundancy route is unavailable rather than fabricated.

Ranks two and four are fitted only at the selected H and Q sites with the same five seeds, data, steps, schedule,
optimizer, and evaluation. They are matched-opportunity falsifiers. They cannot become a successful alternative.

## 9. Fixed validation gates

All gates use the universal medoid/median/four-of-five rule.

### H local-carrier gates

1. Every A1/A2 paired and two cross-noun matching has direction $\ge0.80$ and recovery $\ge0.50$.
2. A1→A2 and A2→A1 cross-syntax arms have direction $\ge0.75$ and recovery $\ge0.40$.
3. Both P positive-transfer matchings have direction $\ge0.75$ and recovery $\ge0.40$.
4. P paired leakage passes the registered upper bars.
5. `A1_only` on A2 and `A2_only` on A1 each has direction $\ge0.70$ and recovery $\ge0.35$.

Passing these means H carries transferable local head number. It is never called complete-subject number.

### Q complete-subject gates

1. All H ordinary answer-changing/transfer gates also pass at Q.
2. `C_to_ordinary_singular` separately for A1 donors and A2 donors has direction $\ge0.70$ and recovery $\ge0.35$.
3. `ordinary_singular_to_C` separately for A1 targets and A2 targets has direction $\ge0.70$ and recovery $\ge0.35$.
4. The full-state ceiling for every C↔singular direction/family has positive mean effect and direction $\ge0.65$.
5. C absolute alignment passes every base, donor, pooled, and quartile bar in section 5.
6. P paired, C paired, C→ordinary-plural, and ordinary-plural→C controls pass their section-6 upper bars.
7. Every ordinary subject-number $\times$ attractor-number cell has direction $\ge0.70$.
8. Every C row and endpoint is reported; no native error or lexical group may be removed.

### Rank-one falsifier gate

The binary account fails if rank two or rank four at either selected site improves median VALIDATION recovery by more
than 0.10 on A1, A2, or either C↔singular direction; changes a failed affirmative rank-one arm into a pass; or is
required for C alignment or any control to pass. A higher rank cannot be relabeled as the discovered circuit.

## 10. Necessity and two-site redundancy

For a rank-one direction $u$, DISCOVERY defines the ordinary class midpoint

$$
a_0=\tfrac12(\mu_++\mu_-),
\qquad
r^{neutral}(x)=r(x)-u(u^{\mathsf T}r(x)-a_0).
$$

Let $X_{f,s}$ be the **unique** VALIDATION natural endpoints (not one copy per donor record) for ordinary family
$f\in\{A1,A2\}$ and subject state $s\in\{-1,+1\}$. For every family × subject-direction cell define

$$
B_{f,s}=\operatorname{mean}_{x\in X_{f,s}} s\,m(x),
\quad
D_{i,f,s}=\operatorname{mean}_{x\in X_{f,s}}s[m(x)-m(x^{neutral\ at\ i})],
\quad d_{i,f,s}=D_{i,f,s}/B_{f,s}.
$$

Every $B_{f,s}$ and required numerator must be finite. A finite $B_{f,s}\le10^{-6}$ makes that necessity cell fail
without evaluating its quotient; it is not instrument-invalid. Single-site Q necessity is supported only when, in
**each** of the four family × state cells, $B_{f,s}>10^{-6}$, $d_{Q,f,s}\ge0.25$, $D_{Q,f,s}>0$, and at least
0.65 of endpoints have $s[m(x)-m(x^{neutral\ at\ Q})]>0$. Each scalar and fraction separately uses the universal
seed rule; cells are never pooled before thresholding.

If selected-Q single necessity fails, and only then, the frozen `top_two_Q={i,j}` pair may supply a redundancy route.
Using the same endpoint cells and denominators, compute $d_{i,f,s},d_{j,f,s},d_{ij,f,s}$ and

$$
\iota_{ij,f,s}=d_{ij,f,s}-d_{i,f,s}-d_{j,f,s}.
$$

Two-site redundancy is supported only if, in each of the four cells, both singleton ratios are strictly below 0.25,
$d_{ij,f,s}\ge0.50$, $\iota_{ij,f,s}\ge0.20$, and at least 0.65 of endpoints have the correct signed joint necessity
effect. All terms use the universal seed rule and the pair frozen on DISCOVERY. If either singleton reaches 0.25 in
any cell, this is not the registered redundancy pattern; if selected-Q necessity also failed, causal necessity
remains unsupported.

The joint intervention is executed once in ascending boundary order. At the earlier boundary, neutralize its current
activation. Continue the intervened forward pass to the later boundary and neutralize that boundary's **current**
activation, not a cached native activation. This is the only $d_{ij}$ intervention; reversing order or independently
patching two cached states is not equivalent and is not licensed.

This makes the routes coherent and mutually exclusive: selected-Q single necessity passes, or it fails and the
strict weak-singleton/strong-pair interaction may pass, or neither passes.

## 11. Ordered H→Q reader test

The selected H boundary must strictly precede the selected Q boundary. Otherwise reader status is unresolved and no
alternate H is selected after VALIDATION.

For H neutralization, define $a_H(x)=u_H^{\mathsf T}r_{H}(x)$ and compute the equally A1/A2-weighted DISCOVERY
ordinary singular/plural means exactly as in section 5, but at selected H with the joint rank-one direction. Freeze
$a_{0,H}$ to their midpoint. Then
$r_H^{neutral}(x)=r_H(x)-u_H(u_H^{\mathsf T}r_H(x)-a_{0,H})$. A nonfinite mean is instrument-invalid; a finite class
separation at most $10^{-6}$ makes every reader cell fail without evaluating a rescue ratio.

For each opposite-state record let $m_0$ be native-target margin, $m_i$ upstream-H patch margin, $m_{ir}$ margin after
the same upstream patch followed by resetting the selected Q coordinate to the **native target Q coefficient**, $m_n$
the upstream-H-neutralized margin, $m_{nj}$ the margin after that neutralization followed by inserting the **natural
donor Q coefficient**, and $m_j$ the downstream-Q-only donor patch margin. Aggregate signed effects within each arm:

$$
e_i=\operatorname{mean}\sigma(m_i-m_0),
\qquad
e_{reset}=\operatorname{mean}\sigma(m_{ir}-m_0),
$$

$$
e_j=\operatorname{mean}\sigma(m_j-m_0),
\qquad
e_{rescue}=\operatorname{mean}\sigma(m_{nj}-m_n).
$$

The reset baseline is native target, and the rescue baseline is upstream-neutralized target. Nonfinite $e_i$ or
$e_j$ is instrument-invalid. A finite value at most $10^{-6}$ makes that reader cell fail without evaluating its
ratio. When both are greater than $10^{-6}$ nat, define

$$
M_{i\to j}=\frac{e_i-e_{reset}}{e_i},
\qquad
R_{i\to j}=\frac{e_{rescue}}{e_j}.
$$

Reset passes only if $0\le e_{reset}\le e_i$ and $M_{i\to j}\ge0.70$. Rescue passes only if
$0\le e_{rescue}\le1.25e_j$ and $R_{i\to j}\ge0.70$. Sign reversal or overshoot fails the reader arm; it cannot make
a ratio pass. A1 and A2, both causal directions,
and all three same-syntax donor matchings pass separately under the universal seed rule. There is no additional
"composed P/C leakage" statistic: P paired, C paired, and coordinated↔ordinary-plural relations are same-state and
therefore do not define the opposite-state H→Q reset/rescue ratios above. Their exact single-Q-site leakage gates in
section 9 remain mandatory prerequisites for any reader terminal. This finite reset/rescue evidence, not the first
rise of $T_b$, identifies an ordered reader handoff.

## 12. Exact exhaustive terminal precedence

Evaluate terminals in this order and stop at the first matching clause:

1. **`instrument_invalid`** — any hash/source/schema/order/completeness/runtime/nonfinite-value/optimizer-health
   failure; any missing required source row, seed result, or evaluation of a selected site; an invalid gradient or
   natural-margin screen denominator under section 7; or a physical compiler not yet independently approved. A
   finite causal null, collapsed learned coordinate, or failed reader denominator follows its explicit scientific
   failure rule above and is never promoted to this terminal. Having no eligible H/Q site is handled by clause 2.
2. **`no_intervention_ceiling`** — the instrument is valid, but no H or no Q site passes the full-state eligibility
   screen.
3. **`fit_binary_state_rejected_higher_rank_needed_or_better`** — a rank-two/four falsifier rescues any failed
   affirmative/control/alignment arm or exceeds rank one by the registered 0.10 recovery amount.
4. **`fit_rank1_complete_subject_state_not_identified`** — rank falsifiers do not fire, but any mandatory H carrier or
   Q complete-subject sufficiency, transfer, absolute-alignment, invariance, seed-stability, or cell gate fails.
5. **`fit_rank1_state_sufficiency_only`** — every rank-one semantic/control gate passes, but neither selected-Q
   single-site necessity nor the mutually exclusive two-site redundancy route passes.
6. **`fit_rank1_state_and_ordered_reader_supported`** — semantic/control gates pass, selected-Q single necessity
   passes, and ordered H→Q reset/rescue passes.
7. **`fit_rank1_redundant_state_and_ordered_reader_supported`** — semantic/control gates pass, single necessity fails,
   the registered two-site redundancy route passes, and ordered H→Q reset/rescue passes.
8. **`fit_rank1_state_supported_reader_unresolved`** — semantic/control gates and selected-Q single necessity pass,
   but ordered reader reset/rescue does not.
9. **`fit_rank1_two_site_redundant_state_reader_unresolved`** — semantic/control gates pass, single necessity fails,
   two-site redundancy passes, and ordered reader reset/rescue does not.

The clauses are mutually exclusive and exhaustive after valid execution because single necessity and registered
redundancy cannot both pass. Only terminals 6 or 7 may motivate a separately preregistered SELECT localization study;
8 or 9 permit only a new FIT reader design, and 5 permits only a new FIT necessity/interaction design. No terminal
opens TEST, OOD, weight translation, or automatic retry.

## 13. Opposing predictions and interpretation

| Hypothesis | Must happen | What defeats it |
|---|---|---|
| H carries local noun number | A1/A2/P transfer across noun and syntax at H. | Only paired or one syntax works. |
| Q carries complete grammatical subject number | Ordinary and coordinated plural align above the frozen midpoint; C↔singular patches work in both directions; plural↔plural patches are inert. | C lies on the singular side, only ordinary morphology transfers, or C causal direction fails. |
| One binary direction is sufficient | Rank one passes semantic, necessity, and control gates; ranks 2/4 do not rescue or materially improve. | Higher rank is needed or better by the fixed amount. |
| An ordered H→Q reader handoff exists | H patch is removed by target-Q reset and H neutralization is rescued by donor-Q insertion with valid finite denominators. | Q rise alone, temporal misordering, reset failure, rescue failure, sign reversal, or overshoot. |
| Two Q routes are redundant | Weak top-two singleton necessity, strong joint necessity, and positive interaction repeat on validation. | One singleton is sufficient, joint effect is weak/additive, or pair was selected on validation. |

Gradient magnitude, probe accuracy, basis overlap, component rank, and native head/MLP identity remain screens or
descriptions. They cannot satisfy an identification terminal.

## 14. Later bilinear translation, still closed

After a future successful FIT result and separate authorization, an identified Q state may be translated into a
bilinear reader. For a normalized MLP input $z$ and output read direction $v$,

$$
F(z)=W_D[(W_Lz)\odot(W_Rz)],
$$

$$
v^{\mathsf T}F(z)=z^{\mathsf T}Q_vz,
\qquad
Q_v=\tfrac12\left[
W_L^{\mathsf T}\operatorname{diag}(W_D^{\mathsf T}v)W_R+
W_R^{\mathsf T}\operatorname{diag}(W_D^{\mathsf T}v)W_L
\right].
$$

For the validated normalized-input projector $P_q=qq^{\mathsf T}$, the exact subject-only,
subject-by-context, and context-only terms are $P_qQ_vP_q$,
$P_qQ_v(I-P_q)+(I-P_q)Q_vP_q$, and $(I-P_q)Q_v(I-P_q)$. RMSNorm requires revalidating the state at the actual
normalized input; a pre-RMS residual direction cannot be inserted into this formula silently. This is an exact weight
identity, not a reason to accept a subspace that failed the causal gates.

Nothing in v2 licenses this translation. It requires its own prospective contract, physical price, interventions,
and independent review after causal identification.

## 15. Physical execution remains unfrozen

This authority freezes 38 logical sites, 1,088 donor relations, five seeds, ranks 1/2/4, optimizer schedule, metrics,
bars, selections, and terminals. It intentionally does **not** freeze a model call plan. A separate CPU compiler must
enumerate and hash every native, full-state, gradient, training, necessity, rank-control, two-site, reset/rescue, and
validation call; exact batches and order; forward/backward/update counts; retained arrays, shapes, dtypes, and raw
bytes; cache lifecycle; runtime/canary/checkpoint checks; create-only namespaces; and hard GPU-time maximum. It must
prove FIT-only closure and absence of SELECT/TEST/OOD bytes.

The worst-case frozen logical training schedule is explicit: if all 19 Q sites are eligible, joint rank-one fitting is
$(19+3)\times5\times400=44{,}000$ optimizer steps; the selected H/Q A1-only and A2-only cross-fits add
$2\times2\times5\times400=8{,}000$; and rank-two/four joint falsifiers at selected H/Q add another
$2\times2\times5\times400=8{,}000$, for at most 60,000 optimizer steps before screen and validation interventions.
This is a logical science ceiling, **not** a physical forward/backward count: one step can require several target and
donor evaluations. The later compiler must price the exact implementation and hard-abort before any model access if
the approved budget cannot cover it. Partial Q trajectories, fewer seeds, shortened fits, or interpreting an early
stopping result are not licensed substitutes.

No producer, managed adapter, authorization amendment, or enqueue may be built from v2 until a fresh different-agent
review approves the exact committed builder, tests, artifacts, and this document.
