# Rung 592 prospective preregistration amendment: executable centered-factor contract

**Frozen:** 2026-09-04 UTC, before any R592 implementation, model call, or outcome

**Status:** prospective repair of the blocked R592 preregistration; no execution authority until independent review

## Authority and unchanged science

This amendment resolves the four implementation-determining blockers in the independent review of the first R592
preregistration. It does not edit or silently reinterpret that frozen document. These exact bytes are authorities:

- blocked R592 preregistration: SHA-256
  `870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a`;
- independent blocking review: SHA-256
  `9b76b91995374697b8a828ce042e59d81bfddcbaa5f6e843cb0f32f6b01e57f7`;
- review's five-test packet: SHA-256
  `7356aebd017ba6c6c5ce92176ff95fbffd01d5924b5b7d4cc91dd90e0618b07c`;
- R585 replacement amendment: SHA-256
  `98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf`;
- R585 manifest: SHA-256
  `7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962`;
- handoff version 7: SHA-256
  `595b43156117e0ba2e568972f76af81ac4e716ed5537861ac48f13b23d4ed9fd`.

The following remain unchanged: model/checkpoint, R578 rows, both registered directions, four sites, two semantic
roles, target and control families, 20 target plus 32 control cells per split, 124 bootstrap cells per split, 2,000
group-bootstrap replicates, all scientific numerical gates, FIT-first opening, FIT-frozen scales, $10^{-5}$ operational
exactness boundary, FINAL/OOD closure, and the claim boundary of a partial output-space factor only.

R592 has 1,872 FIT rows, 1,728 FIT endpoints, and 3,744 FIT directions; SELECT has 936 rows, 864 endpoints, and 1,872
directions. Endpoint $\times$ site $\times$ role counts are 13,824 FIT and 6,912 SELECT.

## 1. Exact instrument-gate supersession

“Scientific gates unchanged” means the target transfer, opposing-arm, bilinear interaction, active-control,
selectivity, bootstrap, denominator, and FIT-to-SELECT gates. It does not inherit numerical comparisons that R591
proved invalid for the old implementation.

| R585 check or predicate | R592 status | Exact R592 predicate and consequence |
|---|---|---|
| complete native attention reconstruction | retained hard gate | `native_full_write_reconstruction_max_abs <= 1e-5` at every endpoint/site; otherwise runtime invalid diagnostic |
| exact equality support, role/site census, semantic positions | retained hard gate | exact equality and count predicates; otherwise preflight abort if model-free, runtime invalid diagnostic if observed |
| every saved scalar/vector finite | retained hard gate | `all_observed_values_finite`; otherwise runtime invalid diagnostic |
| actual hook change equals planned change | retained and redefined | `actual_centered_hook_delta_max_abs <= 1e-5` for every direction/machine-arm/site |
| structural full-vocabulary output identities | retained hard gate | reconstructed from raw replay-relative logit differences at `1e-5` |
| factorized $B(E_x,U_x)$ equals differently contracted native equality write $C_x$ | **not** an R592 validity gate | save `factorized_vs_native_equality_max_abs`; a value above `1e-5` blocks only `literal_remove_insert`, which R592 does not claim |
| canonical equality term plus canonical non-equality remainder equals native head write | retained hard gate | `native_equality_plus_remainder_max_abs <= 1e-5`; this checks native decomposition without injecting $B-C$ |
| padded versus unpadded native logits | deleted | no natural-length or length-sorted comparator is permitted |
| native versus replay across different endpoint batches | deleted | replaced by directed native versus literal-zero replay on the identical token tensor |
| padding validity | replaced | every capture and directed tensor has physical width 30; membership/order/batch size/query positions and the complete int64 token tensor are hash-bound per call |
| cached self product only | strengthened | componentwise factor transport plus all operational hybrid checks below |

The forbidden old predicate names `canonical_term_failure`, `factor_mismatch`, and `padding_failure` must not appear in
an R592 terminal. The new numerical invalid predicate IDs are exactly:

```text
native_full_write_reconstruction_failed
native_equality_remainder_reconstruction_failed
centered_hook_delta_failed
structural_output_identity_failed
nonfinite_observation
fixed_width_token_manifest_failed
directed_native_zero_replay_failed
factor_transport_failed
```

All use the same $10^{-5}$ maximum-absolute boundary where numerical equality is required. No tolerance is loosened.

## 2. Frozen machine arm IDs and operational labels

The R585 manifest's machine identifiers, control maps, bootstrap cell IDs, namespace, and draw bytes remain exactly
unchanged. R592 machine IDs are:

```text
replay, score, payload, joint
```

Every result, evidence join, scale lookup, structural identity, failure clause, and bootstrap identity uses only those
machine IDs. The mandatory human-readable mapping is:

```text
replay  -> literal_zero_centered_replay
score   -> registered_equality_factor_coefficient_swap
payload -> registered_projected_content_swap
joint   -> registered_joint_output_factor_swap
```

The result metadata contains this exact mapping as `operational_arm_labels`. The strings `coefficient` and
`projected_content` may appear in explanatory prose but never as machine keys. Preflight reconstructs all 248 inherited
bootstrap IDs and draw hashes from the legacy IDs. Any mixed or renamed machine namespace aborts before model load.

## 3. Frozen endpoint mediator and capture-to-directed transport

All scientific factors are captured once from native endpoint execution at physical width 30 before any intervention.
They are frozen endpoint mediators. For endpoint $x$, site $h$, and role $r$:

$$
e^x_{h,r}\in\mathbb R,
\qquad
u^x_{h,r}\in\mathbb R^{1152},
\qquad
B_h(E_x,U_x)=\sum_r e^x_{h,r}u^x_{h,r}.
$$

For every registered direction $x\leftarrow y$, all four cached operational terms are constructed before any arm call:

$$
B_h(E_x,U_x),\quad B_h(E_y,U_x),\quad B_h(E_x,U_y),\quad B_h(E_y,U_y).
$$

The planned frozen centered deltas are

$$
\begin{aligned}
d^{\tt replay}_h&=0\quad\text{constructed by `zeros_like`},\\
d^{\tt score}_h&=B_h(E_y,U_x)-B_h(E_x,U_x),\\
d^{\tt payload}_h&=B_h(E_x,U_y)-B_h(E_x,U_x),\\
d^{\tt joint}_h&=B_h(E_y,U_y)-B_h(E_x,U_x).
\end{aligned}
$$

Every factor in these formulas is from the frozen native endpoint cache. At each site the selected delta is added to
the head output computed in that arm's live forward. In particular, at L7 and L8 the live head output may already react
to an earlier L5/L7 delta; R592 leaves that endogenous reaction intact and adds the fixed registered delta. It never
subtracts a live equality term and never calls any field `live_removed`. L8H3 and L8H4 use one shared pre-modification
layer-8 state and receive their deltas in one transaction.

The directed native call re-observes the recipient's $e$ and $u$ before modification. For every occurrence—not one
representative endpoint—it must satisfy all of:

$$
\max_{h,r}|e^{x,\mathrm{live}}_{h,r}-e^{x,\mathrm{cache}}_{h,r}|\le10^{-5},
$$

$$
\max_{h,r,j}|u^{x,\mathrm{live}}_{h,r,j}-u^{x,\mathrm{cache}}_{h,r,j}|\le10^{-5},
$$

and, in common residual-stream units,

$$
\max_{h,j}|B_h(E_a,U_b)_{\mathrm{live/cache}}-B_h(E_a,U_b)_{\mathrm{cache/cache}}|\le10^{-5}
$$

for $(a,b)\in\{(x,x),(y,x),(x,y),(y,y)\}$, using the live recipient factor wherever $a=x$ or $b=x$ and the frozen donor
factor otherwise. The all-donor term is cache-only in this direction; the registered reverse direction makes $y$ a
recipient and supplies its componentwise live check. Nonfinite or above-bound transport is
`factor_transport_failed`.

Activity is defined from the actual centered deltas, not from an inserted-minus-removed quantity. For group $g$ and
machine arm $a$,

$$
A_g^{(a)}=\operatorname{median}_{h\in H}\left\|d^{a}_{h,g}\right\|_2.
$$

The FIT target scales and active-control rule use this $A_g$. The margin and vocabulary scales and every scientific
threshold remain as in R585. Saved actual hook changes must equal the planned $d^a_h$ within $10^{-5}$.

## 4. Exact calls, evidence, and terminal closure

### Call order and tensor manifests

Every complete split uses this deterministic order:

1. endpoint-capture chunks in manifest order;
2. directed chunks in manifest order; and
3. within each directed chunk: `native`, `replay`, `score`, `payload`, `joint`.

FIT has 54 endpoint tensors and 117 directed tensors. SELECT has 27 endpoint tensors and 59 directed tensors, the last
of which has shape $[16,30]$; every other tensor has shape $[32,30]$. Each unique int64 tensor is stored, with row IDs,
query positions, shape, byte length, and SHA-256. Every call references that tensor record and rehashes its bytes before
execution. The five calls for one directed chunk must reference the same tensor record. No fill, duplicate, reorder, or
natural-length call is permitted.

The complete prices remain:

$$
\mathrm{FIT}=54+5(117)=639,
\qquad
\mathrm{SELECT}=27+5(59)=322,
\qquad
\mathrm{maximum}=961.
$$

There are zero backward passes and zero weight updates.

### Phase-local evidence schema

All numeric arrays are little-endian, C-contiguous `.npy` files. Float arrays are float32 unless explicitly stated;
aggregate arithmetic is float64. Every path, dtype, shape, byte length, row-order hash, and SHA-256 is in the receipt.
For an opened complete split with $N_e$ endpoints and $N_d$ directions, save:

1. `authority.jsonl`: the inherited R585 row fields and both directions; exactly 1,872/936 rows.
2. `endpoint_tokens.npy`: int64 shape $[N_e,30]$ and `endpoint_records.jsonl`; exactly 1,728/864 endpoints.
3. `factor_e.npy`: shape $[N_e,4,2]$; `factor_u.npy`: $[N_e,4,2,1152]$; `support.npy`: bool
   $[N_e,4,2]$; native equality term, factorized equality term, native non-equality remainder, native head write, and
   independently reconstructed full native write arrays, each $[N_e,4,1152]$ where applicable.
4. `directed_tokens.npy`: int64 shape $[N_d,30]$, plus exact 117/59 chunk boundaries and tensor hashes.
5. `directed_records.jsonl`: exactly 3,744/1,872 records with authority IDs, recipient/donor references, target tokens,
   per-condition target logits, other-target logits, log-normalizers, CEs, $n_g,d_g,q_g,c$, transport maxima, activity,
   and array indices.
6. `directed_live_e.npy`: $[N_d,4,2]$ and `directed_live_u.npy`: $[N_d,4,2,1152]$, captured only by the directed native
   observer. These raw arrays independently reconstruct every factor-transport check.
7. `hook_deltas.npy`: $[N_d,4,4,1152]$, axes `(direction, machine_arm, site, residual)`, with machine-arm order
   `(replay, score, payload, joint)`. This is exactly 59,904 FIT and 29,952 SELECT arm/site rows. Replay rows must be
   bitwise zero. Planned deltas are reconstructed from endpoint factors rather than trusted.
8. `logit_differences.npy`: $[N_d,4,50257]$, with difference-axis order
   `(native_minus_replay, score_minus_replay, payload_minus_replay, joint_minus_replay)`. These raw differences
   independently reconstruct the elementwise native/replay gate, every full-vocabulary structural identity, and every
   vocabulary RMS. Its raw byte price is 3,010,595,328 FIT and 1,505,297,664 SELECT.
9. The inherited cell/bootstrap table: exact members, 124 cells, ordered groups, 2,000-draw hashes/statistic hashes,
   point statistics, intervals, FIT-frozen scales, predicates, and failed clauses.

The raw hook arrays cost 276,037,632 FIT and 138,018,816 SELECT bytes. The raw directed live projected-content arrays
cost 138,018,816 FIT and 69,009,408 SELECT bytes. Together with the logit-difference arrays, the principal maximum raw
audit payload is 5,136,977,664 bytes, before smaller endpoint and JSONL tables. This is experimental evidence storage,
not a proposed model or compression price. Lossless file compression is allowed only after the canonical uncompressed
hash, dtype, shape, and byte length are recorded.

### Terminal-specific closure

The mutually exclusive publication cases are:

1. **Dependency or preflight failure:** zero model calls; no R592 result, diagnostic, evidence, or receipt namespace.
2. **Runtime instrument failure during FIT:** stop at the first failing completed call or chunk. Publish only
   `induction_centered_fixed_geometry_rung592_invalid_diagnostic.json`, its prefix evidence, and a diagnostic receipt.
   The receipt contains the exact executed call IDs, which must be a prefix of the 639-call FIT manifest, and the first
   failing predicate. It contains no `split_scores`, scientific terminal, or held/null claim. The normal R592 result
   path remains absent.
3. **Complete valid FIT with a scientific null:** exactly 639 calls and complete FIT evidence; publish the normal result
   with the applicable inherited FIT null terminal. SELECT remains unopened.
4. **Complete valid and scientifically held FIT:** freeze all FIT scales, then open SELECT exactly once.
5. **Runtime instrument failure during SELECT:** exactly 639 FIT calls plus a prefix of the 322-call SELECT manifest.
   Publish only the invalid diagnostic, complete FIT evidence, SELECT prefix evidence, and diagnostic receipt. Record
   `select_<predicate_id>` but no cross-split identification claim or normal result.
6. **Complete SELECT scientific null or hold:** exactly 961 calls and complete FIT plus SELECT evidence; publish the
   normal result with the inherited SELECT-prefixed null or the held terminal.

For cases 2 and 5, partial arrays have first-axis length equal to the completed manifest prefix and are never padded to
look complete. The diagnostic auditor reconstructs the prefix and failure only. For cases 3 and 6, the scientific
auditor requires the complete shapes above; missing raw differences or rows are an integrity failure, not a scientific
null.

Publication is receipt-last. Files are written under a temporary evidence namespace, hashed and fsynced, then the
evidence directory and JSON artifact are atomically renamed, and the receipt is renamed last. A recognized prefix
diagnostic can never be recovered as a scientific result. Exceeding the call manifest, opening SELECT early, opening
FINAL/OOD, or publishing a mixed/partial normal result is an integrity abort.

## Independent review and implementation order

This amendment must receive an independent exact-byte review before implementation is accepted. The reviewer must
reconstruct the supersession table, legacy arm identities and bootstrap hashes, frozen-cache delta semantics, all four
hybrid transport checks, centered activity, the call schedule, evidence shapes/bytes, and every terminal closure. Tests
must retain the five blocking fixtures and add old-gate inheritance, mixed machine names, live-removal reintroduction,
missing native logit differences, fake complete partial arrays, and premature SELECT failures.

Only after the amended specification is approved may a prospective producer, managed adapter, dry run, and adversarial
test suite be frozen. Those exact implementation bytes then require a different-agent review before managed GPU use.
