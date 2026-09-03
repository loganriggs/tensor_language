# Rung 532 preregistration: downstream-defined equality of one score factor

**Registered:** 2026-09-03 12:25 UTC

**Owner:** Codex

**Status:** implementation in progress; no rung 532 outcomes inspected

## Why this changes the object

Rung 531 rejected one-to-one scalar matching of complete score-factor matrices across four equality heads. That was
a function-space test. It did not test the user's proposed criterion: two internal quantities should count as the
same when later computation uses them in the same way.

This rung performs that causal test. It takes one factor from `L8H3`, composes it with the other, native factor from
`L8H4`, and asks whether the resulting physical attention term has the same effects as native `L8H4` across the 62
pre-existing circuit families. It therefore tests cross-head grouping, within-head splitting, held-out prediction,
selective manipulation, and source-by-companion interaction. It is not a rank or compression experiment.

## Frozen pair and scales

Rung 531 used rows `0:500` and selected `L8H3 -> L8H4` without using downstream outcomes because:

- it is one of the three parent-authorized portable equality-score pairs;
- its direct/swap choice was stable on both confirmation halves;
- it selected the swapped assignment; and
- it was the only parent-authorized pair whose branch-scale product agreed with its independently fitted product
  scale within the registered 10% bar (observed difference `2.836%`).

Write the donor factors as `(a,b)` and target factors as `(c,d)`. Freeze the rung 531 values:

```text
c_hat = alpha*b,       alpha =  1.227983240318439
d_hat = beta*a,        beta  = -0.8533769036200292
product_scale gamma    = -1.0785167862928777
```

The direct-assignment control uses its separately frozen rung 531 scales
`alpha_direct=-1.268044102615207` and `beta_direct=0.6995515454196305`.

## Physical score terms

At the equality-fetch edges, the target head normally writes a term using `c*d` and its native value/output path.
For each arm below, replace only that target equality term and retain the rest of the model:

- `native`: `c*d`;
- `absent`: `0`;
- `swapped_first`: `(alpha*b)*d`;
- `swapped_second`: `c*(beta*a)`;
- `swapped_both`: `(alpha*b)*(beta*a)`;
- `product_control`: `gamma*(a*b)`;
- `permuted_first` and `permuted_second`: the corresponding swapped factor after causal-prefix key reversal;
- `direct_first` and `direct_second`: the wrong direct factor with its own frozen scale.

Run every arm in two backgrounds: with the donor `L8H3` equality term present and with that term removed. Because the
two heads are in the same attention layer, their scores read the same incoming residual state; removing the donor
term changes downstream coexistence without changing the score inputs. This is a finite test of the interaction that
ordinary single-component patching would mix into an indirect effect.

## Data and observations

- Use only natural census rows `500:1000`, unseen by rung 531.
- Report document halves `500:750` and `750:1000` separately.
- Evaluate CE changes on the frozen `member` and matched `slice_control` masks for all 32 discovery circuit tags and
  all 30 held-out circuit tags. No tag is fit or selected in this rung.
- Also report the calibrated copy-positive task effect and all-noncopy change.
- OOD/code rows remain sealed.

For each circuit tag, define the target effect in nats per selected token as

```text
E_native[tag] = CE_absent[tag] - CE_native[tag]
E_arm[tag]    = CE_absent[tag] - CE_arm[tag].
```

Compare the 32- and 30-number effect vectors using cosine and relative error, without another fitted scale. Lower CE
is better; positive `E` means the target equality term helps that circuit family.

## Registered predictions

### A — exact live instrument

Native and analytical replay logits agree exactly; every physical edit has nonzero RMS; the factor product matches
the parent product exactly; all 2,625 forwards and every circuit/task support reconcile; checkpoint and source hashes
match; and no OOD row is loaded.

### B — whole-product positive control transfers

In both donor backgrounds and document halves, `product_control` has member-effect cosine at least `0.85`, member
relative error at most `0.60`, and copy-task recovery in `[0.65,1.40]` for both the 32- and 30-circuit tag sets.
This establishes that the downstream test can recognize the already-authorized product-level action on new rows.

### C — the donor's second factor substitutes for the target's first factor

`swapped_first` meets B's cosine, error, and task-recovery bars in both backgrounds, halves, and tag sets. Its member
cosine exceeds both `permuted_first` and `direct_first` by at least `0.15`, and its mean absolute CE change from native
on every slice-control vector is at most `0.01` nat.

### D — the donor's first factor substitutes for the target's second factor

The identical rule holds for `swapped_second` against `permuted_second` and `direct_second`.

### E — held-out interaction-defined factor unit

A and B pass, at least one of C or D passes, and the passing identity is the same in both donor backgrounds and both
the 32- and 30-circuit tag sets. E is the identification claim: downstream computation groups one factor across
heads even though rung 531 showed the raw factor functions are not scalar-equal.

### F — the two replacements compose predictably

For every background, half, and tag set, the member-vector interaction

```text
I = E_swapped_both - E_swapped_first - E_swapped_second + E_native
```

has norm at most 30% of `E_native`, and `swapped_both` differs from `product_control` by at most 30% of
`E_product_control`. This is a compositionality screen, not required for E.

### Strong null

A and B pass, but neither C nor D passes. Then the downstream computation recognizes the product-level equality
action but not either source factor composed separately with the native target companion. Close literal factor
transplants; continue with a shared feature vocabulary or a downstream causal-response basis rather than relaxing
the thresholds.

## Literal price and opening rule

There are 10 physical arms in two donor backgrounds, plus one direct-native identity forward, for exactly 21
forwards per four-row batch and `21 * 125 = 2,625` forwards. There are zero backward passes and no fitted vector
parameters. First run a one-batch, 21-forward managed smoke that exposes only identity/edit/support diagnostics. Only
a passing smoke may open the full rows `500:1000` run.
