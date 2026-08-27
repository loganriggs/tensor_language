# C512 → MLP1 physical interchange v3: authoritative result

## Verdict

The physical assay does **not** license a live-background MLP1 adapter and does not
certify C512 as an equivalent causal interface. It does, however, sharply localize
the discrepancy:

1. the C512 upstream state with the exact MLP1 write is close to the exact parent;
2. almost all measured C512-parent damage is carried by the changed physical MLP1
   write, not by the pre-MLP1 residual state;
3. the state-by-write interaction is tiny on the registered logit scale;
4. deployed MLP2 suppresses most of this write error, whereas omitting MLP2 exposes a
   large, powered CE failure;
5. consequently the missing interface is a three-layer compensation involving MLP2,
   not a licensed standalone MLP0 → MLP1 repair.

This is a causal localization result, not executable recovery. C512 retains zero
whole-model recovery credit, and no conditional compiler is licensed by this assay.

## Artifact and integrity

- Result:
  `basis_aligned/bilinear_quotient/mlp0_c512_mlp1_interchange_v3_results.json`
- Result SHA-256:
  `43958396ff76739cde3063e4c02bad1f7807b4890830c4ab059196cb3e8f5477`
- Frozen authority:
  `basis_aligned/bilinear_quotient/mlp0_c512_mlp1_interchange_v3_eval_authority.json`
- C512 program SHA-256:
  `3ecf43b485d343bc5413e817dbd4236e5ce6cdaa7a3e0e653214e812b84ce470`
- Rows: 384 new FineWeb source documents, 1,170 windows, split into two
  independent 192-document waves; 192 code windows from 48 independent source
  files.
- FineWeb coverage: 0.93779 / 0.93671 in waves A/B and 0.93724 pooled.
- Code coverage: 0.86424, below the preregistered 0.90 OOD gate.
- Parent replay is bit-identical at raw logits, capped logits, and CE in both live
  and MLP2-omit backgrounds.
- Exact call counts pass: 0 candidate calls to original MLP0 `Down`, 1 poison
  canary call, 2,968 MLP1 teacher calls, and 1,364 C512 proxy calls.
- Exact/candidate `x0` and carried `v1` identities are both zero-error on FineWeb
  and code.
- The saved sufficient-statistics ledger reproduces the complete 20,000-draw
  simultaneous source-unit inference object exactly.
- Runtime: 374.44 seconds.

V1 and V2 execution failures occurred before any evaluation forward and remain
preserved separately. V3 is the first scientific result in this namespace.

## Physical factorial

For exact path `O` and C512 path `C`, the assay captures the pre-MLP1 residual
`s` and physical MLP1 write `m`, then replays an identical suffix from

```text
OO = s_O + m_O        exact parent
CC = s_C + m_C        C512 observational parent
CO = s_C + m_O        C512 state with exact-state MLP1 write
OC = s_O + m_C        exact state with C512-state MLP1 write
```

The registered contrasts are therefore:

- `CC versus OO`: complete observational C512 effect;
- `CO versus OO`: upstream-state effect after installing the exact MLP1 write;
- `OC versus OO` and `CC versus CO`: MLP1-write effects on the two states;
- `CC` versus the centered additive prediction: state-by-write interaction.

## FineWeb result

The table reports the maximum standardized effect across 16 cells and the three
registered consumers. Equivalence requires each wave UCB below 1 and pooled UCB
below 0.8.

| background / contrast | wave A point / UCB | wave B point / UCB | pooled point / UCB / LCB |
|---|---:|---:|---:|
| live / observational `CC` | 0.7747 / 1.6486 | 0.8029 / 1.7203 | 0.7807 / 1.4126 / 0.1489 |
| live / write on `O` | 0.7438 / 1.6177 | 0.7721 / 1.6895 | 0.7498 / 1.3816 / 0.1179 |
| live / write on `C` | 0.7467 / 1.6206 | 0.8102 / 1.7276 | 0.7526 / 1.3844 / 0.1207 |
| live / upstream state | 0.1042 / 0.9781 | 0.1054 / 1.0228 | 0.1048 / 0.7367 / -0.5270 |
| live / interaction | 0.0287 / 0.9026 | 0.0291 / 0.9465 | 0.0289 / 0.6608 / -0.6030 |
| MLP2 omit / observational `CC` | 3.4727 / 4.3466 | 3.4847 / 4.4021 | 3.4718 / 4.1036 / 2.8399 |
| MLP2 omit / write on `O` | 3.4918 / 4.3657 | 3.4917 / 4.4091 | 3.4918 / 4.1236 / 2.8599 |
| MLP2 omit / write on `C` | 3.5300 / 4.4039 | 3.5312 / 4.4486 | 3.5306 / 4.1625 / 2.8988 |
| MLP2 omit / upstream state | 0.1346 / 1.0085 | 0.1363 / 1.0537 | 0.1349 / 0.7668 / -0.4969 |
| MLP2 omit / interaction | 0.0386 / 0.9125 | 0.0373 / 0.9547 | 0.0367 / 0.6686 / -0.5951 |

The binding live cell remains
`pos0_freq0_prev0_dev1`: early positions, low fit-frequency tokens,
non-punctuation predecessors, and high pre-MLP0 residual norm. Its pooled live
observational centered-logit nRMSE is 0.03904 against the 0.05 margin. The wave-B
CE ratio is 0.8029, corresponding to 0.00602 nat/token against the 0.0075 margin.

Under MLP2 omission the binding observational CE is 0.02604 nat/token, or 3.4718
times its margin, with a pooled simultaneous LCB of 2.8399. That is a powered
failure, reproduced in both waves.

## What is localized

### The upstream state is small after the exact MLP1 write

`CO versus OO` has pooled point maxima 0.1048 live and 0.1349 with MLP2 omitted.
The exact MLP1 write removes roughly 87% of the live family maximum and 96% of the
MLP2-omit maximum at the point-estimate level.

Every registered coordinate is pointwise no worse after this repair in both waves
and both backgrounds. Under MLP2 omission the family-wise rescue lower bounds are
positive in both waves:

| scope | MLP2-omit point reduction | simultaneous rescue LCB |
|---|---:|---:|
| wave A | 3.3381 | 1.5903 |
| wave B | 3.3484 | 1.5136 |
| pooled | 3.3368 | 2.0731 |

Thus the exact MLP1 write causally repairs the exposed MLP2-omit failure.

### The MLP1 write carries the mismatch

The two MLP1-write contrasts nearly equal the complete observational contrast in
each background. Their pooled maxima are approximately 0.75 live and 3.5 with
MLP2 omitted, while the upstream-state maxima are only 0.10 and 0.13.

The state-by-write interaction is also very small: pooled point maxima 0.0289 live
and 0.0367 with MLP2 omitted. On the registered output metrics, the factorial is
close to additive. The difficult composition is therefore not an inseparable
`s × m` interaction at MLP1; it is the downstream response to the changed MLP1
write.

### MLP2 is the compensating interface

With deployed MLP2, the observational family maximum is 0.7807 and the suffix is
not detectably sensitive to the norm-matched native-write control: its pooled point
maximum is 0.3579 and UCB is 0.9897. The live exact-write rescue has a favorable
point reduction of 0.6759 but a simultaneous LCB of -0.5878, so it is not powered.

After omitting MLP2, observational and native-write effects both become large and
powered. Exact-write rescue becomes strongly positive. Holding every other suffix
operation fixed, the presence of MLP2 therefore suppresses the MLP1-write
perturbation. This is direct evidence of a compensating MLP1 → MLP2 interface.

It does not yet say whether compensation is carried by the MLP2 write itself, by
the state passed through MLP2 attention, or by their interaction. A physical MLP2
factorial is required for that decomposition.

## Registered decisions

```text
fresh observational equivalence:                 false
downstream null on registered backgrounds:       false
powered cancellation/interface-break label:      false
live MLP1 repair license:                         false
broad code-register equivalence:                  false
```

The apparently contradictory `powered interface break = false` is intentional.
The MLP2-omit observational failure is powered, but the registered cancellation
label required observational equivalence first. Live C512 is inside the point
margins but misses simultaneous equivalence; it is neither certified equivalent
nor poweredly rejected on the live background.

The live repair license is also correctly false: although pointwise repair is
large and never worsens a registered coordinate, the live suffix lacks a positive
sensitivity control and the simultaneous rescue LCB crosses zero.

## Code OOD result

Code support passes, but evaluated coverage is only 86.42%, below the registered
90% gate. Point maxima are also much larger than FineWeb—4.088 live and 3.505 with
MLP2 omitted for observational `CC`—while the 48-file simultaneous intervals are
wide and their LCBs cross zero. The code register therefore vetoes broad OOD
equivalence but does not license a powered causal rejection.

## Consequences for the MLP0 program

1. C512 remains a good continuous **observational** approximation on natural text,
   but it is not a certified standalone causal module.
2. The primary missing physical variable is the MLP1 write induced by the C512
   state. The upstream residual state itself is not the dominant error after an
   exact write transplant.
3. Fitting a standalone MLP1 adapter is not licensed: deployed MLP2 already changes
   the sensitivity regime, and live rescue is not powered.
4. The next interface decomposition should cross MLP1 and MLP2 physical writes. A
   useful program must explain or reproduce the compensation rather than merely
   reconstructing MLP1 activation space.
5. The older 256-quadratic complete-MLP0 program remains important discovery
   evidence, but its 97.9% legacy denominator cannot be compared directly with this
   source-unit causal family. It should be re-authorized as a matched complete-MLP0
   baseline in a later joint-interface assay.

## Pruned next actions

1. **Physical MLP2 compensation factorial.** On new rows, capture MLP2 pre-write
   state and write under exact/C512 and exact/C512-conditioned MLP1 writes. Cross
   state and write while replaying an identical later suffix. This is the shortest
   falsifiable test of where compensation resides.
2. **Re-authorize the pre-1500 256-quadratic MLP0 program.** Serialize and price the
   complete map `sum_r u_r(a_r·z)^2`, poison the original MLP, and compare it with
   C512 and matched-byte continuous controls under the same physical interfaces.
3. **Only conditionally compile the compensator.** If the MLP2 factorial identifies
   a low-dimensional compensating write, fit it from the live upstream state using
   causal suffix KL/CE, not Euclidean write reconstruction, with no teacher forcing.
4. **Composition test.** Any promoted complete-MLP0 or compensator program must enter
   the exact MLP0/1/2 cube and preserve its conditional gains; singleton CE is not a
   valid selection currency.
5. **Return to whole-model coverage.** Once this early interface is either compiled
   or decisively bounded, the largest untouched scope remains the unevaluable middle
   12 MLPs and the clean-to-ship residual outside the early band.
