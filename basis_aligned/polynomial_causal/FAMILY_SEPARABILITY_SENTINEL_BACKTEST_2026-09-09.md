# Retrospective sentinel-cover backtest for family separability

**Time:** 2026-09-09 00:41 UTC
**Scope:** CPU-only analysis of two immutable exhaustive family receipts; no model execution,
projector refit, circuit recount, or prospective claim.

## Decision and limitation

The exhaustive family protocol is quadratic because every target projector is optimized against
nearly every sibling and then scored on every sibling. The measured approximation

```text
seconds = number_of_targets * (19 + 9.3 * number_of_controls)
```

gives `18,840.8 s = 5.23 h` for 44 targets and 44 controls. This analysis asks whether a small,
rotating control panel can expose the **known** collision geometry in completed receipts.

It cannot show that a projector newly trained on those sentinels remains inert on unseen siblings.
That requires a prospective fit-sentinel versus audit-sentinel test. The sentinel method is a cheap
screen and scheduling policy, not a replacement for occasional exhaustive calibration.

## Authority and computation

- verb family: `unit_family_separability_spec_v281_result.json`, SHA-256
  `7b8fcbf2d56e0aec387c938f63d71261089d95700fe8421825c7bd7ffe82ef46`;
- adjective family: `unit_family_separability_spec_v260_result.json`, SHA-256
  `61b6b4e3748c000eaf4350e67871d79d705f9b11a56a220ef71a539ebcc9c912`.

For each target `t`, define a collision edge to sibling `s` when the target's own-C-trained
projector caused absolute held sibling CE damage above the frozen `.05` cross-family bar:

```text
edge(t,s) = 1[abs(CE_damage(t projector on sibling s)) > .05].
```

A greedy set cover repeatedly chooses the sibling detecting the most uncovered targets. After one
cover is complete, its controls are removed and the algorithm constructs another, giving disjoint
fit/audit/rotation panels. Lexicographic tie-breaking makes the computation deterministic.

## Results

Every evaluated target in both receipts had at least one own-projector collision. Nevertheless, the
collision graph was highly compressible:

| family | targets with a collision | cover 1 | cover 2 | cover 3 |
|---|---:|---:|---:|---:|
| verb preposition | 22 | 3 sentinels | 3 | 3 |
| adjective preposition | 15 | 2 sentinels | 3 | 3 |

The first verb cover is `with_against`, `through_with`, and `over_with`; it detects all 22 targets.
The second and third disjoint covers also detect all 22. The first adjective cover is `fond` and
`of_toward`; it detects all 15, and two disjoint three-control covers independently do the same.

Under the empirical timing model, auditing ten new/weak targets against three controls costs
`469 s = 7.82 min`, inside the circuit-loop target. Rechecking all 44 targets even against three
controls would still cost `34.4 min`, so reducing the control panel alone is insufficient: each
cycle must also restrict targets to new members, previous boundary cases, and a rotating incumbent
slice.

## Prospective use

The next family audit after v289 should freeze one disjoint cover for fitting and another for sealed
validation, always include new members and prior weakest-margin incumbents, and rotate the remaining
incumbents. Kill the shortcut if validation sentinels miss a collision found by a scheduled full
calibration, or if a sentinel-trained projector changes extraction/control decisions beyond the
frozen tolerance. This changes circuit grouping efficiency; it does not change the evidence tier of
any existing circuit.
