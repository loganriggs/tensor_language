# Rung495 preregistration — exact attention1 QK1 × QK2 × OV pieces grouped by downstream use

## Question and claim level

Do pieces below attention1's native-head boundary become the same operational variable when the existing downstream
circuits read them the same way? This is a **screen** for cross-head grouping and within-head splitting. It is not an
executable replacement, compression result, or causal interchange claim.

The circuit targets are cross-module grouping, within-module splitting, held-out prediction, stable identification,
and a concrete candidate for selective manipulation. Rank and reconstruction are instrument checks only.

## Frozen authority and data split

- Model and exact MLP0 `T/C/I/S` branches: the rung401 authority reached through rung493.
- Attention factor implementation and closure convention: rung484.
- Normalized attention-to-suffix derivative convention: the production facade and rung477b's downstream-response
  contraction.
- Downstream masks: all 62 curated circuit tags in `circuits/BATTERY.json` and `census_state_diverse.pt`.
- Discovery uses documents0:500 and the 32 discovery tags already frozen by rung477b/rung481.
- Candidate selection uses documents0:250 only. Documents250:500 are the fixed confirmation half.
- The 30 validation tags and documents500:1000 remain unopened unless the discovery and confirmation gates select
  exactly one cross-head pair. No odd-root, FINAL, SEALED, or other reserved outcome is opened.

## Exact pieces

At attention1, for each head and query/source position, define `A` as the first normalized Q/K score, `B` as the
second normalized Q/K score, and `V` as the value followed by that head's slice of the output projection. For each
MLP0 branch removal, rebuild the eight combinations choosing each of `A/B/V` from the normal or branch-absent
trajectory. Möbius subtraction gives the seven nonempty finite pieces

`A, B, V, A×B, A×V, B×V, A×B×V`.

There are `9 × 7 = 63` pieces. They must sum to the complete normal-minus-branch-absent attention1 write. Native head
identity is retained only as provenance; comparisons include all cross-head pairs and all within-head factor pairs.

The seven-term equality is exact at attention1's **raw residual-stream write**. It is deliberately not propagated
piece by piece using MLP1's quadratic polarization. The production model first computes

`z = RMSNorm(residual_before_attention1 + attention1_write)`

and only then applies MLP1. RMS normalization couples every raw attention piece with every other piece and with the
incoming residual. Treating the raw pieces as additive MLP1 inputs would therefore be false. This correction was
made before any downstream-use outcome was opened.

## Downstream-use signature

For each circuit tag, compute separately the mean CE on its member positions and its matched in-slice control
positions. Differentiate each scalar with respect to attention1's **raw write** on the branch-absent trajectory. This
derivative runs through the real RMS normalization, MLP1, the direct residual route, and all later layers. The
coordinate assigned to raw piece `theta` is the gradient inner product `<dCE/d attention1_write, theta>`. The signed
fingerprint is `member response - control response`.

Because differentiation is linear, the 63 piece responses must sum exactly, up to floating-point contraction error,
to the derivative applied to the complete normal-minus-absent attention1 write. This is an exact first-order
identity of the production graph, not a claim that a finite raw-write replacement is additive.

Fingerprints stack all four MLP0 branches. A piece is material when its fingerprint norm is at least 5% of the
complete attention-route fingerprint norm in both discovery halves. Pair similarity uses cosine and the symmetric
best-scale residual `sqrt(1-cosine²)`; the reported left-to-right scale is `<left,right>/||left||²`. Sixteen
circuit-label permutations are computed by permuting the right member of a pair only. If the preliminary
top pair passes every non-position bar, a second pass over the same discovery documents computes sixteen
token-position rolls for that frozen pair only; it cannot change the selected pair. No failed branch, factor type,
head, tag, or document half may be dropped.

This derivative is a first-order measurement of downstream use. It does not license a physical swap. A selected
pair must face a separately preregistered natural-state interchange in the next rung.

## Predictions

### A — exact and live instrument

All frozen hashes, branch identities, replay checks, circuit support counts, calls, and backward counts match.
The eight factor arms reconstruct both endpoint attention writes with relative squared error at most `1e-10` in
float32. The 63 Möbius pieces reconstruct the complete attention difference to `1e-10`. Contracting each piece with
the real downstream gradient and then summing reconstructs the complete attention-difference contraction to
`1e-9`. Every branch has nonzero complete attention-write and downstream-response norm. Gradients and every control
are finite and nonzero.

### B — one cross-head downstream equivalence survives discovery confirmation

Using only documents0:250, choose the material cross-head pair with the highest stacked signed-fingerprint cosine;
ties are broken by lower scaled residual and then lexical piece name. It passes discovery only if cosine is at least
`.90`, best-scale residual is at most `.45`, each piece is a mutual nearest cross-head neighbour, and the cosine is
at least `.10` above the 95th percentile of both control families.

The frozen pair then passes documents250:500 without reselection if cosine is at least `.80`, scaled residual is at
most `.55`, both pieces remain material, the fitted discovery scale changes by at most 50%, and both control margins
remain at least `.05`. These bars must hold for the raw member-minus-control fingerprint and after removing each
circuit coordinate's shared four-branch mean.

### C — the selected relation predicts held-out circuit families and documents

Only if A and B hold is the same named pair evaluated on documents500:1000 and the 30 validation tags. It passes if
both fixed halves have cosine at least `.75`, scaled residual at most `.60`, both pieces remain material, the sign of
their discovery scale is preserved, and both control margins remain at least `.05`. No pair is reselected.

### D — at least one native head contains downstream-distinct pieces

Using the same fixed halves, select on documents0:250 the within-head pair with the smallest cosine among pieces that
are each material. It counts as a split only if its cosine is at most `.20`, its scaled residual is at least `.85`,
and documents250:500 preserves cosine at most `.30`, residual at least `.80`, and materiality. D is descriptive and
does not rescue failure of B.

### E — interpretation

E is true only if A, B, and C hold. The selected cross-head pieces are then called a downstream-use-equivalent
candidate, not a circuit. The next required evidence is a physical two-way interchange on held-out natural and code
text which preserves the selected 62-circuit effect while disturbing unrelated circuits less than a whole-head swap.

## Nulls and routing

- Strong null: A fails, or B fails. Do not open validation.
- A true/B false: the 63 exact `QK1 × QK2 × OV` pieces do not expose a stable cross-head variable under the current
  32 downstream probes. Next split the score side itself into query versus key changes and test shared halves; do not
  tune a rank or loosen the pair bars.
- A/B true/C false: the relation is corpus/tag-specific and not identified.
- A/B/C true: preregister the physical interchange before opening its outcomes.
- D may establish that native heads are too coarse even when no cross-head merge exists, but it does not by itself
  license validation or adoption.

## Literal price

Discovery uses 500 documents in batches of four. Per batch it makes one normal capture and four branch-absent
forward/gradient captures: `125 × 5 = 625` model forwards. Factor recombination and gradient contractions are
standalone tensor operations, not additional model forwards. The exact backward count is computed from the frozen
nonempty circuit masks before model loading and must match the receipt. The selected-pair position-control pass has
the same 625-forward ceiling and runs only after a preliminary candidate passes every other B bar. Conditional
validation has the same 625-forward ceiling and collects only the frozen pair plus its controls. Thus the maximum
discovery price is1,250 forward-equivalents and the maximum conditional validation price is625. The experiment
saves zero deployed parameters and adds zero runtime parameters.

No threshold, factor vocabulary, tag split, branch set, control, or validation condition may change after any
downstream-use outcome is opened.
