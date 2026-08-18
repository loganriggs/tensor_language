# Does a bits axis track what we care about? Worked examples

The proposal on the table: replace "components replaced" on the benchmark
x-axis with "description length in bits." Before adopting it, here is every
kind of stand-in currently in the assembled model, priced three ways, with a
verdict on whether each pricing matches the interpretive intuition.

The three candidate prices:

- **artifact bits** — the literal size of the stand-in's parameters.
- **generator bits** — the size of the *program that produces* the stand-in,
  given the original weights (Kolmogorov-style, informally).
- **fitted bits** — the information taken *from data* (ridge fits, class
  means measured on text); weights-derived pieces cost only their generator.

## 1. mlp1's token table + absorber (the model's most important component)

What it is: "for each vocabulary token, output a fixed vector — computed
from the weights by one tiny forward pass per token — plus a low-rank
quadratic correction read from the stream."

- artifact bits: 50257 x 1152 halfs ≈ **116 MB** — *larger than the
  component's own weights* (~32 MB). Verdict: absurd. The artifact price
  says our best, most interpretable result made the model bigger.
- generator bits: ~a paragraph of code + the (already-owned) weights.
  Tiny. Verdict: matches intuition — the insight is "this component is a
  context-free token function," and that sentence IS the compression.
- fitted bits: the table costs ~0 (weights-derived); the absorber's
  (1152+528) x 64 ridge matrix is fitted, ≈ 430 KB. Verdict: reasonable —
  it prices exactly the part we could not derive.

## 2. A middle-MLP CP truncation (c4: keep 2304 of 4608 hidden units)

What it is: "the same component with half its hidden quadratics deleted."

- artifact bits: half the original weights, ~8 MB. Mediocre compression.
- generator bits: "keep the top half by a norm product" — one line. Tiny.
- fitted bits: zero — nothing came from data.

Verdict: this is the case that breaks single-number pricing in the OTHER
direction. Generator bits and fitted bits both call it nearly free, but
interpretively we learned almost nothing — the stand-in still performs an
opaque computation, and ledger 21 showed the selection rule carries no
information (a random half works). Cheap description, no understanding.
Any bits-only axis will overrate this rung.

## 3. attn5's mean vector (a scaffolding component)

What it is: one 1152-dim vector — "the assembly only needs this
component's average contribution."

- artifact bits: 4.6 KB. - generator bits: same. - fitted bits: same.

Verdict: all three agree, and they match intuition: a real discovery
(scaffolding role), honestly priced as tiny. The metric works when the
stand-in class is genuinely simple.

## 4. A tail attention class dictionary (e.g. a13)

What it is: "at digit / bracket / sentence-end / comma / name / repetition
sites, output that class's constant; elsewhere, a per-class linear read."

- artifact bits: 10 constants (46 KB) + 4 linear maps (~21 MB for D x D
  floats each). The linear arms dominate and are NOT interpretable objects.
- fitted bits: same as artifact here — all measured from data.

Verdict: the bits axis usefully punishes exactly the right thing: the
constants (the circuit content) are cheap, the opaque linear arms are
expensive. If we adopt fitted-bits, there is immediate pressure to
rank-reduce the linear arms or replace them with named features — which is
the scientifically right pressure.

## 5. The tail span dictionaries (mlp10–17)

Eight 8-dim spans + 10 constants per layer + rank-full linear arms for two
classes. Same shape as case 4: circuit content cheap, escape-hatch linear
maps expensive. Fitted-bits pressure points at the escape hatches.

## Verdict and proposal

No single bits number tracks understanding: artifact bits fail on tables
(case 1), generator/fitted bits fail on truncations (case 2). What we
actually care about seems to be two-dimensional:

1. **fitted bits** (information taken from data, weights-derived programs
   priced as programs) — this is the honest "how much did we have to
   memorize" axis, and cases 1, 3, 4, 5 all price sensibly under it;
2. **computation class per slot** (lookup < constant < linear < low-rank
   quadratic < opaque-subnetwork) — a qualitative tag that catches case 2:
   a CP truncation stays "opaque-subnetwork" no matter how cheap its
   generator is, so it visibly does not count as explained.

Concrete next step if we adopt this: the frontier graph gets fitted-bits on
the x-axis, each point annotated with its census (how many slots are still
"opaque-subnetwork" — currently 5 of 36, the CP middles), and the CP rungs
stop counting as "replaced" in the headline number unless we also state the
opacity census. The current honest headline under that rule would be:
**29 of 36 slots in interpretable computation classes at +2.75 nats**, with
5 more slots compressed-but-opaque.
