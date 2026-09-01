# Rungs 436–437 preregistration — mixed-precision repair and signed adoption gate for the sub-500M tier

Date: 2026-09-01T21:43Z

Status: frozen before implementation or execution. Rung 437 is conditional on every rung436 positive and a false
strong null. No precision, rank, layer, population, or threshold tuning is permitted after either receipt.

## Why this is the next adoption-track object

Rungs392/393 adopted the 495,847,230-scalar QK64 + MLP0/4-p768 + factored-MLP16 program, but its source-format
storage was 1,867,449,228 bytes. Rung414 physically converted the source, QK factors, generated MLP programs, and
the 14,984-value degree-two MLP16 program to two-byte storage. Its aggregate behavior held, but it failed one frozen
tail check: maximum per-position change relative to rung392 was `.117641`, above `.100`.

The pre-outcome diagnosis recorded in the board and hourly reviews was specific: linear-family programs had survived
two-byte storage, while the first quadratic program amplified coefficient rounding. The registered repair changes
only the storage dtype of the four-tensor MLP16 program from BF16 to FP32. It is not a relaxation of rung414; rung414
remains failed.

## Rung 436 exact artifact and price

- source-native tensors: the same source-aware BF16 treatment as rungs412–414;
- 440 QK rank64 factor pairs: physically FP16, runtime FP32;
- generated MLP0 and MLP4 p768 programs: all ten tensors physically BF16, runtime FP32;
- MLP16 program: exactly four tensors and 14,984 values—`output_directions[4,1152]`,
  `form_vectors[4,2,1152]`, `form_values[4,2]`, `constant[1152]`—physically FP32, with no dense forms;
- whole artifact: exactly **495,847,230 scalar values** and **991,724,428 bytes** (`.9236153476 GiB`).

The byte increase over rung414 is exactly `14,984 * (4-2) = 29,968` bytes, or `.00302%` of its bill. Scalar count
does not change. This is storage precision, not interpretability or compression by itself; runtime still explicitly
converts stored tensors to FP32 before arithmetic.

Use the unchanged census/certificate population and saved rung392 CEV. The shifted-text population is the next
untouched WikiText-103 segment `[470824,501664)`, 120 rows of width257. FINAL stays closed.

## Rung 436 frozen predictions

### A — physical identity

Enumerate and verify all source/QK/MLP/MLP16 shapes, dtypes, devices, counts, live hooks, fit roles, selected layers,
no-dense-form condition, and the exact scalar/byte bill. Every non-MLP16 storage family must match rung414's declared
precision. The saved MLP16 FP32 artifact must be byte-identical to the original rung392 four-tensor program.

### B — the degree-aware repair removes the position-level failure

- census damage at most `.070` nat and at least 10 of 62 certificates;
- mean/max absolute per-position CEV difference from rung392 at most `.010/.050`;
- maximum difference at most half rung414's stored `.11764144897460938`.

The `.050` maximum was predicted publicly before this registration; it is not selected from the repaired result.

### C — composition law remains intact

Composition-tax ratio lies in `[.90,1.35]`, normalized 62-vector cosine with the independently saved additive
prediction is at least `.95`, and certificate-count difference is at most 7.

### D — shifted and fresh prediction hold

On untouched WikiText-103, full original-native mean/p95/max damage is at most `.075/.140/.220` nat. Conditional
fresh maximum is at most `.040` nat.

### Strong null and routing

The strong null fires on any physical identity failure, census damage at least `.10`, at most 5 certificates,
shifted mean at least `.10`, inert MLP16 hook, or repaired maximum CEV difference at least rung414's stored maximum.

A/B/C/D with null false licenses only rung437. Any miss preserves rung392/393 as the adopted FP32-source tier and
rung414 as a physical near-miss; no further precision sweep.

## Conditional rung 437 — original-native signed gate

If licensed, apply the same fixed original-native attention16 mean knockout used by rung393. The native knockout is
measured before candidate installation. Candidate signed effect is knockout CE minus the candidate's own unablated
CE at every census position.

Frozen predictions:

- **A:** reproduce rung436 census `<=.070`, certificates `>=10`, shifted max `<=.220`, fresh max `<=.040`, every
  source/QK/MLP/MLP16 physical identity, live factor and ablation hooks, and exact scalar/byte bill;
- **B:** candidate/native signed-effect cosine at least `.95`, normalized error at most `.40`, and norm ratio in
  `[.70,1.30]`;
- **C:** collateral circuit-effect Spearman at least `.95` and median magnitude ratio on attention16-owned behaviors
  in `[.70,1.30]`.

Strong null: cosine below `.70`, collateral Spearman below `.75`, inert effect, dead hook, or baseline identity
failure. A/B/C with null false adopts the mixed-precision 495,847,230-scalar / 991,724,428-byte program as the
physical lower-fidelity tier. This licenses no latency claim and does not make floating-point compression an
interpretability result; it only makes the stored artifact's price honest.
