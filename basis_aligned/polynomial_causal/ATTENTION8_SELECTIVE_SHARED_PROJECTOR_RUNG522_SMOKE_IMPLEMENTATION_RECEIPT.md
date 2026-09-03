# Rung 522 managed-smoke implementation receipt

Frozen at `2026-09-03T06:08:00Z`, before any rung522 CUDA execution.

## Scope

This receipt opens only the instrument-only managed GPU smoke. It does not open model science, does not train a
scientific projector, and does not compute or retain a circuit mask, cross-entropy response, task score, or any
registered A--D outcome. The smoke asks only whether the registered batch shapes execute, whether attention8 is
called exactly once per batch, whether every projected edit is nonzero, whether gradients reach the rank-4 frame,
and whether gradients remain absent from frozen model parameters.

The full scientific entrypoint does not exist at this freeze. Its byte hash and dependency census require a later
pre-outcome receipt before any scientific rung522 run, as clarified prospectively in the preflight addendum.

## Frozen files

| File | SHA-256 |
|---|---|
| `basis_aligned/polynomial_causal/ATTENTION8_SELECTIVE_SHARED_PROJECTOR_RUNG522_PREREGISTRATION.md` | `27bc74c3e19ac310f0ed88f1527a1df44ff52d8990d980971415b32b503126f5` |
| `basis_aligned/polynomial_causal/ATTENTION8_SELECTIVE_SHARED_PROJECTOR_RUNG522_PREFLIGHT_ADDENDUM.md` | `d3343c0acc8233580cdb209d1652c7d30c839823b399f9d19e7ba923ffe53b22` |
| `basis_aligned/bilinear_quotient/ops/attention8_selective_shared_projector_rung522_math.py` | `6cff6f7726dd8f76e786d64abf913fc31adbdfec101a97741a1aa3396f8431c2` |
| `basis_aligned/bilinear_quotient/ops/attention8_selective_shared_projector_rung522_toy_preflight.py` | `5abbb09ec0871e0d7ad5b8cb63a3f6103027848700df36fcc3dc85ce21c42935` |
| `basis_aligned/bilinear_quotient/ops/test_attention8_selective_shared_projector_rung522_math.py` | `42b7b2f41fccf1c4f662f0b7dfeddc6f59836c9a9b26b611977007d8c00542c7` |
| `basis_aligned/bilinear_quotient/ops/attention8_selective_shared_projector_rung522_toy_preflight_results.json` | `398842217e729e743dc4b5fe4947dc7837a40e01a42b2c267faa2249a6ad0fe4` |
| `basis_aligned/bilinear_quotient/ops/attention8_selective_shared_projector_rung522_scheduler.py` | `a92be329e1100d2ceb3c6a0d8035f78de73d13614bc1e4a65e1b08cf7a315a75` |
| `basis_aligned/bilinear_quotient/ops/test_attention8_selective_shared_projector_rung522_scheduler.py` | `6f55187aa7cc01b3e96285be7bbc0cf9518ffeee553cfe7bf4a21da34c21c6f1` |
| `basis_aligned/bilinear_quotient/ops/attention8_selective_shared_projector_rung522_gpu_smoke.py` | `4881d27f984b48b5944cc22e9e403063ce4acc3d8d52e69909be2d4533928462` |
| `basis_aligned/bilinear_quotient/ops/test_attention8_selective_shared_projector_rung522_gpu_smoke.py` | `6385dfb552e29398a6e1127a67886a1d834058788662adbb4dcdfe5716970c04` |
| `basis_aligned/bilinear_quotient/attention8_shared_private_das_rung521_stage_a_results.json` | `6a303e0e62ef3d2443ed6d667f74bc28c703a79ce5f462657bff212c1c5a676c` |

The smoke validates the seven scientific dependencies it imports before importing PyTorch or model code. The
scheduler and its tests are frozen here as part of the eventual scientific implementation but are not imported by
the smoke.

## CPU validation

- Python compilation: pass.
- Focused math, scheduler, and smoke tests: `19 passed`.
- Experiment gate: pass with all three registered smoke predictions recognized.
- Hash-only dry run with model imports forbidden: pass; reports seven checked hashes, physical batch
  `6 * 8 * 2 = 96`, differentiable batch 6, and no retained science metrics.
- Shared standalone fast checks: `PASSED 0 failing`.

Any dependency mismatch, failed smoke predicate, out-of-memory error, dead edit, extra attention8 call, absent frame
gradient, or model-parameter gradient closes model science. It does not license changing the physical batch sizes or
the registered scientific budget.
