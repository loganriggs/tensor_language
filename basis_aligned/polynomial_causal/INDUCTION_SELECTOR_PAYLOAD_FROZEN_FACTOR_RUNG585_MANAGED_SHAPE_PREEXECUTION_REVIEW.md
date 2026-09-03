# R585 managed-shape repair independent pre-execution review

Date: 2026-09-03 UTC

## Decision

**APPROVE exact commit `c4288dbe8ee6213dfc4dcb538024dc119fbb642e` for the registered R585 managed retry.**

The repair changes exactly the three scientific facade calls from
`require_production=True` to `require_production=False`. This removes the
facade's unrelated fixed `(4, 256)` input-shape assertion while preserving the
model computation, frozen variable-padding schedules, checkpoint and model
validation, result/evidence validation, and exact forward budget. A whole-file
AST census finds no fourth facade forward, alias, or direct-model forward path.

This approval licenses only the exact five blobs below. It does not enqueue the
job or predict a scientific outcome.

## Outcome blindness and exact reviewed bytes

I reviewed immutable Git blobs and a detached worktree, never the moving
producer. At `2026-09-03T23:21:59Z`, existence-only checks found the R585 result
JSON, receipt JSON, and evidence directory absent. I did not open, stat, or
inspect an R585 outcome. No model, CUDA/GPU, queue, or registry operation was
performed.

| File | SHA-256 |
|---|---|
| producer | `fd772c3b9d6df4271ecbfc90c00c893db5a65ea06601f0c8f6e7a9e34c9a531b` |
| owner test | `fcaba664269de12a41a5adb8ff089fc9963eeec91577ef94993ff032c02fc885` |
| dry run | `580a570426ce48c9e43f5fce82c976dece6c71e8a11c1b057054c17cf958dcf8` |
| managed adapter | `a65b12c2e88ae57c4d563219ed76f14ddb413b77c4cafcb757a0af415278883a` |
| adapter test | `725f0af145ae0883449ac93b7bdb7f29b1c2cc313d7ffc9e892e10efc74743aa` |

The adapter's `FROZEN_HASHES`, its owner test, and the dry run agree on the
repaired producer, test, and dry-run hashes. The adapter continues to pin the
unchanged dependency lock, manifest, amendment, earlier implementation
reviews/tests, facade, and induction helper.

## Why the change preserves the scientific computation

The three calls are exactly:

1. `collect_capture_replay`;
2. `collect_native_comparator`; and
3. `collect_intervention_arm`.

All call the same `facade.forward_with_dispatch` with the same model, token
tensor, attention dispatcher, and MLP dispatcher as before. The flag does not
select a different forward implementation. Inside the frozen facade it affects
only:

- whether `validate_production_model(model)` is repeated immediately before
  that forward;
- whether `validate_tokens` requires exactly `(4, 256)` rather than any
  nonempty two-dimensional token tensor; and
- whether the final vocabulary dimension is compared with the frozen constant
  or `model.config.vocab_size`.

The model is still loaded by `load_bilin18(...,
verify_weights_sha256=True)`, which calls `validate_production_model` once after
strict state-dict loading. `run_science` separately compares the returned
checkpoint digest with `CHECKPOINT_SHA256`. No code mutates model weights after
load. Thus the config vocabulary size has already been checked against the
production model.

With `require_production=False`, token validation still requires a nonempty
rank-2 `torch.long` tensor and checks every token ID against the 50,257-token
support. Dispatcher output shapes, dtype, device, and finiteness remain checked
at every layer. Final logits still have to match the actual input batch and
sequence dimensions plus the validated vocabulary size, and must be finite.
The residual, attention, MLP, normalization, readout, and soft-cap calculations
are byte-identical.

R585 deliberately uses batch size 32 with semantic lengths 19–30 and variable
padding. Requiring `(4, 256)` was therefore an adapter-shape error, not a
scientific constraint in the amendment. Removing that assertion lets the
frozen schedule run without changing examples or padding.

## Complete forward-path census

The owner AST test checks one call in each named collector. The independent
test strengthens this to a whole-producer census:

- exactly three AST calls have attribute `forward_with_dispatch`;
- their enclosing functions are exactly the three collectors above;
- all three pass the literal boolean `False`;
- no assignment aliases `forward_with_dispatch`; and
- no `model(...)`, `model.forward(...)`, or `model.__call__(...)` bypass exists.

`run_science` directly invokes each collector and no other recognized
scientific collector. The single intervention call site is inside the three-arm
loop, as before. There is no hidden fourth scientific facade path for the new
test to miss.

## Price and evidence contracts

The batch schedules are unchanged:

- FIT: 54 capture + 54 native-comparator + `3 * 117` intervention forwards =
  459;
- SELECT: 27 capture + 27 native-comparator + `3 * 59` intervention forwards =
  231;
- maximum registered total: 690 forwards, zero backwards and weight updates.

All accumulated R585 contracts remain unchanged and covered: semantic
recipient/donor authority, exact factor and independent-remainder identities,
live removed plus delta reconstruction, full-logit structural hard aborts,
split scoping, primitive logits and causal statistics, exact operation census,
typed scales, realized bootstraps, failure-list derivation, terminal precedence,
strict finite JSON, receipt-last atomic publication, conservative recovery, and
FINAL/OOD closure.

## Tests

Exact detached-worktree checks:

- owner suite: `61 passed`;
- managed adapter: `11 passed`;
- manifest: `14 passed`;
- replacement adversarial: `13 passed`;
- next-wave adversarial: `11 passed, 3 xfailed`;
- first repair review: `4 passed, 3 xfailed`;
- second repair review: `5 passed, 4 xfailed`;
- prior final review: `8 passed, 3 xfailed`;
- iteration-4 review: `4 passed, 1 xfailed`;
- iteration-5 review: `5 passed, 1 xfailed`;
- iteration-6 final review: `7 passed`;
- producer and adapter static gates: pass;
- producer and adapter preflight: pass;
- managed no-argument dry run: pass with zero model calls.

Historical strict xfails remain bound to earlier immutable candidates; they are
not failures of the managed-shape repair.

New independent test:

`basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_frozen_factor_rung585_managed_shape_review_adversarial.py`

- `6 passed`;
- static gate: pass;
- preflight: pass;
- `git diff --check`: pass.

## Remaining scientific uncertainty

This repair changes execution compatibility only. It adds no evidence that the
score/payload factor is identified, sufficient, selective, reusable, or stable
outside the registered counterfactuals. Those questions remain determined by
the prospective R585 outcome and later held-out/OOD and selective-removal work.
