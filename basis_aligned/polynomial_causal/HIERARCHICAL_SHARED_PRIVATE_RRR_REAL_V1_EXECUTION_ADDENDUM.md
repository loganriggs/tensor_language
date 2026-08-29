# Hierarchical shared/private RRR real v1 — execution addendum

**Status:** prospective, outcome-blind execution contract. This addendum creates no
authority and licenses no row-tensor deserialization, checkpoint-tensor/model load, or
GPU access until all sources are committed, pushed, and a separate no-outcome authority
is written. Checkpoint metadata and the raw file hash are necessarily bound immediately
before that authority write.

This addendum narrows the larger exploratory grid in
`HIERARCHICAL_SHARED_PRIVATE_RRR_V1_PREREGISTRATION.md` to the smallest endpoint-complete
first execution. Omitted ranks 64 and 256 remain untested and cannot be selected or
claimed from this run.

## Immutable parents and data roles

The runner must content-bind the completed shared-output RRR v2 authority, result, and
receipt with respective SHA256 values `32106d80...c7db`, `19d65e2c...c053`, and
`57f699d6...dd56`. It reuses the exact v2 checkpoint, fit coverage
`fineweb_n96_skip80.pt`, and three repeatedly exposed discovery roles `skip7000`,
`skip11000`, and `skip1200`. Every program choice must be complete before any discovery
role is deserialized. No role is validation, heldout, final, or promotion authority.

## Seven frozen arms

All budgets below exclude the common 224,736,768-float exact-token table, which is
identical across arms.

| Budget | Exact map floats | Shared rank q0 |
|---|---:|---|
| global-q512 | 21,823,488 | 0, 128, 512 |
| typed-q512 | 22,413,312 | 0, 128 |
| independent-q512 | 42,467,328 | 0, 128 |

The q0=0 arm at every budget must be byte-identical in deployed coefficient-map hashes
and ranks to the exact-price independent construction from the same fit statistics.
The global-budget q0=512 arm has zero private ranks and must be byte-identical to the
global rank-512 construction. These are literal execution controls, not approximate
scientific comparisons.

The independent-budget q0=0 arm is the newly realized fit-optimal nonuniform allocation
of 18,432 private slots. The parent's `independent_q512` instead assigns rank 512
uniformly to every site. Their storage is identical, but their factors and CE need not
be. Report q0=0 minus parent-uniform CE as a same-storage comparator; it is explicitly
not a known-answer identity gate.

Fit uses CPU float64. Each factor is independently cast with contiguous `.float()`.
The autonomous callback computes `(embedding @ shared_input) @ shared_basis.T` first,
then computes and adds the private product exactly once. Covered tokens replace that
sum with the immutable native table. The callback may not inspect residual state or
call any native attention/MLP component; attention returns an all-zero v1 sentinel.

## Predictions and reporting

All CE is all-position nats per scored token, separately by discovery role.

1. Global-budget q0=128 beats both q0=0 and q0=512 by at least 0.01 nat on every role.
2. Typed-budget q0=128 beats both q0=0 and the pinned parent's typed-q512 arm by at
   least 0.01 nat on every role.
3. Independent-budget q0=128 beating q0=0 by at least 0.005 nat on every role is a
   nonpromotive diagnostic.

Report exact private rank vectors and total deployed ranks `q0+r_j`, residual spectra,
shared/global allocation cutoff gaps,
per-site private boundary gaps, float32 projector/coefficient-map hashes, map/table/full
prices, dense multiplies, role CE partitions, model/callback ledgers, model-state hashes,
resource ceilings, and the three endpoint controls. Raw factor tensors and their
column-wise hashes are forbidden from result serialization.

Projector/direction identification is licensed only when every relevant reported
shared, private-boundary, and global-allocation cutoff eigengap is strictly positive.
A zero, missing, or unresolved relevant gap permits a literal compression/CE claim but
no scientific identification or naming of the selected directions.

Covered CE identity is evaluated within each fixed role across all seven arms at
`1e-6`; roles are never pooled. The three actual known-answer endpoints (global-budget
q0=0, global-budget q0=512, and typed-budget q0=0) must reproduce their pinned parent
arms within 0.002 nat. The independent-budget q0=0 comparator is excluded for the
nonuniform-versus-uniform reason above. Raw numerical predicates are named
`*_ce_qualifies`; every published `*_pass` is the conjunction of that predicate with
the complete integrity control. Thus neither a failed CE predicate nor a failed
integrity control can be laundered into a pass.

## Lifecycle

Use a new `hierarchical_shared_private_rrr_real_v1_*` namespace and inode/nonce lock.
Required order is pristine namespace -> committed/pushed source and pinned-input checks
-> checkpoint metadata/hash -> create-only authority -> fit-row/model load and native
capture -> freeze all seven programs -> discovery-row load -> native and compiled
scores -> hook removal/model equality/resource checks -> semantic replay -> create-only
result -> complete frozen-input replay -> last-written receipt. A post-authority error
writes one create-only failure only when neither receipt nor failure exists. No partial
result is authoritative; no program bundle or factor tensor is published.

Both terminal frozen-input replays (immediately before result publication and again
before receipt publication) must re-read/hash/re-read all three parent JSON artifacts,
run the exact inherited v2 result and receipt semantic validators, and require the
parent failure path to remain absent. A late parent failure therefore prevents result
or receipt authority rather than being omitted from the inherited input hash map.
