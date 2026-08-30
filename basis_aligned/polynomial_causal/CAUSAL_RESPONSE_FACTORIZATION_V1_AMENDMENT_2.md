# Causal-response factorization v1 — amendment 2

Status: controlling and frozen before any FIT response bundle is deserialized. This
amendment repairs the independent adapter audit's four NO-GO findings. It does not
authorize analysis, validation, EVAL, or a scientific claim.

## Why this amendment is needed

The first adapter correctly derived the signed response but accepted only an
in-memory payload and logical authority hash. Its output exposed both the 229 training
and 114 validation documents, retained no exact bundle/manifest/receipt identities,
and allowed a caller to construct a formally valid dataclass with a false owner-group
assignment. Those are scientific protocol defects even though all numerical tests
passed.

## Controlling role boundary

Factor fitting receives only the 229 training documents. The training adapter has no
factory, field, or method that exposes the 114 validation documents. It returns

$$
R^{\mathrm{train}}_{pstd}
=R_{pst,\,\pi(d)},\qquad d=1,\ldots,229,
$$

where $\pi$ is the preregistered SHA-256 document ordering. It also retains each
training document's original FIT-axis index. The full response and validation indices
are temporary private values inside the adapter and no caller alias survives.

A later validation source closure may expose only the 114 validation documents after
the candidate library, seeds, optimizer health decisions, and prices have been frozen
create-only. No validation adapter exists in the training source closure.

## Exact artifact provenance

Every training input carries an immutable `FitArtifactBinding` containing:

- outcome-blind parent-binding SHA-256;
- receipt and shared-terminal SHA-256, which must be identical;
- authority artifact and logical SHA-256;
- bundle artifact SHA-256;
- manifest artifact and logical SHA-256; and
- historical FIT source-closure SHA-256.

Before tensor deserialization, the outcome-blind parent binder must establish all of
these identities. It imports no `torch`, returns no artifact bytes, and requires:

1. failure terminal and live owner lock absent, both before and after the read;
2. receipt and shared terminal byte-identical and the same inode;
3. current authority, bundle, and manifest records exactly equal the receipt aggregate,
   including path, presence, digest, byte count, device, inode, and timestamps;
4. authority schema and logical identity, exact production protocol, namespace, and
   non-EVAL role;
5. the historical 21-file source closure replayed from the bound published commit;
6. the exact independent source-bound GO audit and frozen parent hashes;
7. manifest logical identity and exact authority/bundle/protocol joins; and
8. receipt status, model-state equality, 12,400 forwards, and event shapes.

The binder treats the bundle as opaque bytes. After a separate factor-analysis
authority is frozen, a one-use loader must stable-read those exact bytes, deserialize
with `weights_only=True`, replay full production bundle semantics and manifest summary,
join receipt checkpoint/model state/ledger fields, pass the private payload once to
the training adapter, destroy its alias, and poison the loading capability.

## Owner topology is an invariant, not caller metadata

Let `owner_components` be the ordered first occurrence of components in the sealed
49-source list. For every source $s$,

$$
g_s=\operatorname{index}\!\left(
\texttt{source\_components}_s,\texttt{owner\_components}
\right).
$$

`FitTrainingInput` recomputes this vector and rejects any supplied assignment that is
not exactly equal. Owner order must be canonical and unique; source and target tags
must be the same sealed order. Thus an optimizer cannot silently turn a
shared-plus-private model into an all-global or differently grouped model by forging
metadata.

## Gate

The original independent audit remains an immutable NO-GO. Analysis remains forbidden
until an independent audit of the amended source closure returns GO. A passing toy
suite is necessary but not sufficient. Validation outcomes and EVAL remain unopened.
