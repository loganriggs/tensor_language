# Causal-response factorization v1 — prospective amendment 10

Status: frozen after the independent audit of source `eade9893` returned NO-GO and
before any production training snapshot exists. It authorizes no FIT tensor access,
model access, validation, candidate fitting, selection, or EVAL.

## Reproduced defects

The Amendment-9 publisher's final census and byte replay passed independent mutation,
record-substitution, and extra-file tests. The no-argument consumer also returned
independent cloned tensors. Two authorization defects remained.

First, a synthetic bundle path could change to the inode of a production role after
the final pathname comparison but before the stable open. Second, a completely
self-consistent substitute terminal could appoint its own GO audit and change
authority semantics because the consumer checked internal hashes but no external
authorization root.

## Opened-inode exclusion

For every bundle, manifest, and receipt stable read, the loader now compares the
device/inode pair returned by the file descriptor that supplied the actual bytes
against every production parent role. A synthetic load fails before deserialization
when the opened inode is protected. The manifest summary is derived from the already-
opened validated payload, removing a second bundle pathname read entirely.

Thus there is no check-then-open gap: the object checked is the object read.

## External authorization root and exact semantics

The production consumer now requires the terminal-local audit bytes to be identical
to the canonical independent-audit artifact outside the terminal directory. That
canonical artifact must itself be byte-identical to the blob at the current Git HEAD,
and HEAD must be published ancestry of `origin/main`.

Internal content-addressing is then joined to this external root with exact schemas:

- the independent audit must be an exact, source-bound GO with no remaining blocker;
- the authority must have the exact training-only protocol, output namespace, outcome
  boundary, and false validation/EVAL flags;
- its audit binding must name and hash the canonical artifact;
- the manifest and terminal payload must have exact key sets, statuses, document
  count, protocol, and authorization flags; and
- the returned artifact binding and all three parent-binding fields must equal a fresh
  outcome-blind replay of the canonical completed FIT parent.

A substituted snapshot can no longer authorize itself merely by making its own bytes
mutually consistent. Any changed audit, role, protocol, or FIT parent fails closed.

The source requires another independent exact-commit audit. A local passing suite is
necessary but does not authorize production execution.
