# Causal-response factorization v1 — prospective amendment 8

Status: frozen after the independent audit of source commit `cdb7b30f` returned
NO-GO and before production training deserialization. This amendment authorizes no
FIT tensor access, validation, fitting, selection, or EVAL.

## Why another read sweep is not the answer

The audit passed 78 ordinary tests and then reproduced seven adversarial cases. A
repeated mutation schedule can make any fixed number of sequential whole-state reads
agree with an old value while the live path changes between reads. Therefore adding a
third or fourth sweep would not establish an atomic snapshot. The failed invariant
was mathematically impossible for independent mutable files.

## The terminal is now a content-addressed historical snapshot

The transaction no longer claims that every mutable canonical path has one
simultaneous value at the terminal-install instant. Instead, its private staging
directory receives exact copies of:

- the independently audited analysis authority;
- the independent GO audit;
- the sanitized 229-document training input; and
- its manifest.

Each source is opened without following a final symlink, checked as a stable regular
file, copied into a private terminal staging directory, fsynced, rehashed, and recorded
by relative name, byte count, and SHA-256 in the receipt. The receipt and terminal are
then same-inode hardlinks in that directory. Semantic replay operates only on these
copies. `renameat2(RENAME_NOREPLACE)` atomically installs the complete directory.

The terminal-local copies are the authoritative candidate-fitting parent. Future
consumers must read only those relative paths, verify the receipt hardlink and every
recorded digest, and reject any modified copy. Top-level authority/input/manifest
paths are construction artifacts, not the post-terminal source of truth. A later
change to a top-level path therefore cannot stale a receipt: it differs from, rather
than changes, the historical snapshot the receipt names.

This resolves the sequential-snapshot impossibility without pretending that more
retries create atomicity. It also makes the sanitized artifact self-contained after
the FIT parent has been validated and reduced; candidate fitting need not reopen the
55.5 MB FIT bundle.

## Remaining reproduced boundary repairs

1. Failure terminals use the same content-addressed historical-snapshot rule for
   whichever construction artifacts exist. They do not claim to describe later live
   path contents.
2. Owner-lock unlink and descriptor-close failures are fully non-propagating after a
   terminal is visible. A stale advisory lock is an explicit recoverable operational
   condition, not a reversal of a committed scientific transaction.
3. Synthetic parent exclusion compares every candidate role against every production
   role. Existing targets are compared by device and inode as well as resolved path,
   rejecting hardlinks, symlinks, dot-dot aliases, cross-role substitutions, and mixed
   production/synthetic sets.

The create-only `renameat2` repair passed the independent empty and nonempty rival
directory attacks. It remains controlling. All Amendment 8 changes require another
fresh exact-source independent GO.
