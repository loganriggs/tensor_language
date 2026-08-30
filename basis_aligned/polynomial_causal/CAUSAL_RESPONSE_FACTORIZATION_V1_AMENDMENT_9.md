# Causal-response factorization v1 — prospective amendment 9

Status: frozen after the independent audit of source commit `f41ae53c` returned
NO-GO and before any production training snapshot exists. It authorizes no FIT tensor
access, model access, validation, fitting, selection, or EVAL.

## Receipt publication is not authority by itself

Amendment 8 correctly changed the object of the claim from mutable top-level paths to
a terminal-local historical snapshot. The next audit confirmed that later top-level
divergence is harmless and that create-only rename, post-terminal cleanup, and direct
alias protections hold. It then reproduced staged-file mutation, substituted snapshot
records, an unrecorded extra staging file, a load-boundary alias swap, and the absence
of a source-closed downstream consumer.

No ordinary writable filesystem can prevent the owning user from changing a file
after its last pre-publication read. Therefore the receipt is a commitment, not a
self-authenticating capability. Scientific authority exists only when the exact bytes
are revalidated at use time and copied into the consumer's returned training value.

## Final publisher gate

Immediately before `renameat2(RENAME_NOREPLACE)`, the publisher now independently:

1. requires the directory census to equal the four recorded snapshot members plus
   exactly one receipt/failure and its terminal hardlink;
2. verifies receipt and terminal are the same regular-file inode and match the staged
   terminal digest;
3. rejects malformed or substituted snapshot records; and
4. rehashes and restats every recorded member against its byte count and SHA-256.

This catches the reproduced pre-rename mutations and extra files. The successful
rename remains the final filesystem operation.

## Source-closed use-time consumer

The sole public consumer is the no-argument
`load_production_training_snapshot()`. It has no caller path, authority, validation,
or EVAL surface. Before returning cloned training tensors it verifies:

- receipt/failure exclusivity and receipt/terminal inode identity;
- the exact directory census and all four snapshot records;
- every member's regular-file status, stable bytes, size, and SHA-256;
- authority logical and artifact identities;
- the independent GO audit and its audit hash;
- the immutable historical git source closure and published ancestry;
- the manifest/authority/input/receipt join;
- the sanitized training payload's tensor hashes, owner topology, exact FIT-parent
  binding, production shape, 229-document role, and absence of validation/EVAL; and
- a second complete census/hash pass after semantic replay.

The returned `FitTrainingInput` owns cloned CPU tensors. Later filesystem mutation
cannot change it. Any altered published snapshot is non-authorizing and fails closed.

## Load-boundary physical alias check

The FIT loader now repeats all-pairs resolved-path and device/inode comparisons after
its one-use capability is poisoned, both before parent replay and immediately before
bundle access. A synthetic path that was absent at construction and is later hardlinked
to any production role is rejected. Cross-role and mixed aliases remain forbidden.

All changes require a fresh independent exact-source GO.
