# Causal-response factorization v1 — prospective amendment 7

Status: frozen after the independent audit of source commit `89b65816` returned
NO-GO and before the production training input is deserialized. This amendment does
not authorize FIT tensor access, validation access, candidate fitting, or EVAL.

## Why the transaction remains closed

The independent audit at commit `367334c0` passed 72 ordinary source tests and then
reproduced seven adversarial failures. In particular, an ordinary `os.rename` could
replace an empty terminal directory, protected artifacts or the owner lock could
change in lookups after their purported final checks, a post-rename exception could
make the caller report failure while a success pair was visible, a linked authority
could fail replay before the outer transaction learned its identity, a failure could
bind a stale sequential observation, and dot-dot aliases could route production FIT
artifacts through the synthetic loader surface. `outcome_access` was false.

## Prospective transaction repair

The repaired source is governed by these rules:

1. The lifecycle owner lock is part of the exact protected aggregate. Immediately
   before terminal publication, the entire aggregate is observed twice and both
   observations must equal the predeclared snapshot, including the owner lock's
   device, inode, nonce hash, and byte count.
2. No independent claim check or terminal-path lookup follows that aggregate replay.
   The next filesystem operation is the serialization point.
3. The serialization point is Linux `renameat2(..., RENAME_NOREPLACE)`. It installs
   the already-replayed two-hardlink staging directory atomically and fails for every
   existing destination, including an empty directory.
4. Directory sync, semantic replay, callbacks, and staging cleanup all occur before
   the serialization point or only on a failed install. The successful install path
   performs no later filesystem operation. Owner-descriptor cleanup cannot propagate
   an error after a terminal is visible.
5. The attempted authority and its exact prospective artifact digest are retained
   before linking. Thus a post-link replay failure can still publish a raw-state-bound
   failure pair instead of leaving an unexplained authority-only state.
6. Success and failure terminal payloads bind the logical digest of their exact
   protected aggregate. A changed aggregate cannot publish a stale terminal.
7. Synthetic loader paths are compared by resolved physical target, role by role.
   Touching even one production parent artifact through a lexical, dot-dot, or symlink
   alias is forbidden.

These changes are prospective. They require a fresh exact-source independent GO.

## Separate candidate-control blocker found outcome-blind

The frozen preregistration promises a dense observation-by-code SVD control whose
persistent storage and per-document code are each no larger than the structured
candidate it controls. The observation dimension is

$$
2\times49\times49=4802.
$$

For every one of the 17 frozen structured rank pairs, persistent storage is below
4,802 values (the maximum is 3,200 for global rank 32). Therefore even dense SVD rank
one violates the persistent-storage coordinate, and the exact matched rank is zero
for all 17 candidates. This was established from dimensions and the frozen price
formula without reading response values.

Collapsing persistent storage and 229 per-document codes would admit dense ranks one
or two for a few points, but that changes the declared two-coordinate simplicity
order and is forbidden post hoc. Before any candidate fit, a later prospective
candidate-lifecycle amendment must either:

- retain the rank-zero result and add a genuinely admissible non-dense null, such as
  matched-price random/separable tensor atoms with the same code dimension; or
- explicitly introduce a third, separately reported amortized-total price without
  replacing the original persistent/code Pareto order.

No SVD outcome may be reported as “price matched” under the original rule unless this
mathematical infeasibility is stated.
