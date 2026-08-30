# Causal-response factorization v1 — prospective Amendment 6

Status: frozen after independent lifecycle audit `2b9a2bd3` returned NO-GO and before
any factor-training response value is deserialized.  The canonical NO-GO remains
immutable.  This amendment authorizes no execution; the repaired exact source closure
requires a fresh independent GO.

## Reason

The audit passed 67 ordinary source-isolated tests but reproduced four adversarial
failures and found an incomplete runtime closure:

1. input or source drift after manifest publication could still produce success;
2. failure of the second success hardlink could leave a terminal-only state;
3. a failure after authority mutation recorded a stale authority digest;
4. the public loader accepted a caller-supplied self-consistent authority object;
5. the model facade and two `jacclust` runtime files were absent from the closure.

These are transaction-integrity failures.  They do not alter the factor family,
document split, seeds, ranks, optimizer, prices, or validation gates.

## Controlling repair

### Exact protected-state replay

Immediately before a success terminal is made visible, the lifecycle must replay:

- the exact published authority artifact and logical identity;
- the exact independent GO artifact;
- every live source byte against the independently audited commit;
- the complete outcome-blind FIT parent binding;
- the semantically replayed training input and its exact artifact digest;
- the manifest and its exact artifact digest;
- the original open lock descriptor, device, inode, and nonce.

No callback or artifact lookup may occur between the end of this replay and the single
terminal publication operation.

### Atomic terminal pair

Receipt and failure are mutually exclusive terminal kinds.  The selected payload and
`terminal.json` are prepared as same-inode hardlinks inside a private staging
directory.  One same-filesystem directory rename publishes the complete pair.  Thus
there is no externally visible state containing only one hardlink.  A failure before
the rename publishes neither; the exception path may then compete for the same empty
terminal directory with a complete failure pair.

### Failure provenance

The failure artifact records both the originally attempted authority digest and a
bytes-only protected-state observation taken after the triggering error.  That
observation includes authority, audit, input, manifest, every declared source, and all
FIT-parent artifact paths.  Its final guard recomputes the complete observation.  A
mutated authority is therefore recorded by its current bytes rather than mislabeled by
the stale attempted digest.

### Production loader authority

Production loading no longer accepts an authority mapping from its caller.  It reads
only the canonical authority path, requires the caller's expected artifact digest,
then independently replays:

- the canonical independent GO;
- equality of audit and authority source closures;
- every live and historical source hash;
- published-ancestry membership;
- the exact canonical output namespace;
- the canonical outcome-blind FIT parent.

The synthetic authority-mapping surface is allowed only with nonproduction FIT paths;
the exact production paths are rejected in synthetic mode.

### Source closure

Add the transitive runtime dependencies:

- `basis_aligned/polynomial_causal/bilin18_observed_model_facade.py`;
- `jacclust/__init__.py`;
- `jacclust/tt_model.py`.

## Required re-audit

The fresh audit must run the full source-isolated suite and independently reproduce the
four former attacks.  It must also attempt synthetic-mode use of exact production FIT
paths and caller-supplied production authority.  Only an exact GO with
`outcome_access=false` and zero blockers may authorize the no-argument lifecycle.
