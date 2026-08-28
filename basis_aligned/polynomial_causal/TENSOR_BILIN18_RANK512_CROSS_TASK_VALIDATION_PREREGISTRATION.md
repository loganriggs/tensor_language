# Preregistration: rank512 cross-task row and fresh-fixture validation

Date: 2026-08-28

Status: validation only. The two FineWeb roles were frozen for an independent early-MLP
compiler but were not used to fit, select, or gate the shared-attention ranks. They are
therefore cross-task heldout for this attention program, not globally untouched data.

## Roles and fixture

- 192 FineWeb rows at skip31000;
- 192 FineWeb rows at skip35000;
- a new deterministic 4 by 256 token fixture, with the registered prefix change at
  position 32 and downstream scoring from 33.

Serialized and raw-tensor hashes must match the prospective canonical row registry.
The rank512 program is refitted only on the original 480-row skip80 fit role.

## Gates

1. Complete price and ownership replay exactly: 503,436,726 values, total support, zero
   native calls/references/fitted tables, disjoint storage, and checkpoint collection.
2. All-position and covered CE harm are each at most 0.025 nat on both validation roles;
   unseen-current harm is at most 0.03 nat.
3. Neither all-position nor covered harm exceeds the worst opened-role rank512 harm by
   more than 0.01 nat.
4. On the fresh deterministic fixture, context-delta recovery is at least 0.90, cosine
   at least 0.95, and the compressed downstream effect is nonzero.
5. Covered harm replicates between validation roles within 0.01 nat.

Passing licenses cross-task heldout validation of the current rank512 point. It is not
distribution-shift OOD, circuit semantics, or consequence validation for extraction and
editing.
