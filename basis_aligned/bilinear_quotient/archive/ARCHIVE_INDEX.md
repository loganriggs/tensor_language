# Archive

Artefacts moved out of the `bilinear_quotient` root because **nothing references them**. Nothing here is
deleted, everything is reversible by name, and every batch carries a manifest.

## The rule

An artefact is archivable only when **no** ledger section, board entry, backlog note, preregistration, script,
shell wrapper or queue file mentions it — and a file mentioning *itself* counts as no evidence at all. The
decision is made by `ops/repo_orphans.py`, which is read-only; the move is made by `ops/archive_orphans.py`,
which re-runs the scan at move time so a file that gained a citation since the last scan is spared. **Deciding
what is dead and acting on it are deliberately separate commands.**

## Two tiers

| tier | meaning |
|---|---|
| `dead/` | no reference anywhere, and `runlogs/runner.log` shows it never executed |
| `ran-but-uncited/` | the runner executed it, but nothing cites it or its results today |

The distinction is kept because "nothing cites it" and "it never ran" are different facts, and the second tier
is the one worth looking at again before anything is ever deleted.

## Archived files still count as references

`repo_orphans.py` includes `archive/**` in its citation corpus. Without that, archiving **cascades**: every
receipt whose only citation was an archived script becomes an orphan on the next scan, and the next, until the
root is empty. An artefact that an archived rung produced belongs with that rung, not on a second sweep. With
it, the scan reaches a fixed point — after the batch below, `repo_orphans.py` reports **0 orphans**.

## Batches

| date | files | size | note |
|---|---|---|---|
| `2026-09-04/` | **583** (577 dead, 6 ran-but-uncited) | 38.0 MB | First sweep. Root went from 4,644 to 4,063 files. Largest items were `*_invalid_*` bundles from superseded rungs. |

## Restoring

```
git -C /workspace/tensor_language mv \
  basis_aligned/bilinear_quotient/archive/<date>/<tier>/<name> \
  basis_aligned/bilinear_quotient/<name>
```

Each batch's `MANIFEST.json` lists every file moved, the rule applied, the tools used, and anything that failed.

## What was deliberately NOT archived

- **Everything the ledger cites.** `ops/audit_ledger_prices.py` resolves 107 `Results:` receipts; all still
  resolve after the sweep, and that check is the gate on any future batch.
- **`induction_centered_fixed_geometry_rung59{2,3}_invalid_evidence/`** (694 MB + 11 MB, untracked). Named
  "invalid" and uncited by any document, but **five of Codex's rung scripts reference those paths**, so moving
  them would break another agent's lane. Flagged on the board instead.
- **Tracked bulk that is live evidence**: `bundle_shards/` (129 MB), `mlp0_native_down_hierarchy_v1_programs/`
  (60 MB). Referenced, therefore kept.

The root is heavy — ~1.57 GB — but after this sweep it is heavy with **live** evidence: the orphan scan
accounts for 0 bytes of it.
