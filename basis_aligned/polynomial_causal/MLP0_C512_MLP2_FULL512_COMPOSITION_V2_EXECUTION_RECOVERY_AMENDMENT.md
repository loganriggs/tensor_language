# C512 × FULL512 composition v2 execution-recovery amendment

The first execution attempt failed before authority creation, row opening, model
loading, or artifact publication because a concurrent unrelated commit moved local
`HEAD` ahead of `origin/main`. Its terminal failure SHA-256 is
`6c375e461def332a38584e11a542ab4ec3c6822e6a385a31c46d8e51c98e42e1` and records
`evaluation_may_have_opened=false`, `authority_exists=false`, and no artifacts.

The recovery changes only the create-only execution output/lock namespace and source
admission rule. Scientific source bytes remain the exact 27-file family frozen in
the V2 row receipt and audited at commit `56db10ad`. Branch-pointer movement is
ignored when those exact bytes, row receipt, programs, checkpoint, audit, and failure
lineage remain unchanged.

The recovery wrapper and this amendment receive a separate outcome-blind audit. Its
hash and source binding are injected into the execution authority and every protected
snapshot. Arms, rows, metrics, bootstrap, gates, and physical programs are unchanged.

