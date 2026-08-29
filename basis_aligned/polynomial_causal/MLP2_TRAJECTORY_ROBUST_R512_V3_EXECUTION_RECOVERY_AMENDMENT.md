# MLP2 trajectory-robust R512 V3 execution recovery amendment

The V2 fresh-row transaction completed outcome-blind row freezing, then failed before
authority publication, row loading, checkpoint loading, or model evaluation.  During
the short interval between the row receipt and evaluator admission, the shared
worktree advanced to local commit `43a6ff16e4cb4f779a0e8673345896c5c03b4516`
before that commit appeared on `origin/main`.  The prospective ancestry check rejected
that state.  V2 is spent and its failure is preserved exactly.

V3 changes only execution admission and terminal namespaces.  It reuses the exact V2
row receipt and row tensor, all V1 scientific arms/contrasts/thresholds, and all V2
transaction code.  Its source commit is read from a new independent outcome-blind
audit rather than from the mutable shared-worktree `HEAD`; the audited commit must be
an ancestor of `origin/main` and every source byte must match it.  This removes the
irrelevant race with concurrent commits without weakening source binding.

V3 may run only if the exact V2 failure says `authority_exists=false` and
`evaluation_may_have_opened=false`, V2 authority/ledger/result/receipt/lock remain
absent, the V2 row receipt and tensor retain their exact hashes, and the original V1
recovery admission still passes.  It uses distinct V3 authority, ledger, result,
receipt, failure, and lock paths.  No scientific gate or bootstrap rule changes.
