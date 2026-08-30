# Causal-response factorization v1 — Amendment 13: stable audited-source identity

Status: prospective after the first production launch was interrupted and before any
terminal grid exists. Six completed rank/seed cells from that launch are preserved in
`causal_response_factorization_v1_grid_results_interrupted_moving_head_20260830T1256Z`.
They are real FIT-only training outcomes but are not a complete grid, candidate
selection, validation result, EVAL result, or circuit claim.

## Preserved failure

The first audited runner bound its logical source closure to the repository's current
`HEAD`. The production tree has multiple agents committing unrelated experiments.
Although all 40 audited source files remained byte-identical, one unrelated commit
changed `HEAD` during fitting. The runner's final replay would therefore reject its own
source closure, and a later invocation could not resume the completed cells. The run
was interrupted after six cells rather than spending the remaining GPU hours on a
known nonterminal transaction.

The archived cells are the three seeds at global ranks one and two. All six are healthy:
rank one final training MSE is approximately 0.04155324–0.04155326 and rank two is
approximately 0.03974880–0.03975012. They remain explicitly partial and cannot enter a
frontier or seed comparison beyond these two completed rank points.

## Stable identity

Production source identity is now the tuple of:

1. the exact independent audit's `audited_source_commit`;
2. the SHA-256 of the exact independent audit artifact; and
3. the exact path-to-SHA-256 map of every audited runtime, test, protocol, and price
   source plus the audit artifact itself.

At preflight and final replay, current working bytes must match this frozen map. The
audited commit must be published ancestry, every source hash must replay from that
commit, and the audit artifact must byte-match its blob on `origin/main`. Unrelated
current or future `HEAD` commits are deliberately absent from the logical identity.
Thus source drift still fails closed, while unrelated published work cannot invalidate
an in-flight four-hour experiment.

The archived namespace is not resumed under the new identity. A fresh canonical
namespace and a fresh independent audit are required before relaunch.
