# Sparse MLP1 factorial v2 lineage-only execution recovery

**Frozen before FIT, SELECT, FINAL, checkpoint, or model access.**

The v1 row freeze completed outcome-blind, but an unrelated concurrent commit landed
between its source audit and freezer snapshot.  The immutable artifacts therefore name
different commits:

- source audit: `15ed37b9fec29685a415c7b940e026011f0c20ef`;
- row receipt: `236ae134ce80c78144b9eae1420336be06399c83`.

Both commits resolve the same exact 21-file source map recorded in the receipt.  The v1
FIT runner has not run: its authority, bundle, result, receipt, failure, and lock are all
absent.  Launching it would fail before data access because its validator requires the
two commit labels themselves to be equal.

V2 changes only this lineage predicate.  It accepts the two frozen labels if and only
if both commits independently replay to the identical frozen 21-file hash map, the
receipt and audit bytes match their pinned SHA256 values, all v1 execution artifacts
remain absent, and a fresh independent audit covers this amendment, recovery wrapper,
and tests.  That recovery admission is added to protected input replay and the v2
authority/receipt/failure chain.

The following are unchanged: all 288 rows and their FIT/SELECT/FINAL permissions,
scored positions 64--255, model/checkpoint, sparse program topology, 512 atoms, TopK
32 with ReLU, intercept, three seeds, Adam settings, selection variable, convergence
descriptors, CE admission threshold, price, and the sealed FINAL factorial.  The v1
runner remains the scientific executable; v2 only configures a new output/lock
namespace and the source-equivalent lineage proof.

No v2 execution is licensed until the recovery source audit is GO.  No FINAL tensor may
be opened by FIT/SELECT recovery.
