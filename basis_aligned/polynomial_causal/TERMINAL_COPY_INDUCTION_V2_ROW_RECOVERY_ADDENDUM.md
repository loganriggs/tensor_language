# Terminal copy/induction v2 row-recovery addendum

The v1 row transaction terminated before importing or loading the model because a
prior registry authority refers to a row tensor that does not exist.  V1 published a
terminal failure and is never retried.

V2 preserves every scientific choice in the v1 preregistration and screening
amendment.  It changes only the recursive registry census.  A missing row-like tensor
may be omitted if and only if all of the following are true before and during replay:

1. the reference occurs in the authority that names that exact tensor as its `rows`
   output;
2. the same authority names an existing terminal failure artifact;
3. that failure has the matching schema, status `terminal_failure_no_receipt`, and
   literal false values for `rows_exists`, `manifest_exists`, and `receipt_exists`;
4. the named rows, manifest, and receipt are in fact absent;
5. the exact authority and failure bytes are recorded in the v2 receipt; and
6. every other row reference in that authority and every other registry file is
   processed by the original strict loader.

Any mismatch spends the v2 namespace with no receipt.  The v1 failure, the failed
authority, this addendum, the recovery implementation, and adversarial tests belong to
the source closure.  The recovery is outcome blind and cannot import a model or read a
scientific result.
