# C512 × FULL512 composition v2 row-recovery amendment

The audited v1 row freezer failed before selecting, caching, or publishing any row.
Its terminal failure SHA-256 is
`0c760bbd6798960eb037dbdd01e820fa1a924c98aff6e2b645c01964592055c3` and records
`cache_exists=false` and `receipt_exists=false`.

The cause was mechanical: a prior successful fresh-row receipt preserved two exact
failed-row waiver proof records, and the recursive registry scanner mistook the
proofs' documented absent path for a new live row reference.

V2 changes only the row lifecycle:

1. validate an embedded `waiver_proofs` family against the exact two canonical
   authority/failure lineages and registry hashes;
2. omit those validated proof metadata objects from recursive row-path discovery;
3. retain the original two direct registry waivers and every other registry file;
4. publish under a new create-only v2 cache, receipt, failure, and lock namespace.

The source-document start, role sizes, token length, model programs, four arms,
metrics, bootstrap, gates, and unopened evaluation contract are unchanged. The v1
failure remains preserved and cannot be overwritten.
