# Rung 586 preregistration: clean replication of R580 induction native capability

**Frozen:** 2026-09-03 UTC, after R581 exposed the R580 `next_step` serialization defect and before any R586 model output exists

## Purpose

R586 is a prospective clean rerun of R580 in a new result and receipt namespace. It does not edit, reinterpret, or replace the immutable R580/R581 artifacts. It preserves R580's rows, prompt census, native-logit computation, batching, raw evidence, bootstrap, scientific gates, thresholds, split policy, and terminal scientific decision exactly. Its only intended behavior change is that held and null `next_step` values must serialize as scalar JSON strings. Every envelope field is type-checked before a scientific artifact can be written.

R586 may open exactly FIT and SELECT. FINAL_TEST and OOD remain closed. It evaluates 3,024 unique prompts once in batches of 32, hence exactly 95 model forwards, zero backwards, and zero weight updates. A failed scientific gate writes a complete scientific null; an authority, type, census, split, or price mismatch is an integrity error.

## Frozen scientific contract

The complete scientific contract is R580's preregistration, SHA-256 `8f80926d0a90360a66ebce605732d32ff3e283a3428eb7245f4813a521d12580`. In particular, R586 retains all five gates in both FIT and SELECT without changing a threshold:

1. every selector-by-payload factorial cell has positive-margin fraction at least 75% and group-bootstrap lower 95% bound above zero;
2. selector-by-payload interaction has lower bound above zero;
3. both endpoints of every neutral-source, neutral-payload, filler, and lag cell have positive-margin fraction at least 75% and lower bound above zero;
4. selected-match breaks have positive-drop fraction at least 70% and lower mean-drop bound above zero; and
5. selected-versus-neutral paired gaps have lower bound above zero.

The same 2,000-replicate SHA-256 group bootstrap, quantile conventions, non-gated contrast-source reporting, three runner predicates, and scientific-null behavior apply. The model computation remains the exact clipped native forward used by R580.

## Frozen authority

R578:

- rows: `8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6`
- receipt: `9e4e63ebd98503d6aa5daa27617a20fea595829c5a372f27b1ce4371d7c05b45`
- preregistration: `276d801bbf5795e6421488dd4971b3a2d2dcb56e4fc7c4bc7ecdd2f61a73e9ce`
- builder: `d47bb3d46bd2c6061132c13b356e58ba9dfe2a56a2629f8b49a03f280d290bbd`
- focused test: `9d795df358dfef9c5d17a539307f8e781f2a4debeb4909078858a242b3dfc512`

R580:

- preregistration: `8f80926d0a90360a66ebce605732d32ff3e283a3428eb7245f4813a521d12580`
- implementation: `62d11395d845d663257433936773780dd4bb9ddbcb9286400c420dadd3a73249`
- focused test: `9f166a61409c12d6a4a58e16640af654378151f99c05597f9c63dbb2dec64550`
- dry run: `3d21b62972aa0794598860228554068035af10fd743e8958bfc7a05d56d68588`
- scientific result: `7c7463a95931a51cd848ff9e8033bed77a26f7889a1a5fd1a3512ec2d1224b84`
- result receipt: `6a1ef728bca424ed27ec145adad1918923e91f190b96a9ff452b6838413b670a`

R581:

- preregistration: `d2989383791cb179fecfa930742812cf8036a85bb9d2f3cfdd6555bb00640887`
- implementation: `812c28bd1987d0978cbf0c2b0d09f0669b159b515b8a4b3f8db5dd1a73663841`
- focused test: `70782c35c4aac7089d363360de3f0365dfa19bdd9947c90ddc73ad3f096f1e93`
- dry run: `c6a8bb32ec0bfae17257507682b2b942be6cdf645afa93736e135a53174e14ea`
- audit result: `8ecc1562632212ee876a794377e31966776ec15de02b5cb8d31798e438502cdb`

Checkpoint weights SHA-256 remains `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.

Shared result-contract authority, incorporated before any R586 model output:

- helper: `af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272`
- focused test: `2f26e3125e1208b9b7e9f1b138cfc90921157143303f098f853d3f65432f0645`
- usage contract: `4b2ed9bc32ed5cd5e4151bc39d3a7a6a83fa8498a97b7ff1e928a82d6c8ac304`

R586 must execute this generic contract in addition to its rung-specific checks. It must validate literal finite JSON, exact R578 FIT/SELECT row and group membership, declared field types including scalar-string `next_step`, exact split closure, the 95-forward/zero-backward/no-update envelope using field `model_weights_updated`, and all required provenance hashes.

## New namespace and integrity rule

The only allowed scientific outputs are:

- `basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_results.json`
- `basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_receipt.json`

The model-free dry run has its own R586 path. It must exercise held and scientific-null fixtures, exact R580 score equality, the literal price, full envelope validation, rejection of tuple/list `next_step`, and immutability of all frozen R578/R580/R581 artifacts. The scientific runner must validate the complete result envelope before writing either new artifact. R587, authored independently after R586 hashes are frozen, will audit any eventual R586 result.
