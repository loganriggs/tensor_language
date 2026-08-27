# Conditional code-OOD live-ship oracle specification

## Status and trigger

This experiment is registered before the FineWeb oracle result is known. It must
run only for sites listed in `ship_content_oracle_screen_results.json` under
`training_license_sites`. A FineWeb failure does not license searching code for a
replacement target; it redirects the project to a non-content residual interface.

The goal is not merely to show that code differs from prose. That is already
known: the frozen prose basis captures `0.1659` of code contextual variation,
versus `0.5146` for code's own top-64 basis. The question is whether either
content coordinate system contains the part of the *missing deployed-ship
computation* that causally repairs code CE.

## Frozen corpus and splits

The original `code_oracle_corpus.pt` is retained as a failed v1 artifact and is
barred from scoring. A preregistration audit found that its sequential token
concatenation cuts both split boundaries inside a source file and makes rows cross
all 36 file boundaries. It therefore has neither independent file clusters nor
valid code transitions at those boundaries.

`freeze_code_oracle_corpus_v2.py` constructs the scored immutable corpus from
Python files tracked at the literal v1 source commit. Files are read from git
objects, not the dirty worktree, and assigned to splits by a deterministic hash of
their path. Each 257-token row is wholly contained in one git blob; incomplete
chunks are discarded and each file contributes at most four rows. It saves 480
rows with per-row `(path, token_start, token_end)` provenance:

- basis: rows `[0:96]`;
- discovery: rows `[96:288]`;
- held-out: rows `[288:480]`.

The manifest records the literal source commit, every contributing file and blob
SHA256, tokenizer version/fingerprint, split boundaries and file clusters, every
row's token offsets, tensor SHA256, and construction-script/spec hashes. File sets
are disjoint across splits and no row crosses a file boundary. This is a frozen
repository-Python register sample, not a claim about code in general.

## Frozen singleton interventions

For every FineWeb-licensed site `l`, reconstruct the identical complete ship and
compute its live missing residual

```text
e_l(z_l) = MLP_original,l(z_l) - MLP_plank,l(z_l).
```

On basis rows only, construct:

1. the already frozen QR-orthonormalized prose content basis;
2. a code content basis using the identical token-conditional deviation recipe at
   residual layers 8, 10, and 12;
3. the site's top-64 through-origin code residual PCs;
4. a top-256 code residual support for matched nulls.

Evaluate singleton injections of full `e_l`, its prose-content projection, its
code-content projection, and its local residual-PCA projection. For each content
basis, reuse the same 20 seeded Haar 64D directions inside the site's top-256
support and scale their basis-set correction RMS separately to that content arm.
There is no learned decoder or CE optimization. With exactly 20 nulls, a content
arm passes the one-sided 5% Monte Carlo gate only by beating all 20; its exact
empirical p-value is `(1 + #null >= content) / 21`.

Report paired 2,000-draw file-cluster-bootstrap intervals for global CE,
code-token strata, correction RMS, content-minus-prose gain, and fraction of the
full-oracle gain. Row-bootstrap values may be secondary diagnostics only. Negative
effects are not clipped. Discovery and held-out use the same frozen token strata.

The code evaluation must load the exact serialized ship realization used by the
authoritative FineWeb oracle, including `TWALL`, every derived `SHIP` object, glue,
configuration hashes, and a baseline fingerprint. Reconstructing randomized
low-rank fits is not identical and is forbidden. The code run must make no network
request and must pass the baseline fingerprint before any intervention.

## Registered decisions

A site licenses a learned code residual predictor only if:

1. the full-oracle held-out CE gain has 95% bootstrap lower bound above zero;
2. at least one content arm improves both discovery and held-out CE;
3. that arm's held-out gain exceeds its matched-null 95th percentile; and
4. its held-out gain is at least `max(0.02 nats, 0.10 * full-oracle gain)`.

Classify the interface without moving thresholds:

- **shared prose coordinate:** prose passes all gates and is no more than `0.02`
  nats worse than code-content;
- **domain-typed coordinate:** code-content passes and beats prose by at least
  `0.02` nats;
- **non-content residual:** the full or local-PCA oracle passes but neither content
  arm does;
- **compensatory-only site:** even the full original residual fails.

The labels above leave one threshold edge case implicit: a content arm can pass
its own causal/null gates while the prose-versus-code difference is inside the
`0.02` boundary but the other arm narrowly fails. Before execution, freeze that
case as **inconclusive content coordinate** and license no learned predictor. This
is fail-closed rather than silently rounding the result into shared or typed.

No learned correction runs until this classification is stable on held-out code.
Any later predictor must additionally transfer across an alternate ship background
and preserve the output/intervention families relevant to its target cell.
