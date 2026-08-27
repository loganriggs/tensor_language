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

`freeze_code_oracle_corpus.py` constructs an immutable local corpus from Python
files tracked at one git commit. Files are read from git objects, not the dirty
worktree, in lexicographic path order and tokenized with GPT-2 BPE. It saves 480
rows of 257 tokens:

- basis: rows `[0:96]`;
- discovery: rows `[96:288]`;
- held-out: rows `[288:480]`.

The manifest records the source commit, every contributing file and SHA256, the
tokenizer, split boundaries, tensor SHA256, and construction script SHA256. This
avoids redefining the OOD corpus as the repository changes.

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
basis, generate 20 seeded Haar 64D nulls inside the site's top-256 support and
scale their basis-set correction RMS to that content arm. There is no learned
decoder or CE optimization.

Report paired 2,000-draw row-bootstrap intervals for global CE, code-token strata,
correction RMS, and fraction of the full-oracle gain. Negative effects are not
clipped. Discovery and held-out use the same frozen token strata.

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

No learned correction runs until this classification is stable on held-out code.
Any later predictor must additionally transfer across an alternate ship background
and preserve the output/intervention families relevant to its target cell.
