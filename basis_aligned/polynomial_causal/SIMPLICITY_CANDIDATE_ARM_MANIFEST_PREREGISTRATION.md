# Rung 441 preregistration: hand-reviewed candidate-arm manifest

Status: registered before generating the manifest. CPU-only archive repair; no predictor fitting, model forward, or
GPU queue entry is authorized.

## Why this rung exists

Rung440 found130 clean experiment receipts but only16.9% explicit arm maps. One receipt is not one program. This rung
constructs a conservative manifest in which every row is one specific executable or screened candidate arm, its
complete available structural price is explicit, and duplicate appearances of the same program across follow-up
receipts are merged rather than counted as independent samples.

## Frozen source families

Only these manually reviewed families may enter generation1:

1. vocabulary shared/independent/hybrid programs from rungs300,304,305;
2. mixed104 MLP-PCA pair/rank/gradient programs from rungs311–313;
3. mixed104 MLP0 context-metric input programs from rungs325–329;
4. attention0 sparse Q/K programs from rungs426 and430.

Rung312 rank256 and rung313 `pca256` are the same p8+17 rank256 program already measured in rung311; they merge into
one candidate. Control arms are retained and explicitly typed as controls. Rung426 P54 has no downstream consequence
measurement and is excluded with that reason. No other receipt may be added after seeing the coverage result.

## Physical separation and common schema

The structural file contains candidate ID, family, grammar, control type, source receipt hashes, complete available
scalar/byte price, rank/atom/sparse-row counts, and declared sharing/locality fields. It contains no outcome value.

The consequence file contains the candidate ID and available values in this fixed schema:

- `validation_ce_damage` and `ood_ce_damage` (CE added above native; lower is better);
- `census_ce_damage` and `certificates_valid`;
- `intervention_effect_cosine`, `intervention_normalized_error`, and `intervention_collateral_spearman`;
- `composition_ratio` or `composition_prediction_error`;
- `local_relative_squared_error`, attention-write error, and downstream-reader error as auxiliary diagnostics.

Missing measurements are null, never imputed. A SHA-256 join binds the two files. The structural file is written and
hashed before consequences are read.

## Frozen bars

- A: at least40 unique candidate arms across all four source families; every row has source hashes and positive
  scalar or byte price; no forbidden outcome key occurs in the structural file; duplicate candidate IDs are zero.
- B: OOD CE has at least25 candidates across at least two source families.
- C: extraction/certificates has at least10 candidates across at least two source families.
- D: removal/intervention and composition each have at least10 candidates across at least two source families.

A historical predictor for a named consequence is licensed only if that consequence has at least20 candidates in
at least three whole program families; this stricter fitting license is reported separately from A–D. A family that
generated several ranks counts once for held-family coverage.

Strong null: fewer than30 priced unique candidates or no consequence has at least20 candidates across two families.
Under it, stop historical repair and build a prospective candidate bank. Otherwise preserve the usable consequence
slices and prospectively generate only the missing families; do not fit any consequence that lacks the three-family
license.

Literal price: archive-only CPU work, zero deployed values and zero native calls. Claim level: dataset instrument.
