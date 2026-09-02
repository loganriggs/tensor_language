# Rung454 preregistration: complete vocabulary-program teaching consequences

Status: registered after MLP0 and MLP-PCA count as teaching families1–2, before any vocabulary candidate consequence
is computed on TEACHING. GPU execution must use the managed queue; SEALED_CONFIRMATION is forbidden.

## Candidate family and exact computation

Rebuild all23 rung445 vocabulary programs from the frozen r300/r304/r305 sources and fit caches:

- uniform shared-code and matched independent output maps at labels0,128,256,512;
- count-weighted rank512 shared maps with1,129 exact Fisher-, row-norm-, or seeded-random rare-token corrections;
- square-root-count and count-weighted shared/matched-independent maps at ranks512,640,768.

Every program keeps the50,304×1,152 input embedding exact and changes only the output vocabulary map. Therefore capture
the exact final hidden states on the same96 TEACHING documents under native, original-native attention16 mean knockout,
and the physical14,984-value MLP16 partner once each. Apply each factorized output program to those states, the native
`30*tanh(logits/30)` soft cap, and exact full-vocabulary cross entropy. This is algebraically the same candidate model
as23×3 transformer runs because no candidate changes any upstream hidden state.

Use the registered normalized removal and composition errors and fixed48-document waves. Before outcomes, freeze14
adjacent-rank directions: three uniform edges for each of shared/independent, and two weighted edges for each combination
of two weighting rules and shared/independent. Each predicts that higher rank has lower consequence error. Sparse-row
variants are measured but have no invented total order.

## Predictions, bars, and null

- **A — exact/live instrument:** all hashes,23 IDs/prices, output-only scope, matrix dimensions, deterministic sparse
  selections of1,129 rows, native hidden/CE replay,32,768 attention-mean positions,24 knockout calls,24 partner calls,
  and SEALED closure hold.
- **B — structured rank ordering:** at least`.85` of the14 adjacent-rank directions hold for both removal and composition.
- **C — label variation:** the23-arm spans are at least`.05` for both normalized errors.
- **D — document reliability:** each48-document wave preserves at least`.70` of the14 directions for both metrics, and
  the continuous23-arm wave-to-wave Pearson correlation is at least`.70` for both.

**Strong null:** A or any native/candidate/partner liveness clause fails; both spans are below`.01`; either full structured
accuracy is below`.50`; or either wave correlation is negative. Only A/B/C/D with no null counts vocabulary as teaching
family3/3 and licenses consequence-predictor fitting. Failure requires a new outcome-free teaching family; it does not
permit fitting on only two families. This run does not fit a predictor, open attention0, establish semantics/OOD, deploy
a vocabulary program, or add deployed values.
