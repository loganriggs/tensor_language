# MLP9 full-width DCT unsupervised-reader test V1

## Question

The one-layer MLP17 certificate established that the analytic weight Hessian,
autodiff Hessian, and orthogonalized-ALS factors agree in a random
16-dimensional input / 8-dimensional output probe.  Test the DCT briefing's
stronger practical proposal at full width: can rank-eight factors recovered
from the raw MLP9 bilinear weight tensor, with no behavior or prompt access,
supply the downstream reader used by the CrossFirst spelling behavior?

For `MLP9(x) = D[(Lx) * (Rx)] + b`, use the implicit symmetric Hessian

`T(u,v,w) = u^T D[(Lv)*(Rw) + (Lw)*(Rv)]`.

Run symmetric orthogonalized rank-eight ALS twice from fixed independent seeds.
Only after both fits are frozen, obtain the exact suffix gradient of the
UK-minus-US logit contrast with respect to the post-MLP9 residual on the 48
already-open `CROSSFIRST_HESSIAN_TOP2_FRESH_V1` rows.  Report the fraction of
reader norm in the recovered output-factor span.  Compare against 16
deterministic Haar rank-eight output subspaces.  This is an opened-panel
diagnostic, not an OOD or causal sufficiency claim.

## Predictions

- **A — tensor instrument:** eight deterministic random scalar contractions of
  the analytic Hessian agree with nested JVP through the raw MLP9 module within
  `2e-5` relative error.
- **B — solver health:** both fits are finite; input-factor orthogonality error
  is at most `2e-4`; and each rank-eight fit reduces the fixed random-probe
  residual tensor norm by at least 2%.
- **C — identifiability:** the minimum principal cosine between the two
  rank-eight input spans and between the two output spans is at least `.80`.
- **D — reader coverage:** the first fit captures at least `.50` of pooled
  CrossFirst reader norm, at least `.35` in every construction family, and a
  median of at least `.40` across individual prompts.
- **E — specificity:** pooled reader coverage exceeds the median of the 16
  equal-rank random-subspace controls by at least `.15`, and exceeds every
  random control.
- **F — audit:** serialize both weight-only fits, promptwise/family reader
  coverage, random controls, probe residuals, and exact price.

If A fails, the run is invalid and only mechanics may be repaired.  If B or C
fails with A passing, rank-eight DCT factors are not canonical enough for this
use.  If D or E fails with A--C passing, this weight-only DCT basis does not
recover the behavior-selected reader; prompt-conditioned/open-context DCT is
then required.  Passing does not establish extraction, removal, or OOD
prediction; it nominates a behavior-blind reader basis for those tests.

## Price

One checkpoint load; zero language-model forward passes during factor recovery;
two rank-eight implicit tensor fits; 64 fixed tensor-residual probes; eight
analytic/autodiff contraction checks; 48 opened prefixes and one downstream
gradient per prefix after factor freezing; 16 random rank-eight controls; no
physical intervention, behavioral fit, coefficient fit, or parameter update.
