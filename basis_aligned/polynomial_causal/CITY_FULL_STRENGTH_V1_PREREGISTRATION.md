# Complete city-value removal on opened natural contexts

Frozen candidate: city_full_value_removal_v1.execute at strength=1. Remove the
entire city-token value contribution (inherited plus current) of head8.2 at the
declared destination positions. Keep both QK factors, both ordered MLP8 cross
terms, quadratic term and skip; keep the same approximate background/denominator.
Native reference recomputes the full suffix after the corresponding attention8 edit.
This tests prediction and selective manipulation at a prospective intervention
strength on twenty already-opened Pile documents. No fresh-data confirmation,
token-only execution or branch-composition claim.

All forty sequences and 240 diagnostic rows from CITY_FULL_PILE_V2 are retained.
Same four controls; sixteen norm-matched block9 nulls with seeds
17093100+1000*k+context_id. Frozen text/constant predictors are scaled by two,
with no refit. Also compare against twice the actual native half-removal effect
from CITY_FULL_PILE_V3. This is an opened, native-informed linear-scaling baseline.

Opposing predictions: the explicit quadratic response stays accurate and selective
as strength doubles; or context/normalization omissions produce a larger error or
collateral effects. A separate prediction tests whether the formula adds value
beyond linearly scaling the measured half-strength effect.

Gates (aggregate across documents, no post-outcome family selection):
- a: independent FP32 native reference local relative error <=1e-4; final reentry
  <=1e-4 absolute AND <=1e-5 relative; finite values, support exactly zero outside
  mask; null norm error <=1e-5; exactly800 model-body forwards.
- b: >=90/120 capable pairs (native margin>=.1); target relative effect error<=.35
  and candidate target RMS>=1e-5.
- c: every unrelated reader RMS <=.5 candidate target RMS.
- d: positive attenuation on >=.90 capable pairs, mean attenuation>=.02.
- e: candidate target RMS >=2 times null median and beats all16 nulls.
- f: target error <=.8 times each doubled frozen text/constant baseline error.
- g: unedited scores replay V3<=1e-5 absolute and independently calculated native
  block8 baseline matches captured block9 input<=1e-5 absolute.
- h: target error <=.8 times error of twice the measured native half-removal effect.
  Failing h limits the claim that explicit nonlinear response is needed at this
  endpoint; it does not erase a–g or prior half-strength evidence.

Price:800body forwards,300second cap, one native residual7 input, the same head8.2
and full MLP8 factors plus396token tables. Independent effective units remain20
documents. Save all arms and per-document outcomes, including any failures.
