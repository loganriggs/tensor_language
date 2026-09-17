# Prospective structural prediction with frozen constant and fitted baselines

Rows: ten new constructions in five families, two previously named city pairs,
six unchanged endpoint pairs. Twenty context/city cells,40sequences,240rows.
No exact prefix overlap with1873prior prefixes. All outcomes fresh at freeze;
this is new-construction transfer, not new-city/new-endpoint or corpus OOD.

Reuse TYPED_FACE_STRUCTURE_V1's exact intervention/scoring code unchanged,
including destination support, both QK factors,20arms,16equal-norm nulls and
four unrelated readers. 800bodyforwards/300seconds. Its six gates remain:
A finite/exact/norm instruments; B>=18/24capable pairs and parentRMS>=1e-5 per
family; C face/parent effecterror<=.35 perfamily and endpoint; D>=.75 positive
attenuation and mean>=.02; E>=2timesmedian null and beats>=15/16; F everycontrol
RMS<=.5target, all perfamily. A capability miss remains inconclusive for transfer
in that family; no replacement or dropping cells.

New pred_g: face relativeL2 error <=.8 times BOTH cue-constant and frozen-text-OLS
error in EVERY family. Constants/OLS are the exact saved numerical coefficients
from STRUCTURE_PREDICTION_NULLS_V1_RESULT.json. Do not refit or add features.
The same city pair index (Cambridge/Phoenix=0,Leeds/Chicago=1) and endpoint meanings
are preserved. Features: intercept,cue sign,length/32,cityposition/32,pair index,
five endpoint indicators. Training rank9/10 and eight base sequences: report
this weakness, no claim to beat all fitted predictors. Zero-effect baseline
has error1. Compare conditional native-state formula to these token-feature
baselines; differing information access prevents a matched-compute claim.

No source omission, extraction/port-count change, composition rescue or stronger
city/endpoint/token-only claim follows. All gates frozen before model scores.
