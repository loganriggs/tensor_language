**Broader calibration states improve transfer at the same program budget.**

Native51.53seconds; allthreepredictionsPASS. Fit32quadraticfeatures/656products903168coefficients withsameinitialfactors/300Adamstepsandfourdonorfamilies. Calibration32→96prefixes,positions0–63unchanged,donorswithin2048-rowgroupspreservetheoldpairs. Moretrainingdata/perstepcomputeisexplicit;notmatchedFLOPs. Oldfourprefixcapturereplayandcachedteacherreplayarezero;FP32export3.59e-7.

|Second-panel metric|Small-calibration ensemble|Expanded calibration|
|---|---:|---:|
|Half response, held shift613|19.88%|15.39%|
|Half response, held shift1379|19.32%|14.84%|
|Ordinary value error|11.44%|8.99%|

The15%relativeimprovementbarpassesbothshifts;actualimprovements~22.6/23.2%. Expandedtrainingvalueerror3.16%;fittedpairresponses5.83–5.89%,unfittedpairs6.92/7.02%. Againsttheexpandedtraininginputs,theoldsmall-calprogramhas8.52%valueerror/~14.6%unseenpairerror;thatisadiagnosticontheexpandedset,notitsoriginaltrainingerror.

The96prefixescontain95distinct65tokenprefixes;oneoriginalduplicatewasretainedtopreservetheoldweighting. Noexactprefixoverlapwithevaluationpanel,butdocumentidentityanduntouchedhistoricalstatusarenotcertified. Thisisdata-informedweights/functiondecomposition,notpureweights-onlydiscovery. Greaterinputcoveragehelpsthefixedfeaturebudget,butthe14.8–15.4%remainingresponseerrorandstate-transfergaparestillmaterial.

Registerednativeconditionalinterchangesuccessornowcomparesfrozenexpanded656withsmallcal656andcheap26. Actualnormalization/softcapandcodebaselineCEaredifferentmetricsandmustbetested;earlierhalf-responsefitimprovedresponseswhileworseningcodebaseline. Nocircuitadoptionorsemanticunitsareclaimedhere.

[Plan](QUARTIC_EXPANDED_STATE_PLAN_V1.md) · [Fit receipt](QUARTIC_EXPANDED_STATE_V1.json) · [Native successor](EXPANDED_STATE_NATIVE_PLAN_V1.md) · [Input provenance](QUARTIC_CAPTURE_COVERAGE_AUDIT_V1.json).
