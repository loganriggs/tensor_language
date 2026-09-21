**Four donor families improve unseen pairings, but miss the required second-panel gain.**

Native18.89s, controlsPASS, combinedtransferpredictionFAIL, valuepreservationPASS. Sameinputstates/32features/656products903168coefficients/300Adamsteps; additionalconstraintsandperstepcomputeexplicit.

|Half-strength response metric|Single-pair fit|Four-pair fit|
|---|---:|---:|
|Fitting inputs, unseen shift613|8.98%|6.96%|
|Fitting inputs, unseen shift1379|8.82%|7.02%|
|Second panel, shift613|21.49%|19.88%|
|Second panel, shift1379|20.93%|19.32%|
|Second-panel ordinary values|11.85%|11.44%|

Bothunseenfitting-pairerrorsmeet20%relativeimprovement. Second-panelgains~7.5/7.7%miss10%bar; failureisretained. Fitting-pairerrors4.62–4.68%remainbelowunseenpairs~7%,whichremainwellbelownew-state~19–20%. Paircoveragehelpsbutdoesnoteliminatestatecoverage/generalizationlimitations. Thisisapolynomialnumeratorresult,notnative-logitfidelityorsemanticadoption.

AnactualCPU provenanceauditreplaysbothcachedtokenhashes andfindseach2048-rowpanelcontains32prefixchunks×64positions(0–63). Calibrationhas31distinct65-tokenprefixes,not32uniqueones. Noexactprefixoverlapwithsecondpanel. Originalmetadataword“documents”doesnotcertifydocumentdisjointness; useprefixeswhenmakingcoverageclaims.

Thereare64additionaldistinctprefixes(rows32:96)inthecalibrationtokenarchive,4096potentialnewstates,samepositionrange. Noneexactlymatchesoldcal/evalprefixes. Thisisacontrolledwaytobroadenstateswithoutfittingevaluationpanelorchangingcontextpositions; itdoesnotproveuntouchedhistoricalusage/documentindependence. Noadditionalactivationcapturehasrunyet. Oneexpanded-statefitwithsamemulti-pairobjectivecanseparatestatecoveragefromcapacity; moredonor/alphasweepsalonearedemoted.

[Plan](QUARTIC_ENSEMBLE_FIT_PLAN_V1.md) · [Fit rows](QUARTIC_ENSEMBLE_FIT_V1.json) · [Capture provenance audit](QUARTIC_CAPTURE_COVERAGE_AUDIT_V1.json) · [Audit code](audit_quartic_capture_coverage.py).
