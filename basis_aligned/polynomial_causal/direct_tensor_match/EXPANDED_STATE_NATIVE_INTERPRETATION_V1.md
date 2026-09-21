**Broader calibration improves native responses and baseline preservation, but absolute fidelity remains inadequate.**

Native2.20s; numericalcontrolsPASS (replay4.73e-8,zeroedit0). Absolute10%responsepredictionFAIL; combinedrelativepredictionFAIL becauseFineWebgainagainstsmallcalisbelow15%. Bothcodecellsimprove>15%; allcellsbeatcheap26by>10%; baselineCE<.02bothdomains. Keepthecombinedfailure,notjustitspassingpieces.

|Domain|Strength|Cheap26 error|Smallcal656 error|Expanded656 error|
|---|---:|---:|---:|---:|
|FineWeb|.5|42.55%|31.42%|29.42%|
|FineWeb|1|28.41%|25.11%|24.04%|
|Code|.5|36.28%|26.27%|20.97%|
|Code|1|31.59%|23.77%|18.99%|

BaselineCEaddedFineWeb .00215→.00191;code .02391→.01228. Lowerisbetter. ExpandedcodeCEresponse .12898/.62265 versusnative .12647/.62121 iscloseonaverage, butfullresponsevectorerrors20.97/18.99%showwhymeanCEagreementisnotenough. Nochangein656product/903168coefficientprice.

Scopeunchanged: purequarticterm'sinputswapped/interpolated,recipientdenominatorandothertermsfixed,eachcandidateownbaselineandactualfinalnormalization/softcap. Notfullupstream-sourceinterchange,semanticsoruntouchedOOD. Thisimprovedparentprovidesabetterstage2compressiontargetthantheretiredsmall-calibrationparent,butdoesnotjustifyadoption.

ActualCPU successor `weighted_output_projection.py` computesoptimaloutputsubspaceforafixedparentandPSDfeatureGram. Fiveindependentdirect-predictionSVDcontrols,including singularmetricsandcancellation,pass<5.5e-15. Plannedcomparisonusesexactweight-onlyquarticcoefficientGramversusactivation-informedvalue/responseGram,thenexistingroot/paircompiler. Sinceparentfeatureswerelearnedwithdata,weight-onlycompressionisnotend-to-endweights-onlydiscovery. Preservebothmetricresultsandsameoperationaccounting.

[Native plan](EXPANDED_STATE_NATIVE_PLAN_V1.md) · [Rows](EXPANDED_STATE_NATIVE_V1.json) · [Next compression plan](EXPANDED_ROOT_COMPRESSION_PLAN_V1.md) · [Projection controls](WEIGHTED_OUTPUT_PROJECTION_CONTROLS_V1.json).
