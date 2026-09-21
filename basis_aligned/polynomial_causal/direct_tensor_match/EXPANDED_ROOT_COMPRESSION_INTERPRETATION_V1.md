**The improved parent also compiles to384products, but compression geometry matters.**

Bothfixedrank16metricarmsuse384products,322048FP32coefficients,768int64indices(1,294,336parameter/indexbytes). Parent656products903168coefficients. NumericalintegrityandpricePASS;primaryempiricalfidelityFAILbecauseoldcalibrationerrorsrise>10%. Thisfailureisnoterasedbybettersecond-panelpreservation. Runtime1.51s.

|Compression metric|Coefficient error to parent|Joint empirical error to parent|Oldcal/evalnativevalueerror|Evalnativehalf-responseerrors(997/613/1379)|
|---|---:|---:|---:|---:|
|No compression|0|0|3.24/8.99%|14.94/15.39/14.84%|
|Weight-only coefficient|18.61%|13.12%|12.60/14.80%|20.23/20.58/20.11%|
|Empirical values+responses|21.40%|3.52%|4.29/9.12%|15.14/15.59/15.05%|

Thetwoerrors-to-parentuse differentnorms; do notcompare21.40%coefficienterrorand3.52%functionalerrorasthesamestatistic. Weight-onlycompressionoptimizesexactquarticcoefficientFrobeniuserror;empiricalcompressionweightsparentvaluesandfourdonorresponseson6144states. Parentfeaturesaredata-informed,soevenweight-onlycompressionisnotend-to-endweights-onlydiscovery.

Eacharmusesoptimalrank16outputprojectionforthefixedparentanditsmetric, thenexistingexactquadratic-pairrewrites. Directspectralcertificates<7e-15;FP32graphreplay<8.9e-7bothpanels. All8rootpairsshareproductswithoutfallback;noadditionalfeaturelearning. Thediscrepancycomesfromrankrestrictionandmetricchoice,notpaircompilationerror.

NativeconditionalinterchangesuccessorretainsBOTHarms,parentandcheap26. Primaryempiricalrootmustpreserveparentresponsewithin10%relative,beatcheapby10%,andkeepbaselineCE<.02/<=parent+.002. Theoldcalibrationfailure remains regardlessofnativeoutcome. No semanticidentity,selectiveremoval,untouchedOODorwholemodel-speedupclaim.

[Compression plan](EXPANDED_ROOT_COMPRESSION_PLAN_V1.md) · [Results](EXPANDED_ROOT_COMPRESSION_V1.json) · [Native screen plan](EXPANDED_ROOT_NATIVE_PLAN_V1.md).
