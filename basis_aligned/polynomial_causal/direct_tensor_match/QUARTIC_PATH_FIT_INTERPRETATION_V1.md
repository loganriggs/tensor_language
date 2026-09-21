**Exact full-path integration does not materially improve transfer over halfway-response training.**

Native fit22.51s, same32quadraticfeatures/656products903168coefficients and300Adamsteps. IntegrityPASS;10%integrated-response improvementFAIL; value/point preservationPASS.

|Fit objective|Fitting path error|Evaluation path error|Evaluation value error|Evaluation half/full response error|
|---|---:|---:|---:|---:|
|Values|14.91%|23.48%|13.02%|27.69/19.19%|
|Halfway responses+values|3.97%|18.91%|11.85%|21.11/16.83%|
|Integrated path+values|3.73%|18.88%|12.05%|21.11/16.80%|

Pathimprovementis~.139%relative, farbelow10%bar. Exactquadrature/controlsalonecannotrepairdata/featuregeneralization. The widertraining/evaluationgapremains. No newnativeinterventionadoptiontestisjustifiedsolelybythisnear-tie; retainprevioushalfmodel'snativecode-baselinedamagetradeoff. Do notprolongalpha-objectivesweepswithoutanewmechanism. PolynomialexactnessdoesnotextendthroughRMS/softcap.

ActualCPU successor measuresprincipalanglesbetweenquadratic-featurecoefficientsubspaces, removingfeaturebasisgauge. Mean squaredprincipalcosine is.525betweenvalueandhalf fits,.544betweenvalueandpath,and.843betweenhalfandpath. Allspacesrank32; nohalf/pathprincipalcosineexceeds.99,25/32exceed.9. Againstinheritedfeatures,half/pathmeans.584/.597. Thus responsefeaturelearningchangedthequadraticspan,itwasnotmerelyareadout/basisrotation; halfandpathfitschoosecloserbutnotidenticalspaces. Thisdoesnotestablishunstableindividualsemanticsoruniquerecovery: objectivesdifferandnoindependentrestartcomparisonwasmade. CoefficientFrobeniusgeometryisnotactivation-weighted.

TheauditusesimplicitquadraticGramcontractions; independentdenseGramreplay andrescale/permutationcontrolsPASS. A response-aware dictionaryhasusefulbutincompletefunctionstructure. Nextinvestigation should distinguish insufficientcoveragefromstructuralcapacity/optimization, ratherthanassumeamorepreciselyintegratedsame-pathlosswillfixit. Retainmatchedcheapbaselinesandnativeinterventioncriteria; do notcallfeaturessemanticunitsfromspanoverlap.

[Registered plan](QUARTIC_PATH_FIT_PLAN_V1.md) · [Fit result](QUARTIC_PATH_FIT_V1.json) · [Subspace audit](QUARTIC_FEATURE_SUBSPACE_AUDIT_V1.json) · [Audit code](audit_quartic_feature_subspaces.py).
