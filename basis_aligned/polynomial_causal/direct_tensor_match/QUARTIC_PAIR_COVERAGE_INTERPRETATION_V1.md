**Both donor-pair specificity and input-state transfer contribute to the response gap.**

Frozen diagnostic0.74s; replayPASS(previoushalferrorswithin6.7e-10). Pairgeneralization<=1.5trainingerrorFAIL; eval>=2cal onbothuntrainedshiftsPASS.

|Half-trained dictionary|Trained shift997|Untrained613|Untrained1379|
|---|---:|---:|---:|
|Fitting inputs|3.92%|8.98%|8.82%|
|Second-panel inputs|21.11%|21.49%|20.93%|

Changing donors among identical fitting inputs morethandoubleserror. Yet second-panelerrorisstill>2timeserroronunseenfitting-panelpairs. Thus neitherpairmemorizationnornew-statecoveragealoneexplainsallthepattern. Full-pathobjectivecloselyreproducesit(fittingunseen8.89/8.71%,second21.67/20.98%). Value-onlybaselineisbadacrossallpairs, so responsefitgainsarenotlimitedtotheexacttrainingpairs.

ThesearepurequarticnumeratorchangesinunembeddingEuclideanmetric. NoRMS/logitresponseclaim, freshOODorsemanticidentification. Frozenprogramsandtwoexistingpanels; targetresponseenergiesaresimilaracrossshifts, so .04→.09isnotasimplevanishingdenominatorartifact.

Nextcontrolledchangeincreasesdonorpaircoveragewhileholdinginputstatesandarchitecturefixed. `ensemble_design` nowbuildsajointvalue-plus-multiple-responseobjectivewithoneglobalresponseenergynormalizer. Itdoesnotgive eachsmall-energyfamilyequalrelative-errorweight. CPUcontrolsverifyK1equivalencewiththeoldobjectiveandK4duplicationinvarianceoflossandgradients,plusactualprofiledfitcallbacksmoke. Thesearepreparatoryalgebra checks,notnewnativefitresults.

A subsequentfour-shiftfitshoulduse997pluspreviouslyunfitted313/1427/179; keep613/1379outoffittingandretainpanel1evaluation. NativebaselineCE/interventionvalidationremainsrequiredregardlessofpolynomialimprovement. Moreinputstatesremainaseparatepossiblechange, notsilentlycombinedwithpaircoverage.

[Preregistered diagnostic](QUARTIC_PAIR_COVERAGE_PLAN_V1.md) · [Rows](QUARTIC_PAIR_COVERAGE_V1.json) · [Ensemble objective](quartic_finite_response.py) · [Controls](QUARTIC_ENSEMBLE_DESIGN_CONTROLS_V1.json).
