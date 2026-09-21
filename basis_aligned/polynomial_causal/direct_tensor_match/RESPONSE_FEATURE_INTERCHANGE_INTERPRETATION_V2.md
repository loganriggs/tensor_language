**Finite-response feature learning improves some native responses, with a baseline-damage tradeoff.**

V2completed2.19s after fixing the documented V1evaluator-dispatch error. Exactreplay4.73e-8, zeroedit0: instrumentPASS. Absolute10%responseFAIL; relative20%improvement/everycellFAIL. No threshold waschanged.

|Domain|Strength|Narrow26|Matchedvalue656|Response656|
|---|---:|---:|---:|---:|
|FineWeb|.5|42.55%|42.35%|34.29%|
|FineWeb|1|28.41%|26.83%|27.19%|
|Code|.5|36.28%|37.12%|27.76%|
|Code|1|31.59%|33.42%|25.76%|

Response656improvescodeandFineWebhalfstrength, butFineWebfullslightlyworsensvsvaluecontrol. MeanbaselineCEaddedFineWeb .00355→.00312;code .01293→.02927. The candidatewouldmissappliedhistorical.02codepreservationbar, thoughthatbarwasnotaregisteredpredictioninthisrun. Reported responseimprovementdoesnotjustifyignoringbaseline damage.

Allchangesarepurequarticnumeratorchangeswithrecipientdenominator/backgroundheldfixed. Eachcandidate'sownbaselineissubtractedbeforecomparingfinal-logitresponses. Thisisnotfullsourceinterchangeorselective semanticbehavior. OpenedFineWeb/codepanels, notuntouchedOOD. Originalcheap26retained. Value/controlandresponseprogramsbothexactly656products903168coefficients;comparisonisolatesfitobjectiveatfixedarchitecture/initialization/steps.

Nextmechanisticuncertainty: fittingonlyalpha=.5mayunderconstrainthequarticresponsealongadirection. CPU successor `quartic_path_objective.py` implementsuniformalpha0..1integratedsquaredresponseerror. Adegree4residualhasdegree8squarednorm;fiveGaussLegendrepointsintegrateitexactly. Fiveindependentdensepolynomial/gradientcontrolsagreebelow9e-16. Exactnessappliesonlytopolynomialnumerators,notRMS/logitendpoints. ThisdoesnotpromiserepairedcodeCE. Useexisting300stepmatchedfitmachineryforasinglecomparisonbeforebroaderhypothesissearch.

[Native results](RESPONSE_FEATURE_INTERCHANGE_V2.json) · [Original plan](RESPONSE_FEATURE_INTERCHANGE_PLAN_V1.md) · [Execution correction](RESPONSE_FEATURE_INTERCHANGE_EXECUTION_CORRECTION_V2.md) · [Path objective controls](QUARTIC_PATH_OBJECTIVE_CONTROLS_V1.json).
