**Exact coefficient-contraction feature learning, native pilot**

Target:16fixedvocabulary-metric projectedpurequarticreaders throughnativeMLP16→17. Sameinherited32quadraticfeatures4products each and16fixedwriters. Global outputscale19054614563.464127 onlynumerical; rootcoefficientsconvertedbackonexport. No text/syntheticinputsample participatesintrainingloss.

Computeexact symmetric coefficientGramGandnativecrossX usingbank_gram/native_bank_cross. ProfileC=X(G+1e-6I)^-1. Loss tr(CGC^T)-2<C,X>+1e-6||C||² omitsconstantteachernorm. OptimizeU/Vwith25Adamsteps at.03sqrt(4/1152), cosinefloor1%, normalizebankeveryevaluation. Normalizeoptimizerlossbyabsoluteinitialloss only;constantnormalizationdoesnotchangeoptimum. Recordevery5steps,time,peakGPUmemory; besttrainingobjectivecheckpointonly. Oneinheritedinitialization, no convergence/stabilityclaim.

Compareinitialexactprofilewriter,finallearnedexactwriterandinheriteddata-fit. Reportexactexplainedregularizedcoefficientenergy(-loss), not an unsupportednormalizedteacherFrobeniuserror. Independentdiagnostics:4096uniformcoefficienttuples(seed951),1024Gaussianvectors(seed939),and2048openedtextstates. Coefficientqueryerror is sampledvalidation, not thetrainingobjective. Teacherconstantnormnotcomputed, so noexactrelativeerrorcertificate.

Predictions(a)finiteobjective/gradients,exactnormalresidual<1e-8,FP32export<1e-4;(b)finalnegativeobjective magnitude>=1.1initial andsampledcoefficienterror<initial; (c)freshGaussianerror<=.9initial andtextaggregatevalueerror<=1.1inherited, with<=384products/330240coeff. Pricefallbackhonest. NumericalGPUdoubles; CPUfivefamilydensecoefficient/gradientcontrols alreadyPASS. Bounded25steppilotexplicitlyexpensive; no coefficientmetric improvementimpliesnativecausalbehavior.
