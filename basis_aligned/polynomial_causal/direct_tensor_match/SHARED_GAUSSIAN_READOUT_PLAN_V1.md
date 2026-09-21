Exact shifted-Gaussian fitting for the lower-cost shared hierarchy

Use fixed graph artifacts SPARSE_SUPPORT_EXCHANGE_SEED1101/1102_V1.pt:144quadratics with4products each,512learned rootpairs,16fixed outputs. Same1088products/1353728floats/1024indices. No direction changes, new pairs or calibration-output fitting. The Gaussian law is N(mean,covariance) from6144calibration inputstates; readers remain data-informed. Native target remains the pure MLP16→MLP17 quartic, not the entire normalized network.

Compute exact coefficient Gram/cross G0,X0 and exact shiftedGaussian G1,X1 without materializing the quartic tensor. Candidate selfGram uses low-rank quadratic cumulants; native cross expands only16rootpairs at a time into256CPterms. Reuse the cached native mean/gradient/Hermite-quadratic projection for Gaussian teacher contractions. Save the four Grams/crosses for later CPU audits rather than rerunning native contractions.

Compare budget0(coefficientoptimal), preregistered primarybudget1, and unconstrainedGaussian readout. Budget1 permits one captured coefficient score of regularized coefficient deterioration; it is not100%of total teachererror. BothGrams use explicitridge1e-6. Fixed-feature convex solver andexactconstraints unchanged. No best-armselection on evaluation.

Pred_a_integrity: archivevalue replay<1e-4, physicalFP32export<1e-4, positivebudget KKT<1e-8, feasibilitywithin1e-8(1+budget). Pred_b_values: bothprimarytext<=.8coefficientbaseline andsampledcoefficienterror<=1.1baseline. Pred_c_component: bothprimaryroot1same-tokenresponse andsensitivityweightederror<=10%. Preserve failedcriteria andallthreearms. Higher-cost mixedCP6.32/6.59%text anditsnativefinite-removalresults are references, not equivalent-cost claims. No nativefinite-removal/adoptionfromthisfit alone.

Evaluation: same4096coefficientindicesseed951,1024isotropicGaussianpointsseed939,2048openedtextstates,30directed/20independent same-tokenpairs. These are not untouchedOOD. Gaussian fit itself uses exact weight contractions plusinputstatistics, not theseevaluation labels.

Implementation controls:10cases across5plantedfamilies, bothzeroandnonzeroaffinebiases. Independent5^3Gauss-Hermitequadrature integratesdegree8exactly; selfGram,nativecrossandcombinedgradients agree<1e-10 (observed~1e-15). Actual144x4/512pairCPU shape executes bothGrams,bothcrosses,threebudgetsolves andpredictionbranch beforemanagedenqueue. No hidden eigenfloor or changedregularizer.
