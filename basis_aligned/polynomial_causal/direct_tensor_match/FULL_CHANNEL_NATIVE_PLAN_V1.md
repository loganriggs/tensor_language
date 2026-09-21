**Full lastMLP replacement through native output operations**

Freeze the two3686channel candidates and mean/tangent affine corrections from FULL_CHANNEL_FUNCTION_V1. Reconstruct from weights with unchanged selection and fitting before native evaluation. Evaluate bothraw andcorrected variants on the previously opened two panels(64FineWebdocuments+32codefiles),256tokens, scoredpositions16:254. Shared capture for all4arms. No text-based selection/refitting.

Replace full lastMLP polynomial residual write; preserve Downbias, native finalRMS, unembedding andsoftcap. Compare changedlogits withoriginal. Normalize error by norm of nativeMLP logiteffect relative tobackground withits polynomialremoved. Also report CEadded(lowerbetter), same nexttokenlabels. This is full-layer effect fidelity, not sixselectedreads, languageaccuracy or semanticselectivity.

Predictions: reconstructed finalresidual replay<1e-5; everycorrected panel/domain effecterror<10%; correctionimprovesevery correspondingrawcandidate. Allfailedcells retained. Price frompriorrun:11.668%reducedfactorparameter savingafterdenseaffinecorrection, not measuredwholemodelspeedup. Openedpanels diagnostic only; independentvalidationandinterventions stillrequired.
