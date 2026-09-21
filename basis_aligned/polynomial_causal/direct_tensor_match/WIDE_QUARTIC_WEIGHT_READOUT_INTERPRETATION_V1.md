**Exact coefficient readout fitting does not repair the broad dictionary's functional transfer.**

For the fixed learned32-feature dictionary, empirical fitting gives2.16%calibration/15.05%second-panel error. Isotropic coefficient fitting gives55.27%/54.38%; savedsecondmoment coefficient fitting gives20.39%/21.44%. The randomdictionary controls give nearly100%error underisotropic fitting and62.38%/67.36%undersecondmoment fitting, versus37.21%/63.49%empirical. All programs have656products and903,168storedcoefficients.

The fixed-dictionary coefficient objective improves, FP32/FP64crosschecks are<6.3e-7, normal-equationresiduals<1.4e-15, andphysicalexportreplay<4.7e-7. Integrity andweighted-vs-isotropic predictionsPASS; the20%transfer-improvement/calibration5%predictionFAIL. This is evidence of metric/dictionary limitations ratherthan an obvious solve or export error. It is not a proof that weight-based optimization generally fails. Objective values omit the fixedteacher norm, so they are not relative coefficienterrors.

Only randomdictionary plusisotropic fitting is fullyweight-only. The learned dictionary andsavedsecondmoment bothuseactivationinformation. Both evaluationpanels were alreadyopened andthetarget ispurequartic, excludingresidual/biaspaths andinterveningnormalization. No semantic ornativebehavior adoption.

Successor: exact Gaussian functional Gram andteacher cross, retaining the Hermite0/2/4 contributions that coefficientFrobenius doesnotweight asfunctionalerror. ExistingGaussiantraceidentities andsmallrootmoment work are reused; newoperatorhandleswidebanks throughcompressedleafspan. A Gaussian input law is still an assumption, not establishednaturalinputgeometry.

[Primary results](WIDE_QUARTIC_WEIGHT_READOUT_V1.json) · [Preceding empirical comparison](WIDE_NATIVE_QUARTIC_INTERPRETATION_V1.md).
