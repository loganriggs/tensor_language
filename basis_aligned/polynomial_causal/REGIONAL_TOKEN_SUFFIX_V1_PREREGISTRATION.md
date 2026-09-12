# Full-output conditional token suffix

Native tokenpathcompiler A/B/C passed. Use its prespecified firsttwo contexts,
notbestcasesselectedbyoutcome. Sixedits:0,donor,-donor,.5donor,1.5donor,normal
synthetic std.1 seed61209. Flatten54token-valueports. CompileL*K,R*K, inputnorm
quadratic coefficients; preserveDown+bias, finalRMS, all50304unembeddingrows
and30tanh cap. FP64 referenceuses samecheckpointweights andnativefloat32epsilon.
Require hiddenstate,fulllogits andnonzero logit-effect relativeerrors<=1e-10.
This isexactconditionalalgebra,notnewnativebehaviororOODprediction. Arbitrary
synthetic54portvectors neednotrepresentvalidtokens. Kstillnativecontext-generated.
CPU2threads,2contexts,6edits,fullreadouttransientonly; save scalarreceipts (~KB).
PriceL/Rmaps497664numberspercontext, inadditiontoK62208,Down,fullU,andcontext
construction. Theseare notchargedasa smallerindependentmodel.
