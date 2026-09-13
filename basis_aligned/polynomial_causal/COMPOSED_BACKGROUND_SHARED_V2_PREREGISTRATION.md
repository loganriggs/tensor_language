# Full-panel generated-background validation of shared interaction graph

Reuses existing8-prefix closure protocol, now all160 historical prefixes:96regional and64FineWeb. Candidate uses composed_mlp10_inputs_v2 plus generated g10(child)+g10(remainder)-g10(pristine). Separate baseline drift, transported local-cross effect and total output discrepancy through native suffix.

A: maximum generated-background state error<=1e-5 and full-cross product error<=0.001. B: each regional group target-effect relative error<=1%, control<=5%, maximum total target/control output error<=1e-4, no target sign reversal when native effect magnitude>=1e-5. C: each FineWeb group target CE-effect relative error<=10% and maximum total output error<=1e-4. Preserve tiny-effect misses; earlier FP64suffix discriminator does not waive nativeFP32criteria.

160pristine prefixes,320native branch reference calculations,800suffix readouts;180second managed GPU limit. Old sources unchanged. Native changed branches only score reference; candidate changed attention/jointnorm/background are generated. Native pristine context, frozen scalarfields and suffix remain. No newOOD or full child/remainder interaction sufficiency claim.
