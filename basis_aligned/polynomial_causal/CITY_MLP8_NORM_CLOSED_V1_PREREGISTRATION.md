# Generate the edited RMS instead of supplying a counterfactual native scalar

Fresh direct/MLP8 composition fails the reversed-group bar(.4908>.35). Retain the coupled MLP8 mediator and stop independent-part promotion. Next WEIGHT_FOLDING step closes its edited-RMS port exactly from native factors, with explicit cost.

Inputs:native unnormalized MLP8 input z8[1,T,1152],upstream intervention delta[1,T,1152],tokenIDs[1,T]. Compute native bilinear hidden vectors h0=(Lz*Rz)/S(z),h1=(L(z+d)*R(z+d))/S(z+d). Compute edited MLP8 output D h1+b, edited mixed9=lambda0*(z+d+D h1+b)+lambda1*x0(token), then its RMS with FP32epsilon. Return(1−mu)*lambda0/rho9edited*(WvD)(h1−h0). Weight-derived x0 table from checkpoint; no fit/frozen-normalizer approximation. Bias affects norm even though it cancels in the value difference.

Store native L,R,D,b,Wv,lambda9,mu and supported-token initial embeddings. Cache WvD as a derived contraction, separately priced; do not count derived cache as independent storedweights. This grows literalweights to remove native editedcontext, not a storage win. Native z8 and native upstreamdelta still external; no token-only wholecircuit or independentcomposition claim.

Opened40fresh-mediator fixtures from CITY_VALUE_MEDIATION_FRESH_V1. pred_a:generated editedRMS<=1e-5relative persequence against supplied nativeRMS. pred_b:mediator output<=1e-4relative persequence against independent nativeMLP8correction. pred_c:finite/exact support0;rejectunsupportedtokens,batch>1;allfloat32storedfactors physicallyowned;report every stored/derivedfloat andbyte. TwoCPUthreads,zero model forwards for localcheck.

Then installed equivalence:80sequence-equivalent forwards/36blockcalls,2CPUthreads180second cap,native versus exactupstreamswap+generatedRMSmediator. Prior fresh native/folded-correction score replay<=1e-4abs AND1e-5relative,all5effecterrors<=1e-4relative. Do not claim behavioral equivalence from local RMS equality alone. Fresh generalization of the expanded generator remains separately scoped after this opened interface check.
