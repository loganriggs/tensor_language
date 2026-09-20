# MLP8 value mediator with generated edited RMS

`runtime=execute.prepare(torch.load("program.pt",weights_only=True))`, then
`execute.execute(runtime,z,delta,token_ids)` returns the FP64[1,T,128] value correction.
Only these files and PyTorch are needed. CPU batch1; native z8 and upstreamdelta
are[1,T,1152];token IDs[1,T]. Frozen402-token vocabulary; unsupported tokens/batch>1 fail.

The program computes normalized bilinear MLP8 hidden products, edited MLP8 output
including bias, block9 residual/reentry mixing and editedRMS9. It folds the head9.8
value reader into Down8 once at prepare time. No supplied native norm or edited
residual state remains. The initial-token table is checkpoint-derived, not fitted.

Subtract output from head9.8 value while retaining the upstream cityswap, bothkeys,
otherheads and nativefullsuffix. Nativez8 and the interventiondelta still require
external computation. No token-only, independentcomposition or storagewin claim.

Price16,536,963FP32values/66,147,852bytes plus3,216token-IDbytes. Derived FP64fold
cache589,824values/4,718,592bytes. FP64runtime factor arrays137,014,296bytes before
activation workspace. Nativez8 atT32 is36,864scalars;delta36,864intervention scalars
counted separately. Storedweights increase versus the suppliednorm interface.

See manifest for opened local/installed/isolated evidence. Fresh confirmation of
this expanded generator remains pending. Preserve prior fresh-mediator evidence
and failed direct/MLP8 composition as separately scoped results.
