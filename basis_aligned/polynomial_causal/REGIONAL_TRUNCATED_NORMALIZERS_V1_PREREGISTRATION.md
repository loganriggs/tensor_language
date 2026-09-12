# Kernel-preserving normalization comparison

Same eigenreaders,32/64/128ranks,48rows and9bodybatches as spectral normalizersV1.
Only change: omit isotropic remainder. The PSD quadratic is sum of retained
lambda_i*(v_i dot x)^2, divided by128 plus nativeepsilon. Every native null input
remains null beforeepsilon. A full128/native/prior component replay<=1e-5;
B both32 write/removal-effect<=.1 EACHfamily; C both64 same. Query32/key32 are
separate diagnostics. No data fit, rank increase or threshold change.7suffixarms,
180secmanagedcap,<4MBartifact. Reader/eigenvalue packages cost4*9*k*(1152+1)
scalars; original upstream/numerator dependencies remain. Null: preserving
native kernel does not suffice at these reader counts; no absentstructure claim.
