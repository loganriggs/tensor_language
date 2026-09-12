# Weight-generated background audit of token routing

Rawtokenroutingfailedwrite/effectfidelity. Preserve that result. Generate a
zero-input trajectory throughactualmodelweights: x0=0, length11,14blocks,
no embeddingtokenchosen andno corpusobservations. Beforeselectedblockj, cache
rawr_j^zero. Exactembeddingcoefficientrecurrence e_-1=1,e_j=lambda_j0*e_(j-1)
+lambda_j1. Approximate rawinputbyr_j^zero+e_j*RMS(embeddingtoken), thenRMS
andapplynativeQK1/QK2norm/RoPE. This restoresa weight-generated background,
while omitting input-dependent nonlinear update terms. It isnotanexactunroll.

Same32rows andfivearms asrawtokenrouting:baseline,nativefirstswap,anchoredboth,
nativequeries/anchoredkeys,anchoredqueries/nativekeys. BothQKs remainjoint.
A:nativeproducer/attention/savedfirstwritereplay<=1e-5relative.
B:A+anchoredboth branchwriteerror<=.1relative.
C:A/B+regional signedprefixeffecterror<=.1 andeachcontrol RMS<=.05nativeRMS,
meanabs<=.25nativefirstregional-effectRMS,no capableflips. Hybridsdiagnostic.
Null: missingcontext-dependentupdatesstillpreventclosure. Anyimprovement over
rawtokenbaseline isdiagnostic anddoesnotreplacethe10%fidelitybars.

Price6ordinarybodyforwards32rows +onezeroanchor14-blockpass11positions,
5suffixarms,180sec managedGPUcap,~1.7MBartifact. No parameterfit ordata-adapted
anchor. Prefix-causalarchitecture permits slicingzeroanchor toshorterlengths.
