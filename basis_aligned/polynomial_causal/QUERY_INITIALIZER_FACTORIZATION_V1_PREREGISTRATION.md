# Exact document/local query-initializer factorization

Local-only initialization fails native full-output fidelity. Keep that null.
Extract the exact two-input computation instead of enlarging the failed local
candidate and claiming it passed. At the first layer the final query projects
only its original hop-token embedding. Each source projects only its own token.
Unnormalized attention therefore gives the exact final-position state

    x1(document, entity, hop) = S(document, hop) + L(entity, hop).

S is the sum of initial-binding contributions across all four heads. L consists
of delimiter, queryentity, selfhop contributions and the .5 residual embedding.
Positions and length are explicit arguments: this claim fixes48 bindingtokens
and the final3 querytokens at positions48..50. It does not assert independence
from changing layout length or arbitrary position shifts. The source partition
defines the zero/erasure semantics and fixes the additive offset ambiguity.

Compute S directly from bindingtokens andhop, and L directly from thelast3tokens.
The compiler must not first compute discarded fullattention to claim deletion.
For each document compute allfour S vectors once and reuse them across all24
queryentity consumers. L may be reused across documents with the same local
tokens/positions. Both use original coefficients; no fitted table or rank choice.
The remainder of the model stays native/exported and explicitly charged.

Fresh34909/34910 first8worlds/population,all24entities x4hops,1536 requests across
16 independent worlds, IID24-cycle/OODthree8-cycle. Four keep masks: both,
summary-only (local removed), local-only (summary removed), neither. Native
oracle masks firstlayer finalquery sourcecolumns and removes residualhop when
local is absent. It leaves all earlier states and all subsequent layers native.

A extraction: exact factorization/reuse/source-partition controls andallfour
native/export fulloutput cells max<=1e-9/relativeRMS<=1e-10 (floor1e-6), finite,
earlieroutput closure. Both-kept program must reproduce independent native token
execution, meanKL<=1e-12. Allmodels/rows/sourcehashes andcachecosts recorded.
B removals: each compiled-minus-native removal vector must match its independently
hooked native counterpart withthe same1e-9/1e-10 bars. Report perhop/pop goldP,
taskaccuracy, fullKL, centeredvectorRMS andrelativeeffect. No semantic-selectivity
claim follows solely from saturated probability or one chosen control family.
C joint manipulation: predict the neither-kept native intervention; record the
complete mixed removal effect, not an additive approximation. In this biasfree
model an identically zero query state is absorbing, so its future logits should
be zero/uniform. Test this explicitly, together with live single-field effects.
D reuse: sameS acrossqueries andsameL acrossdocuments with identical localtokens;
cache invalidation follows document/hop/localtoken/position inputs. No cache may
be transferred silently between different documents or label frames.

This is an exact partial program/infrastructure receipt with explicit computation,
extraction, removal and reuse semantics. It is not full mechanistic identification
or the requested structural simplification. Compare single-request cost separately
from shared many-query execution; report all387968 coefficients, added adapters,
four128-value document summaries and any local-state cache. No native parameters
removed. Generic source-additivity/CSE savings do not count as a novel discovery.
Use managed GPU,FP64,B8,1800s,256MiB per newtensor. No training or rescue sweep.
