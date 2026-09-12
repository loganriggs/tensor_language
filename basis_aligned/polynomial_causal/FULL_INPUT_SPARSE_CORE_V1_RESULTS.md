# Full-input sparse interaction result —12 September05:35 UTC

Folding through MLP16 concentrates edges more than the isolated last layer,
but neither fixed spectral frame is strongly sparse. Full input coverage does
not rescue the earlier best256edge result. All outputs remain in the objective.

| Centered coefficient capture | Last bilinear layer | MLP16 producer pullback |
|---|---:|---:|
| Best256edges | 1.4019% | 2.7123% |
| Best1024edges | 2.3889% | 4.4391% |
| Best4096edges | 4.5614% | 8.1839% |
| Edges needed for90% | 486,547 | 382,163 |
| Active readers among4096edges | 412 | 589 |

The relative improvement at4096edges is79.4%, exceeding the25% prediction;
the absolute50%capture prediction fails. Instrument identities hold to4.4e-15.
Managed V2 completed in11.53seconds. V1 never executed a native calculation:
preflight rejected keyword-style prediction keys; V2 only spells those keys
explicitly. [Raw result](FULL_INPUT_SPARSE_CORE_V2_RESULT.json),
[preregistration](FULL_INPUT_SPARSE_CORE_V1_PREREGISTRATION.md),
[CPU control](FULL_INPUT_SPARSE_CORE_V1_CONTROL.json).

The full-frame best256 result differs negligibly from the restricted128frame:
1.4019% versus1.4011% for the isolated layer,2.7123% versus2.7112% for the
producer pullback. Thus truncating inputs was a plausible explanation checked
and found insufficient for this particular sparse-edge failure. The complete
128input subcore holds6.406% and10.707%, respectively; that is not the same
quantity as energy in its best256edges.

Mathematically, each token's symmetric interaction matrix is transformed into
the complete orthogonal input-marginal eigenbasis. Diagonal entries and
sqrt(2)-scaled off-diagonal entries form orthonormal coordinates. Their squared
norms across the full centered unembedding give exact edge energies. Selecting
the largest is globally optimal **only within this fixed basis**. The producer
case changes the metric through the exact MLP16 quadratic Gram; it still uses
paired coefficients, not the fully symmetric quartic or FineWeb distribution.

Do not infer an arithmetic lower bound from the large90%edge count. The native
layer already has an exact4608product representation using two nonorthogonal,
potentially overcomplete reader banks. It costs more per product than a shared
orthogonal frame, but proves that these edge counts are representation-dependent.
No causal circuit or native extraction success follows from this screen.

**Metric clarification:** centering U defines the coefficient objective, not an
exact separately preserved common-logit channel in the physical graph. Earlier
preregistration wording overstated this. See the [execution interface and
correction](SPARSE_PRODUCER_GRAPH_INTERFACE_V1.md); frozen predictions/code are
unchanged, and eventual native validation must use the complete U and tail.

Next action executed: [streamed frame optimizer](streamed_sparse_frame_v1.py)
reuses exact gradients and Armijo while streaming complete support selection.
CPU dense-selection error is0, orthogonality error5.83e-16, capture increases
0.40360→0.60474 over20updates with no decrease. This validates selection/ascent,
not global recovery. [Next preregistration](STREAMED_SPARSE_FRAME_V1_PREREGISTRATION.md)
tests the fixed-frame restriction with full inputs. Native wrapper remains to
be completed; no optimizer GPU run is represented as queued or converged.

## Live optimization diagnostic —12 September05:49

The spectral arm remains unconverged. Between saved updates80and100, only10of4096edges change (0.244%; Jaccard0.99513), while median same-labelled reader cosine is0.999826 and the relative gradient remains0.01119. Rapid support churn is not supported as the dominant explanation in this interval. This is within-run continuity, not independent-start circuit identification. [Turnover receipt](STREAMED_FRAME_SUPPORT_TURNOVER_V1.json) preserves actual iterations and retained support arrays. A prospective improvement is several exact fixed-support gradient steps between expensive complete reselections; retain monotone ascent and reselect before declaring stationarity. The registered live run remains unchanged.

## Optimization recovery limitation —12 September05:54

The amortized PR+ solver recovers both diagonal planted starts, then4of5starts on a connected mixed12edge/8reader graph. Allfive satisfy localgradient/gap criteria; one independent start stops at92.8556% despite an exact100%solution. Its tangent Hessian has eigenvalues[-0.93867,-0.05157] with positive supportgap0.0006923, numerically consistent with a strict local maximum rather than an escapable first-order saddle. This is a numerical check at an approximate stationary point, not a certified theorem. [Mixed control](AMORTIZED_MIXED_GRAPH_V1_CONTROL.json), [curvature](AMORTIZED_MIXED_GRAPH_V1_CURVATURE.json). Native convergence can therefore not establish absence of a better graph; independent starts and their whole-function comparison remain necessary and insufficient for a global guarantee.

## Full-frame gradient run terminal —12 September05:59

Both arms hit600second limits after114updates each: spectral capture8.18386%→9.08584%, independent0.80584%→5.86478%. Relative gradients0.00817/0.00879 miss1e-6; fullfunctioncosine0.651218 misses0.95. Numericalidentity bar passes; convergence,25%gain andstability fail. Total1209.21seconds. [Terminal receipt](STREAMED_SPARSE_FRAME_NATIVE_V1_RESULT.json), [frozenframes](STREAMED_SPARSE_FRAME_NATIVE_V1_FRAMES.pt). These aredifferent unconverged fits, not identified circuits or proofagainst bettergraphs. Prepared frozennativefidelity andamortizedPR+continuation nowboundtofinalartifacts; originalresults retained.

## One recurring reader inside unstable full fits —12 September06:03

Maximum-weight one-to-one matching of the two complete frames finds exactly one paired-quadratic reader cosine>=0.95: spectralnode0 andindependentnode251 at0.992898. Median assignedcosine is0.09855. Only their self-edge has two stable endpoints in bothgraphs; incidentedge counts are671and1152, so similar readers do not establish similar complete consumer functions. [Correspondence](STREAMED_FRAME_READER_CORRESPONDENCE_V1.json). The recurringreader hascosine0.88309/0.90430with thepreviousconvergedsharedparent, hence it is related butnotanexactalias under thismetric. [Prior-parentcheck](STREAMED_FRAME_PRIOR_PARENT_V1.json). Do notcounta newcircuit orassumeinterchangeability. Frozen nativefidelity andcontinuation remain queued.

## Frozen native fidelity —12 September06:03

The2.48second [native screen](SPARSE_FRAME_NATIVE_V1_RESULT.json) passes numerical/old-reference replay, but primary swap/removal/write bars fail. Overall write error improves14.521%initial→9.120%optimized. Family swaperrors are18.04%,49.61%,38.24%,14.41%; removalCEdisagreement0.0550,0.1133,0.0385,0.0513, allabove0.02. Bettercoefficient/write fit hasnotpreserved causal effects.

The separately preregistered coefficient-selected node isnode0, the same reader independently foundrecurring acrossstarts. Its671incidentedges/671activereaders cost14,481,792conditionalfloats includingnativeL16/R16. Nodewriteerror5.368%; swaps3.444%,10.536%,3.335%,4.874% withallsignsagreeing. Removalerrors0.00518,0.01082,0.00536,0.00613allpass, butA2swapandthreefamilywritebarsfail. The10%swapbar isnotrelaxed. Exactnode signedmeanremovalCEisnegative inallfourfamilies (−0.01281,−0.00511,−0.04410,−0.01159): no claimthat itimplements ahelpfulmorphologyoperation. Resultsaredevelopmental,notOOD oridentifiedselectivecircuit.

The strongest unresolved methodological explanation isunconverged fitting undera coefficientmetric. ThepreparedamortizedPR+continuation isnowlive andtestsoptimizationprogress without datafit; failednativebarsremainunchanged. Noadditional node/rankselection onA2isintroduced. NativeU/RMS/tanh and exactcompiledwrite replay2.87e-15 exclude a simplemissingtail orfoldingerror as theexplanation.
