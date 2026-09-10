# Numerical correction of semantic-square controls; predictions unchanged

V1 is preserved as INVALID. Its16whole-MLP FP64/native absolute comparisons,
16exactsum/full write comparisons and16local/independent-prediction write
comparisons miss1e-3. Native/FP64 relative discrepancy is at most2.024e-7,
but absolute discrepancy reaches.004272 onlarge outputs. All final-logit
identity/equivalence checks pass, max2.146e-5. Two legacy-parent margin
comparisons also missrelative1e-5 atabsolute5.73e-6. V1's scientific outputs
have been opened; no scientific predicate or candidate change is licensed.

V2 retains the same35squares, directions, local/transport/inherited/full/identity
candidates, scienceB/C/D, native1e-3/1e-5 andFP64identity1e-9bars. No gain,
normalization, anchor orrepresentation selection. Changes concern ONLY controls:

1. Replay the actual deployed FP32 module with independent F.linear calls and
nativebias; compare to capturedsource atunchanged1e-3/1e-5. Separately record
r_i=m_i-[B64(n_i,n_i)+bias]. Require ||r_i||/||m_i||<=1e-5; do notlabel this
residual anexact-zero real-arithmetic identity.
2. Let r_known=r10+r01-r00 andr_mixed=r11-r_known. The independent threecorner
oracle isB64(a,a)+bias+r_known. It now compares to the UNCHANGED local candidate,
which already contains these receiving/native corner roundoffs through its
transport term. No r11/fourth input enters either prediction.
3. The exactsum control additionally includesr_mixed. That control, already
fourth-input-dependent and NEVER a prediction, reproducesnative m11. The
unchanged inherited scientific diagnostic is not replaced with this oracle.
4. Replay the original parent positivecontrol using its EXACT original row
batches and semantic-only readout: nativebase/donor,fullsourcebase/donor for
all4panels. Compare savedmarginandeffectnorms atoriginalbars. Also compare the
native all-token lm_head readout andsemantic-only readout in full-logit frame
at1e-3/1e-5. The old cross-batch smallmargin discrepancy is retained as a
DIAGNOSTIC, not a scientifically meaningful difference divided by a small
referenceeffect. This replaces those two ill-conditioned instrument assertions.
5. Audit replay of V1's UNCHANGED total/interaction arm margin effects andnorms
foridentity/full/transport/local/inherited atoriginalnumericbars. Report their
maximum drift. Outcomes cannot be improved by thecontrol correction.

Price adds16legacyforwardson288sequences:144forwards/1408sequences total,
0backwards/fits. Native FP32 formula calls for16corners are local module
recomputations, reported separately. All545902902nativeweights retained;
saving0. Same600secondwatchdog, managedqueue. Hash-bind V1runner/result/protocol
as well as this correction. Do not overwrite V1 orpresent it as valid.
