# Full-suffix test of generated city normalizers

Freeze CITY_MLP7_GENERATED_NORM_V1 after its CPU screen. No fitted norm correction:
estimate mixed8 RMS from other_city_sources, retaining MLP7 in all three reader
numerators, and generate both key RMS scalars from the resulting projections.
This removes three externally supplied city-side RMS scalars. Native query fields
and the two upstream state vectors remain required. MLP8 and the later suffix are
fully native in this test, not replaced by the earlier approximate response program.

All40opened Pile sequences, twenty independent source documents, six spelling
probes and four control readers. Arms: native, full native city removal, exact
generated-key-norm write with native input RMS, approximated-all-city-norm write,
sixteen same-site random edits norm-matched to the approximation per position.
Inject at attention8 output, recompute all remaining layers. Seeds
18091300+1000*k+context_id, paired seed/opposite cue signs.800forwards/300seconds.
No data filtering, refitting, new corpus claim or independent composition claim.

- a: native/full-reference scores replay CITY_FULL_STRENGTH_V1<=1e-4absolute AND
  <=1e-5relative; finite outputs, support zero outside mask, norm error<=1e-5,
  exactly800forwards. Native reader certificate must pass before execution.
- b: approximate target-effect relative error<=.35 against full native removal;
  target RMS>=1e-5; native capable contrasts>=90/120 (margin>=.1).
- c: every control-reader RMS<=.5 approximate target RMS.
- d: exact generated-key-norm arm replays full native removal<=1e-4absolute AND
  <=1e-5relative, and effect errors<=1e-3 separately for all five readers.
- e: approximate target RMS>=2 times median matched-null target RMS and beats16/16.
- f: positive attenuation>=.90 among capable contrasts, mean attenuation>=.02.

Report untouched-natural and substituted-arm prediction errors separately, without
new subgroup gates. Opposing predictions: the norm approximation preserves causal
prediction/selectivity; or omitted MLP7 norm terms produce amplified downstream
error/sign failures. Preserve failures. Passing this opened screen motivates fresh
validation; it does not close the remaining native query/upstream inputs.
