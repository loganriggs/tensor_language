# Install the folded-reader city write in the native model

Use city_mlp7_integrated_v1.execute and the frozen CPU artifact. This computes
the complete city-value removal write from the folded MLP7 readers, without
materializing MLP7's1152-dimensional output. The normalized MLP7 city input is
reconstructed from saved block7 mixed/attention sources; the direct native-input
reader certificate CITY_MLP7_READERS_V1 remains a separate required check.

Declared inputs: normalized MLP7 city vector[1,1152], other city-source sum[1,1152],
lambda8 scalar, native mixed8 RMS[1,1], two native key RMS[1,2,1], two rotated
native query factors[1,T,2,128], inherited city value[1,128], city index and mask.
Query/context/normalization generation remains external. Moving one reader
calculation backward is not closure of the full regional executor's native ports.

Install the frozen complete-removal delta at attention8 output and recompute
MLP8 and all later layers. Compare to complete native removal in
CITY_FULL_STRENGTH_V1. All40opened sequences,240probes,four controls; native and
folded arms only:80bodyforwards,120seconds. No new fitting, selection or OOD claim.

- a: unedited scores replay<=1e-5absolute AND relative; finite all outputs and
  exactly80forwards. Frozen CPU write error<=1e-4 on every sequence.
- b: folded complete-removal scores replay native removal<=1e-4absolute AND
  <=1e-5relative. Relative intervention-effect error<=1e-3 separately for the
  spelling reader and each of four control readers.
- c: prerequisite direct native-input reader certificate a/b/c all pass; folded
  reader program remains12,386,688FP32scalars versus16,368,768unfolded.

This certifies an opened implementation at the declared native-context boundary,
not fresh generalization, selective-source confirmation or independent composition.
Keep the source random-split failure. A failed installed replay prevents adoption
even if the local reader certificate passes.
