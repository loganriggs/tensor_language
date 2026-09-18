# Full-model CPU check of the earlier residual6 boundary

Compose the attention7 generator, folded MLP7 readers and exact generated-key-RMS
city-write program. Inputs: residual6 prefix through city, token IDs, native
head8 rotated query fields, native mixed8 city RMS, destination mask. Initial
states and inherited values are generated from the same frozen token tables.
No normalized MLP7 input or native attention7 output is supplied to the wrapper.

This is a new earlier-boundary CPU certificate, not a replacement/restart of the
queued GPU tests at their older boundaries. It does not adopt the mixed8 RMS
approximation. Count all required weights and supplied state floats; fewer arrays
does not mean lower information or storage. Native prefix0–6 and native query
generation remain external, as does MLP8 and the full later suffix.

Evaluate all40opened sequences in two right-padded CPU batches: native and
installed exact city removal at attention8 output. Recompute all18blocks and
read original final positions. Two CPU threads,180second cap, no CUDA initialized.
Price80sequence-equivalent full-body forwards,36batched block calls. Compare
to saved GPU native/fullcity-removal reference from CITY_FULL_STRENGTH_V1.

- a: native CPU scores match GPU reference<=1e-4absolute AND<=1e-5relative.
- b: installed CPU scores match native fullcity removal<=1e-4absolute AND
  <=1e-5relative; per-reader effect error<=1e-3 for target and four controls.
- c: composed local write<=1e-4relative to native fullcity write per sequence,
  support zero outside mask, finite outputs,40fixtures/80sequence-equivalents,
  exactly36batched block calls and CUDA uninitialized.

Save all scores and local writes. Passing is opened native-implementation evidence,
not new OOD/selectivity-null confirmation, source independence or full input closure.
