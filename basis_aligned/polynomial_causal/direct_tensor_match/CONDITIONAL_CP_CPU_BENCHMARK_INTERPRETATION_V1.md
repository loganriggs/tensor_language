# Exported graph simplification yields a measured CPU speedup

22 September2026,01:48UTC. AMD Ryzen7 7800X3D, twoTorchthreads, float32 eager, warmweights/inputs, sevenrounds withrotating/reversedprogramorder. Batchsizes1/64/2048; diskcapture/fixedwriter/vocabulary/finalnorm excluded equally. Sharedinstance microbenchmark, notGPU orwhole-transformer speedup.

| Program | Batch1 speedup, seeds1001/1002 | Batch64 | Batch2048 |
|---|---:|---:|---:|
| Fullconditional256 |1.45/1.47x|0.98/0.99x|1.11/1.11x|
| Leanconditional256 |1.75/1.75x|2.44/2.46x|1.48/1.49x|

Forseed1001 theCPparent costs152.21/21.47/19.45microsecondsperstate atbatch1/64/2048; lean25687.11/8.78/13.16. Rank64and128arefasterbutlessaccurate. Nonmonotonicper-statecostacrossbatchsizes reflectsactualcache/kernel/temporary-allocation behavior, notarithmetical flopcount alone. Repetitionmax/minratiosforreportedprograms are~1.01–1.08. This is a seven-rounddescriptivebenchmark, not formalpairedlatencyconfidenceinterval.

The fullconditionalprogram's addedpairproducts andtemporaryarrays consume muchofits theoreticallinear-operation saving. Deletingfouradditionalpairproducts peratom producesa cleareractualspeedup. Thegain concerns anapproximatelyreconstructed16outputpolynomial, so accuracy/effecttestsremainseparate.

Files: benchmark_conditional_cp_cpu.py andCONDITIONAL_CP_CPU_BENCHMARK_V1.json. Noqueued/livehelpersmodified.
