# Minimax deleted-row support allocation

September14 10:26UTC. Fixed-support coefficient changes cannot materially
improve the output-unfolding spectral certificate: current upper1010.8542,
fixed-mask lower1010.6341. This does not rule out native-specific improvements.

Keep the same stored output/head frames, bitmap format and occupied count.
Transform exact T into these literal invertible frames. Within each output
row, retain largest absolute core entries. Allocate per-row retained counts
to minimize the maximum deleted-row squared norm: binary-search the common
cap, then spend any remaining integer entries on the largest current residual
row. No synthetic or native activation statistics, labels or task-loss fitting.

A: frame replay<=1e-10, identical occupied count/frames and byteswithin1%.
B: actual physical output-unfolding spectral error reduces>=10% vs original.
C: relative coefficient Frobenius error<=.12. Record row energies and lower
bound gap; FP32 storage and existing CSR executor. CPU120seconds/two threads.
This is a distribution-free absolute error certificate, not a guarantee of
small relative native effects. Freeze artifact before native validation.
