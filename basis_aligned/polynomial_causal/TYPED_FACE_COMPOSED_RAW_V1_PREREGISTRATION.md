# Three-state head8/head9 write: native end-to-end export

Freeze the CPU-tested composition, keeping recipient current value in head8.
Input current8[B,T,1152],donor-city8[B,1152],rawmixed9[B,T,1152],recipient/donor
token IDs,city index,destination mask. Program owns both head maps,20token table,
reflection coordinates and lambda90. It internally computes half the head8 face
and feeds that delta into the reduced head9 odd-value program. No model import,
first-values, initial-state tensor or externally supplied head8 delta.

Native/original/compiled on40opened prospective sequences:120forwards/300s.
pred_a: every final head9 delta relativeL2<=1e-4, finite, exactly120forwards.
pred_b: all original/compiled readout margins <=1e-5maxabs AND1e-6relativeF;
original native/midpoint anchor <=1e-5maxabs.
pred_c: each target/control effect error<=.001 with reference norm>=1e-6.
Null: composing the independently verified stages changes behavior beyond gates.
pred_d (separate isolated CPU): every final write<=1e-4relative against native
fixtures, zero strength gives zero write,unknown city rejected. Use python-I
with only exported code,weights and fixture files; no model or repo imports.

Price1,788,419float scalars including lambda90 plus20integer token indices;
74,880native-state floats atT32. Suffix after head9 remains external. This is
extraction at a wider boundary, not small causal interactions, full-model
sufficiency, new fresh prediction/selectivity or whole-model simplification.
