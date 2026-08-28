# Simultaneous attention and MLP component-bank identity

Date: 2026-08-28

All 18 attention programs and all 18 bilinear MLP programs execute simultaneously
through the explicit residual facade with every native component object replaced.
The role-free gate passes bitwise:

- all attention writes, v1 buses, and MLP writes have maximum absolute error 0;
- final logit error is 0 and native/program logit hashes are identical;
- both synthetic CEs equal 12.686808586120605;
- both 18-site transactions close ordered with exact block and v1 identity;
- literal native attention and MLP calls are zero;
- native, attention-bank, and MLP-bank tensor storage sets are mutually disjoint.

The complete dense 36-component denominator is 430,003,602 stored float32 values:
143,328,402 attention values and 286,675,200 MLP values. This is 78.77% of the
545,904,054-value exact model storage accounting.

The remaining 21.23% exact shell is explicit rather than explained away:

- 50,304 by 1,152 token embedding;
- 18 residual lambda pairs and x0 skip sequencing;
- parameter-free whole-state RMSNorm interfaces;
- independent 50,304 by 1,152 unembedding;
- parameter-free output softcap.

Thus the 36-component core is compositionally owned, but the complete model is not yet
a model-object-free tensor program. The next exact identity step is to clone the
embedding, residual scalars, and unembedding into an owned top-level program and execute
the RMSNorm/softcap shell directly. After that, no native checkpoint object should be
needed after construction.

This is still a dense identity point. Replacing dense attention by shared-QK-384 would
save 55,738,368 values, or 10.21% of complete stored model values, at approximately
99.4% attention-stake recovery. MLPs remain 52.51% of complete storage and are the
largest compression target.
