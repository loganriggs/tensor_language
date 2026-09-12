# Native mixed-precision source-reader comparison

Same branch,48rows and failure-capture instrumentation. Retain original FP64
four source readers and dual rows. Build every vocabulary first-input reading
using native GPU embedding/reentry/RMS, then FP64 dot products. Runtime computes
four source readings and dual mixture inFP64, casts that mixture toFP32, and
retains FP32 large maps/remaining contractions. No learned correction.
A oldFP32/fullprior replay write<=1e-5/effect<=1e-3 EACHfamily; B mixedprogram
same. C child-reference<=1e-5 and partition<=1e-6, unchanged. FullFP32C remains
failed; this distinct candidate must earn its own verdict. Save worstcase.
Price:50304token initializations(no body) plus9bodybatches48rows13–19tokens,
7suffixarms,180sec cap,<5MB output plus~4.3MB mixedprogram. No corpus fitting.
Current/query generators, positional maps andfinal suffix remain external.
