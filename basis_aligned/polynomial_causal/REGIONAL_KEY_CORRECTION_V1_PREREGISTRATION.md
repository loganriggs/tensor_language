# Joint QK correction: mixed numerator terms and normalization

Freeze the original forward13 support and all48 geographic rows. Retained raw
key input r0 and native full prefix input r give p_i=K_i r0 and delta_i=K_i(r-r0).
Keep both factors together. With native normalized/rotated queries, the score
numerator is (a+da)(b+db)=ab+a db+b da+da db. Native unrotated key projections
set D=128² sqrt(mean(p1²)+eps rho²) sqrt(mean(p2²)+eps rho²), including actual
input RMS epsilon. Rotate numerators with actual rounded native positional maps.

Seven branch arms: baseline, native firstswap, retained-support firstswap,
true-denominator-only correction, true-numerator-only correction,
retained+mixed numerator with true denominator, complete correction.

A: native producer/attention/first-branch replay and complete-correction write
replay <=1e-5, exact helper product control <=1e-12. Complete correction does
not adopt the full native key generator as a small independent circuit.
B: A plus denominator-only write and regional signed-prefix effect errors
<=.1 in EACH family. C: A plus retained+mixed numerator/true-denominator write
and effect errors <=.1 in EACH family. These are distinct structural predictions.
The omitted×omitted term is excluded only in C. Numerator-only is diagnostic.
Report signed transfer and unrelated effects; these are attribution tests,
not new selective-circuit promotion. Preserve the earlier forward13 OOD miss.

Null: neither proposed restricted correction suffices. No fitted scalar or
support change; all raw prefix ports/queries/downstream states remain native.
Terms are algebraic contributions to ONE joint QK computation, not QK1/QK2 tasks.
Price10native batches48rows,7suffixarms,180secmanagedcap,<4MBartifact. CPU exact
helper identity precedes native execution. No new data-dependent factor fitting.
