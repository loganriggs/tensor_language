# Reusing a fixed writer across pristine contexts

13 September2026. The response basis has one globally fixed direction, $v_0=\lambda w$, and two context-dependent directions $v_1,v_2$. This permits exact sharing across contexts, not only across amplitudes in one context. The small-cache implementation improves reused six-product preparation in the CPU control. This is not yet a whole-branch or whole-model speedup.

For $K(x,y)=D[(Lx)\odot(Ry)+(Ly)\odot(Rx)]$, cache

$$
\ell_0=Lv_0,\qquad r_0=Rv_0,\qquad P_{00}=2D(\ell_0\odot r_0).
$$

Each new context projects only $v_1,v_2$ through Left/Right and prepares the remaining five product vectors. This uses the same actual writer and compatible inputs in every context. In particular, $P_{00}$ is computed once and has many consumers; similar learned vectors are not being equated.

An optional larger compilation uses

$$
J_{v_0}=D[\operatorname{diag}(Rv_0)L+\operatorname{diag}(Lv_0)R]
\in\mathbb R^{1152\times1152}.
$$

Then $P_{01}=J_{v_0}v_1$ and $P_{02}=J_{v_0}v_2$, leaving three private bilinear products. Count all new constants: the small cache stores11,520scalars including its writer; the compiled variant stores1,338,624. Neither eliminates the native Left/Right/Down weights.

## Executed controls and the layout countercheck

[Implementation](fixed_writer_products_v1.py) and [benchmark](check_fixed_writer_products_v1.py) use actual checkpoint weights and synthetic pristine contexts. The [first receipt](FIXED_WRITER_PRODUCTS_V1_CONTROL.json) passed algebraic checks but failed the speed criteria for batches16and128. The varying-direction slice was non-contiguous. This suggested a batching/layout problem rather than a failed algebraic saving.

The [separate packed-input receipt](FIXED_WRITER_PRODUCTS_PACKED_V1_CONTROL.json) includes the cost of making that small slice contiguous. It preserves the first failure and changes no mathematics. Seven interleaved FP64 two-thread CPU timings give:

| Context batch | Original six-product preparation | Small cache reused | Dense compilation reused |
|---|---:|---:|---:|
|1|6.96ms|5.05ms|5.24ms|
|16|23.17ms|16.08ms|14.03ms|
|128|158.37ms|105.15ms|87.21ms|

The small-cache reused speedups are1.38,1.44,1.51times, passing the registered10%bar in each case. Contiguous packing is now the helper's default. The larger compilation is slower at batch1, so its all-batch speed criterion still fails; it improves larger batches by15–21%over the small cache.

Measured one-time setup is5.09ms for the small cache and144.64ms for the dense compilation. These are excluded from the reused timings above and must be paid. Relative to the small cache, the dense map's measured setup break-even is roughly1000–1100contexts for the larger batches, and no break-even exists at batch1 in these measurements. Setup was timed once, so these estimates are descriptive. A single context does not justify either precomputation purely on execution time.

The operation counts support the intended saving. With residual dimension$d$and hidden width$m$, the original preparation needs approximately$12dm$main multiply-accumulates per context. The small cache needs$9dm$; dense compilation needs$7dm+2d^2$, after paying its global compilation. Here$m=4d$. Copies, elementwise products, support checks and memory traffic still matter, as the initial timing failure demonstrates.

An additional [per-product audit](FIXED_WRITER_PRODUCTS_COMPONENT_V1_AUDIT.json) prevents large private products from hiding errors in the fixed-writer terms. On independent private vectors with actual weights, all six product errors are at most$3.93\times10^{-15}$. Thus the global aggregate agreement is not masking inaccurate cross terms.

## Interpretation

Prefer the small cache for repeated product preparation. Keep the dense compilation optional until its extra storage and setup are justified by the workload. This is exact computational reuse across contexts of one fixed-writer circuit interface; it does not establish reuse across semantic tasks or reduce the native MLP's stored weights. The previous full five-bank branch still retains its expensive attention/background contractions, so its failed whole-branch price remains unchanged. The same fixed-writer sharing can next be folded directly into five-bank preparation, with a comparison against the already optimized five-bank baseline.
