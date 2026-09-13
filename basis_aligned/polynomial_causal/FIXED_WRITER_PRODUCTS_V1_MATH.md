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

## Combined five-bank preparation and one shared node, 13 September 09:17

That combination is now implemented in [cached_diagonal_products_v1.py](cached_diagonal_products_v1.py). It uses the fixed writer's cached Left/Right projections, constructs the four varying conic-bank vectors before Down, and reuses $P_{00}$. It is compared against the existing direct five-bank preparation, not the older six-bank baseline.

[Matched benchmark](check_cached_diagonal_products_v1.py), [receipt](CACHED_DIAGONAL_PRODUCTS_V1_CONTROL.json): reused speedups are1.33,1.27,1.35times at context batches1,16,128. Each vector is checked separately, with maximum error $3.06\times10^{-15}$. One-time setup costs4.86ms. Including that first setup loses at batches1and16, while batch128 remains1.28times faster. Measured batched amortization is about19contexts. These are local CPU preparation results, not a new full-branch speed claim.

The first version still copied the constant vector into every bank. `prepare_split` now returns one shared fixed vector and four varying vectors per context. `execute_split` multiplies the shared vector by each context's coefficient and adds the four private contributions. The [shared-node control](CACHED_DIAGONAL_SHARED_NODE_V1_CONTROL.json) verifies the fixed vector actually aliases the global cache rather than duplicating storage;128actual-weight synthetic contexts with different amplitudes replay the previous five-bank output within $9.36\times10^{-17}$.

Including **all11,520global-cache scalars**, stored bank-plus-cache size is

$$
4Nd+11520\quad\text{versus}\quad5Nd,\qquad d=1152.
$$

At16,128,1024contexts the savings are7.5%,18.44%,19.80%, respectively. At one context it costs more; storage breaks even at10contexts. The benchmark above measures the expanded-output preparation wrapper; the split-bank has a separate exactness/storage check, so no additional split-executor timing gain is claimed.

This is now a literal shared computation graph: one fixed self-product node serves multiple context-specific branches, with four varying vectors per context. Native matrices, response generators, attention/background processing and suffix remain required. It improves this repeated product-bank representation without changing the earlier conclusion that the complete branch schedule has not beaten direct native MLP execution.

## Shared-node removal and port-conditioned extraction, 13 September 09:23

The shared fixed node is behaviorally material. A managed test zeros $Q_0=P_{00}$ in the child, remainder and parent generated branches while retaining all other terms. In each branch this removes exactly

$$
\frac{a^2}{2\rho_{10}}P_{00}
=\frac{a^2}{\rho_{10}}D_{10}[(L_{10}\lambda w)\odot(R_{10}\lambda w)].
$$

This is the squared direct residual-writer path; it is not removal of the entire writer or every product involving it. The [native receipt](FIXED_SELF_NODE_REMOVAL_V1_RESULT.json) verifies this local deletion against direct MLP algebra within $7.59\times10^{-16}$ and replays native reference corners exactly. It takes12.61seconds on160historical prefixes. Both preservation criteria fail, so the node must not be pruned on this evidence.

[The aligned effect audit](FIXED_SELF_NODE_REMOVAL_V1_AUDIT.json) measures changes relative to the unpruned five-bank program. Joint target effects change by8.28–18.43%of the native interaction norm on regional groups and18.49–33.44%on FineWeb groups. Regional controls also change by13.23–15.35%; this does not identify a selective spelling component. Pruned regional predictions introduce nine material sign reversals across target/control endpoints. Effects are far larger than the verified numerical replay discrepancy.

This is evidence for a useful shared computational primitive, not evidence that it is an independent semantic circuit. Its fixed output direction is now [exported with a checkpoint-free executor](extracted_circuits/fixed_writer_self_mlp10_v1/README.md). The package stores1152FP64scalars (10,793serialized bytes) and takes amplitude plus the actual whole-branch normalizer as inputs. [Extraction control](FIXED_SELF_NODE_EXTRACTION_V1_CONTROL.json) matches direct actual-weight computation within $3.32\times10^{-15}$; the local child/remainder/parent combination matches separate calls within $1.25\times10^{-16}$.

Those supplied ports are substantial dependencies. The package does not generate text-dependent amplitudes, normalization, background or the nonlinear suffix. Combining its local branch differences is exact, but it does not replace separate suffix evaluations when predicting a joint behavioral effect. Fresh/OOD evidence and selective consumer interventions remain outstanding.

The subsequent [signed-effect audit](FIXED_SELF_NODE_SIGNED_V1_AUDIT.json) cautions against calling this a supportive spelling feature. Its removal-defined contribution projects negatively onto the regional target interaction in all four groups (cosines−0.475,−0.680,−0.136,−0.867), while aligning positively with controls (0.842–0.962). It therefore participates in compensation rather than simply scaling the target effect. These descriptive projections are not additive causal percentages; FineWeb target alignment also varies. The arithmetic specification is stronger than any semantic label currently justified.
