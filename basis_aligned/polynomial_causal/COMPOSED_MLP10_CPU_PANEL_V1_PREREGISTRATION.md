# Full existing-panel integrated prediction on CPU

13 September2026, registered after the two-prefix CPU pilot but before remaining158-prefix outcomes. This is confirmation on the existing panel, not fresh OOD. Rows0 and96 are known pilot examples and are included explicitly, not presented as unseen.

Use all96regional and64FineWeb rows in their fixed order. Execute pristine blocks0–8 once per prefix, direct head9 child/remainder edits throughMLP9/attention10, and three native suffix11–17 readouts at the same additive postMLP10 background: no product, observed-input product, generated-input product. Frozen GPU-derived scalar fields are supplied to both branches. Rebuild the generator's attention changes and joint normalization independently of observed interventions.

- pred_a: generated versus observed-product target effect relativeL2 <=2% in each of four regional groups.
- pred_b: each prefix child and remainder input relative errors <=1%, and joint-normalization relative error <=0.1%. This per-prefix bar is stronger than the pending GPU panel aggregate bar and does not redefine that registration.

Report both endpoints and all sign/zero discrepancies. FineWeb effect errors and unrelated regional token-margin errors are descriptive; include absolute units and preserve weak results. No precision-floor rescue without a discriminator. These CPU results do not score pending GPU registrations or prove CPU/GPU numerical equivalence.

Price:160pristine9-block prefixes,480branch MLP9/attention10 computations including pristine,480suffix evaluations,480generator FP64attention evaluations plus product computations. TwoCPUthreads,CUDAhidden,300secondbound. No new weights or large caches. CPU was selected from measured1.64second two-prefix pilot against a managedGPU queue delay of tensofminutes.

Runner SHA256: d1dcdfa9cc95873f81ad97f2bda2a49494ab822d0c08207603df0b60a9a54b42
