# Keep the inherited interaction while sharing the branch generator

13 September 2026. The actual-weight CPU algebra control passes. This is a conditional executable extension, not yet a native-text behavioral result or a new compression ratio.

The previous local target kept the bilinear cross-product of separately generated child and remainder changes. The original-interaction coverage audit found that this omitted a substantial mixed response already created before block10. The exact three-vector MLP9 response lets us carry that term without fitting another model.

Let $a,b\in\mathbb R^{B\times T\times1}$ be removal fields along the same fixed head9 writer. Let $\Delta_9(a)$ be the complete residual-plus-normalized-MLP9 state change. Its three-vector representation and the five attention projection banks are prepared once per pristine context. Write $r_{10}$ for pristine attention10 raw input, $\lambda$ for its residual re-entry scale, and $A_{10}$ for the full native-form attention operator, including both QK factors, positions, values and normalization. The generated pre-MLP10 input is

$$
z_{10}(a)=z_{10}(0)+\lambda\Delta_9(a)
+A_{10}(r_{10}+\lambda\Delta_9(a))-A_{10}(r_{10}).
$$

For $z\in\mathbb R^{1152}$, $L,R\in\mathbb R^{4608\times1152}$, $D\in\mathbb R^{1152\times4608}$ and output bias $d$, define

$$
g_{10}(z)=z+\frac{D[(Lz)\odot(Rz)]}{\|z\|^2/1152+\epsilon_{32}}+d.
$$

Generate $h_N=g_{10}(z_{10}(0))$, $h_C=g_{10}(z_{10}(a))$, $h_R=g_{10}(z_{10}(b))$ and $h_P=g_{10}(z_{10}(a+b))$. Their state interaction is

$$
\mu_{10}=h_P-h_C-h_R+h_N.
$$

Using $\Delta_9(a+b)$ rather than $\Delta_9(a)+\Delta_9(b)$ retains the inherited two-vector mixed response. Attention and MLP10 then propagate and multiply it as required. No changed native branch state or oracle parent output is supplied to this generator.

The later suffix $F$ is nonlinear: the behavioral target is $F(h_P)-F(h_C)-F(h_R)+F(h_N)$, not $F(\mu_{10})$. All four suffix evaluations and their background dependencies must remain charged unless separately simplified.

## Executed control and price boundary

One synthetic17-position context, actual model weights, three nonzero signed/amplitude pairs and two zero-edit controls were frozen before execution. An independent reference recomputes MLP9 at the changed input, projects the changed attention input directly and executes MLP10. Maximum branch relative error is $2.28\times10^{-14}$; mixed-response error is $2.28\times10^{-14}$. Zero-edit cancellation residual is at most $4.55\times10^{-13}$. All registered bars pass; measured CPU execution is0.77seconds, excluding interpreter startup.

This shares preparation across three changed branches rather than two. It introduces no learned parameters, but retains dense attention/MLP maps, pristine context and separate branch execution. The control establishes algebraic correctness on its declared domain. Native FP32 text fidelity, original-interaction prediction and complete execution cost are still untested for this extension. Prior local FineWeb failures are not rescinded.

[Implementation](composed_joint_response_v1.py) · [Independent control](check_composed_joint_response_v1.py) · [Receipt](COMPOSED_JOINT_RESPONSE_V1_CONTROL.json) · [Coverage gap motivating it](COMPOSED_ORIGINAL_INTERACTION_COVERAGE_V1_MATH.md).


## 07:32 — Native original-interaction validation

The eight-prefix CPU pilot passed, followed by a preregistered managed GPU run on all160 historical prefixes. The GPU job terminated normally in9.16seconds, with all three registered criteria passing. The implementation uses the generated pre-MLP10 input rounded toFP32, then native MLP10 and the suffix. All three changed branches are generated; no changed native reference enters the predictor.

An executed cross-receipt audit confirms that all native no-edit/child/remainder/parent endpoint scores exactly equal the original17-arm experiment. Thus the following comparison uses identical native references:

| Regional group | Previous original-target error | Complete generated target error | Complete generated control error |
|---|---:|---:|---:|
|0|11.93%|0.0835%|0.1672%|
|1|12.16%|0.0647%|0.1300%|
|2|16.05%|0.0624%|0.2435%|
|3|14.57%|0.0375%|0.1312%|

FineWeb target errors are7.67%,0.446%,4.12% and1.41%; control errors5.47%,1.31%,2.32% and1.33%. All meet this experiment’s preregistered10%/15% group bars. This does not retroactively change failures of the previous local-product target. Maximum post10 state relative error is2.95e-7; maximum branch endpoint discrepancy8.59e-6.

The sign audit records no regional reversals and six FineWeb reversals: three target and three control. All reference magnitudes are below4e-6. None crosses the descriptive material threshold1e-5. Aggregate success therefore does not establish accurate signs for arbitrarily small effects.

What changed: the generated parent now contains the inherited MLP9 mixed response and its subsequent attention/MLP interactions. The predictor also generates the child and remainder used in its own final interaction calculation. The native comparison supports conditional computation/composition fidelity; it does not show that all relevant computation has been compressed or interpreted. The native downstream suffix still produces much of the effect. There is no new fresh text/OOD test, independent scalar-field extraction, or demonstrated additional speed/storage gain from this extension. Its third branch and all retained native weights must be priced.

Next decision: compare execution/storage of this shared representation against a matched direct three-branch computation, then simplify shared readers where their actual consumers justify the cost. Do not replace this now-valid original-interaction target with the easier local-product target.

[Full-panel receipt](COMPOSED_JOINT_NATIVE_FULL_V1_RESULT.json) · [Registered bars](COMPOSED_JOINT_NATIVE_FULL_V1_PREREGISTRATION.md) · [Aligned reference and sign audit](COMPOSED_JOINT_NATIVE_FULL_V1_AUDIT.json) · [Audit code](audit_composed_joint_native_v1.py).


## Complete-operator cost follow-up

Matched batched CPU execution, including preparation, gives1.07×/1.52×/1.63×speedup at1/4/12amplitude pairs. Local constant weights fall36.7%, but native weights remain needed for pristine context; no whole-model savings. The batched direct countercheck narrows earlier serial speed claims. [Exact scope, comparison and price](COMPOSED_JOINT_COST_V1_MATH.md).

The subsequent practical GPU comparison **fails speed adoption**: the validated mixed-precision generator is2.9–6.7×slower than native batchedFP32, and warm preparation reuse still leaves1.53–2.01×slowdown. State fidelity passes. See the cost note above; CPU timing is not a GPU speed claim.
