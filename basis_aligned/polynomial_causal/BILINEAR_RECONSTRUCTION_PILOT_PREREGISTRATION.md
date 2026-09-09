# Bounded bilinear reconstruction pilot — 2026-09-09

## Controlling success-criterion update (before pilot results)

The user appended a stronger interpretability criterion to the handoff. It supersedes
the earlier 25% engineering-cost gate below as a definition of scientific success:
recover a previously unspecified reusable computation; identify its inputs, explicit
operation and consumers; validate extraction and joint interventions on held-out cases;
reduce structural description cost counting adapters and remaining opaque parameters.
Bytes/MACs/latency remain secondary measurements. A numerical span, exact compiler,
quantization saving, or planted fixture is not a discovered circuit.

Phase C additionally checks the existing bilinear MLP for exactly shared linear factors
and unordered products (proportionality checked as exact dyadic-rational identities).
Canonicalize factors only with paired scaling accounted for. These are bounded local
common-factor proposals, not a global tensor decomposition or semantic identification.
If neither a valid contextual shared update nor a common-product candidate appears,
the decision is a local discovery null. Do not promote faithful recurrence execution
as meeting any missing discovery/extraction evidence. Full linear rank cannot rule
out a shorter nonlinear program.

Authority: explanations/bilinear_circuit_reconstruction_codex_handoff.md and the user's
subsequent direction to use the available GPU where it accelerates work. This replaces
the contrast-screen queue as Codex's active assignment. No training or provisioning.

## Frozen questions and gates

A. Direct and recurrent execution must agree modulewise and at every output coordinate
at atol=rtol=1e-9 on unit-scale FP64 fixtures. Use seeds 909, 910; B=2, L=2, H=2,
key/value width 4, residual width 16, MLP width 32, vocabulary 8, T=8,16,32.
Toggle RMSNorm and RoPE independently and together; retain nonzero independent residual
coefficients and nonzero shared-value mixing. Compare unedited, key removal, value
interchange, head removal, MLP removal, and joint edits with downstream recomputation.
Also compare direct cached decoding with recurrent tokenwise execution, and rational
no-normalization attention with exact integer/Fraction arithmetic. A failure stops promotion.

B. Search the column span of the exact cubic update coefficient matrix on four continuous
input variables (20 monomials), with 128 physical state coordinates. Use rational weights
and a valid dense Hadamard change of state coordinates, transforming readers consistently.
Planted shared inputs depend on two variables; a perturbation of size 2^-20 introduces a
third variable into one key factor. Independent routers with a shared value map are a
negative for router equality, not a negative for the universal cubic-feature baseline.
Certify candidate factorizations with rational matrix identities. Evaluate two readers,
fresh continuous inputs, and independent versus joint removal/interchange of head updates.
Discovery must distinguish natural-state closure from intervention-closed state. Positives
must recover the planted update functions up to an equivalent basis; perturbations must
invalidate the original exact reduction. Compare against elementary cubic-feature/CSE
execution and the separately labeled eight-token count baseline; no claim of new structure
is licensed by these positive fixtures alone.

C. If A/B are sound, use existing runs_hop/attn-mlp-attn-rms-seed0 via hop_ablate.load,
with its full token-to-logit model, unchanged checkpoint and actual normalizers/RoPE.
Use independent generator seeds 1909 (discovery), 1910 (IID), 1911 (longer-prefix shift).
Lengths 64,64,128 are inside its 240-token context. Freeze before holdout access a local
128-column update bank: first four coordinates of k1,k2,v from each of the first two heads
of the second attention layer. Numerical SVD is a proposal only. If full column rank at
the fixed relative singular threshold 1e-10, report a bounded linear-sharing null and do
not tune that threshold. If deficient, validate the frozen span on IID, longer prefixes,
and live upstream/consumer edits, rejecting relative update residual above 1e-9.
An observed span is never called a global identity. Full output distributions and centered
logit intervention differences validate the unreduced recurrence separately from discovery.

Useful reduction requires >=25% saving in constants, state or MACs with <=10% increase
in the other measures against the best relevant exact baseline. Any approximate candidate
must separately pass the handoff's KL and 1% intervention bars; no approximate search is
authorized merely by a failed exact span. If no eligible reduced candidate exists, record
reduced-program distribution/causal validation as not reached, not as passed.

## Resource and measurement contract

Keep candidate banks <=256 columns and every new analysis tensor <=256 MiB; no global
Hankel/Gram, no whole-model polynomial expansion. Enforce these tensor limits at allocation
and a 30-minute execution watchdog per pilot runner; this is a bounded first attempt, not
a replacement for the user's permission to use GPU. GPU execution only through enqueue.sh.
Exact rational work stays on CPU. Record compiler/discovery/verification/execution times,
torch/device/dtype, peak allocations, seeds, source/checkpoint hashes, domains and nulls.
Benchmark equal-weight direct and recurrent prefill and cached decode with warmups, five
timed repetitions, CUDA synchronization and separate state/constant/MAC prices.

Stop this discovery method if controls fail, the perturbation is falsely exact, or useful
sharing does not exceed elementary polynomial/CSE and count/table baselines. A faithful
compiler can be retained without being promoted as a learned circuit discovery.
