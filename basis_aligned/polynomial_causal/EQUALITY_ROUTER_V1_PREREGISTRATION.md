# Shared equality routing: prospective small-model falsifier

Registered 2026-09-09 13:48 UTC, before any candidate outcome. Controlling authority:
`explanations/bilinear_circuit_reconstruction_codex_handoff.md`, including its final
structural criterion. This follows the completed exact local-span null; it does not rerun it.

## Hypothesis and object

The first attention layer of `runs_hop/attn-mlp-attn-rms-seed0/model.pt` implements
approximately entity-renaming-invariant routing. Its four heads consume a shared predicate:
entity equality, entity/non-entity type, or the identity of one of five special tokens.
For query token a, source token b, causal lag d, define the native product-score kernel
K[h,d,a,b] from normalized embeddings, all four Q/K matrices, and real rotary tables.
No task answer, text outcome, or head selection enters candidate generation.

The 29x29 ordered token pairs have 37 orbits under simultaneous permutations of the
24 entity names: equal entities, different entities, 5 entity-to-special, 5
special-to-entity, and 25 special-to-special. Replace each kernel value with its
orbit mean. This is the unique least-squares invariant projection under uniform
token-pair weighting, a fixed hypothesis rather than an unconstrained nonlinear adapter.
All 240 supported lags are compiled from weights; new lags are not held out.

The executable stores 4x240x37 scalar coefficients and one explicit orbit predicate.
It removes four 128x128 Q/K matrices only at layer 0, retaining embeddings, live
normalization/value production, W_O, residual mixing, the MLP, later attention and
the full 29-way output head. Every retained parameter counts. This is a partial
extraction with explicit native background. The plain 29x29 table is a control,
not a discovery. A positive here does not establish a circuit in bilin18.

## Controls and frozen populations

CPU fixtures before checkpoint outcomes: exact orbit-constant positive, a one-entry
perturbation that must be rejected as exact, independent consumer kernels with the
same values, and equivariance under entity permutation. A generic non-invariant
kernel must retain a nonzero residual. Distinct head coefficients stay distinct:
sharing the predicate does not imply sharing the entire router or editable state.

Checkpoint SHA256 must match the completed pilot. Compare an exact first-layer
replacement using the native pattern against the original full model (FP64
atol=rtol=1e-9). The unreduced lag table is a separate positional reassociation
control, at atol=rtol=1e-4 on full logits, since original RoPE tables were generated
in FP32. If this control fails, no symmetry conclusion is licensed from its errors.
Never freeze native activations or normalizers across edits.

Populations, 16 documents each, batch <=4, full 240-token documents (239 input positions):
- IID: fresh random 24-cycles, seed 2909, original generator.
- Metamorphic: the same documents under a fixed entity permutation, seed 2910;
  this is an IID-distribution symmetry control, explicitly not a new OOD distribution.
- OOD: random permutations allowing short cycles/fixed points, seed 2911, with
  answers recomputed from that function. No documents filtered on model correctness.
Report all input-position distributions plus the 48 query answer positions separately.
The model's accuracy on the task is descriptive; teacher fidelity is the target.

## Interventions and gates

For each population compare native, unreduced lag table, and invariant candidate.
Use the identical native/compiled intervention map: at layer 0 delete attention
edges between equal entity-token pairs for head 0, head 1, or both. All other
edges and heads stay live. This edits pattern entries, not an arbitrary neuron.
Report removal effects and the joint interaction vector, with full downstream
recomputation. No assumption of additivity or selective private removal.

pred_a_instrument: all fixtures plus exact reference and lag-table controls pass.
pred_b_distribution: on every population and each of four arms, mean full teacher
KL <=1e-3 nats/token and p99 <=1e-2, both all-position and query-only.
pred_c_interventions: centered-logit effect relative L2 <=0.01 for head 0, head 1,
and joint removals, and their interaction; if target RMS <1e-6 use error RMS <=1e-8.
pred_d_live_reuse: removing the equality edges changes centered-logit RMS by >1e-6
for each head separately; all other gates and structural constants decrease must
hold before calling this a shared useful operation. No neuron identity is inferred.

Null: arbitrary token identity or contextual cancellation is essential; projected
router fails output or removal fidelity. Stop this fixed projection hypothesis on
failure. Do not relax thresholds, select a passing subset of documents/heads, or
relabel the full table baseline as a circuit discovery. If only positional control
fails, repair that concrete instrument before interpreting symmetry.

Price: no training; first-layer 240x4x29x29 bank (<7 MiB FP64), candidate bank37
columns, each new analysis tensor <256 MiB; batch <=4; 1800-second watchdog.
GPU through bqrunner only. Record code/checkpoint hashes, complete constants,
kernel residuals, latency and native-background scope. This screen is a next step
toward all four properties, not satisfaction of the full model-level goal.
