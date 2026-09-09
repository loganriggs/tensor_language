# Whole-model content versus contextual gates under entity renaming

Selected physical-state and V-only raw-field interchanges failed; no expanded
field is adopted. Exact frozen-payload lineage shows that the bare residual is
not the complete own-token payload. Test a different mathematical object: every
attention layer is linear in its payload when its native attention scores and
RMS gains are fixed. The full four-attention model therefore has logits T(G)E,
where G contains all four layers' native scores/gains and E is the embedding
sequence. The folded final reader is part of T and remains fully charged.

This identity alone is not a smaller or input-independent model: computing G
still executes the native nonlinear computation. Ask whether graph-isomorphic
entity renaming permits reuse of G, which would reveal a global reusable routing
operation instead of another selected local field. Do not confuse this with the
earlier local middle-entity match test or a latent-coordinate gauge rotation.

Fresh random-layout seeds27909/27910, first8worlds/population, all24 query entities
×hops0..3:16worlds1536 original/renamed pairs. IID24-cycle and OODthree8-cycle.
Use each world's fixed random24-cycle label map sigma. Rename every entity token
in bindingkeys, bindingvalues and the queryentity; keep Q/hop tokens and positions.
This transforms F into sigma F sigma^-1 and maps every correct answer by sigma.
No native-error filtering, fitting, layout selection or payload rescaling.

Execute four fixed arms:

* native = T(G_old) E_old;
* payload_only = T(G_old) E_new;
* gates_only = T(G_new) E_old;
* renamed_native = T(G_new) E_new.

Candidate uses the independent exported program for gates, embeddings and
execution. Original checkpoint is the native-reference oracle only. Factorial
payload, gate and interaction terms exactly sum to the native rename effect.
This is an explicitly paired-context interchange reference; it is not a claim
that recipient tokens alone generate the donor gates, or a smaller extraction.

A mechanical: focused graph/identity/native/factorial/live-term controls; all
old/new native outputs match original model max<=1e-9, relativeRMS<=1e-10 with
floor1e-6; exact factorial reconstruction<=1e-9; finite and everygroup nonempty.
Use all positions for numerical correspondence. Native original and renamed
task accuracy>=.8 perpopulation/hop is a separately reported capability license,
not a numerical-validity condition.
B full-distribution gate reuse: KL(native renamed || payload_only) mean<=1e-3
and95th percentile pertoken<=1e-2 over all positions, plus finalquery meanKL<=1e-3,
separately in each population. Task correctness cannot substitute for these bars.
C causal rename prediction: centered finalquery logit effect of payload_only
minus native predicts renamed_native minus native at relative RMS<=.01
(denominator floor1e-6), separately perpopulation/hop. Report full-vector errors.
D semantic gate/content separation: with all native capability groups licensed,
payload_only renamed-answer accuracy>=.8 for each population/hop; gates_only
retains original-answer accuracy within.05 of native and abs mean original-goldP
change<=.10 for each population/hop. Report payload/gate/interaction norms as
correlated terms, not additive explained-variance fractions.

A valid failure closes this global gate-reuse hypothesis; no selected layer/head,
normalizer, feature, rank or endpoint-map repair. Do not use favorable task scores
to promote quantitative gate invariance. All387968 constants and all four layers'
derived gates remain charged; report gate-cache bytes. No parameters removed.
B8FP64,1800s,<256MiB/tensor, managed GPU. Primitives first, integrate shared metrics
and fixed population builder, then execute once and preserve every outcome.
