# Mathematical review: shared producers, consumer ports, and selective abstraction

Due 10:49 UTC; begun 10:49:10. Original bilinear handoff and pilot govern,
including the revised structural-discovery criterion. R594 is a valid failed
selective-factor screen; its criteria and verdict remain unchanged. The goal is
OOD prediction, independent extraction, selective manipulation, composition/reuse,
and a simpler executable description. No trained circuit is promoted here.

The current object is the fixed equality-supported attention write
`B(E,U)=sum_r E_r U_r`, where E contains native continuous product-attention
scores restricted by preceding-token/query equality, and U contains native
output-projected mixed values. Four zero-index heads L5H5/L7H3/L8H3/L8H4 are
edited at the last query. U has residual width1152; head width128; the native
suffix retains all18 blocks, 9heads/block, MLP width4608 and vocabulary50304.
The same-layer L8 changes are one physical addition. A centered edit adds
`B(E_d,U_b)-B(E_b,U_b)`, its payload counterpart, or the joint difference to
the recipient's live output. This is a partial output intervention, not a
replacement of all native QK or value states.

B is bilinear in independently supplied E and U. Native score generation is
the product of two normalized QK dot products; normalization, rounded RoPE,
shared-first-value mixing, residual re-entry, and final softcapping remain
explicit. There is no global low-degree polynomial in raw token embeddings.
Rescaling E and inversely rescaling U preserves B algebraically, but need not
preserve the physical intervention map. Natural ties between consumers also
need not survive consumer-specific edits. The relevant domain includes the
registered donor/recipient contexts and edited states, not just natural states.

R594 used639 FIT forwards/20,448 sequences in97.0543 wrapper seconds. All
545902902 native parameters remain; structural saving0. It tested task margin,
correct-token cross-entropy (CE, in nats), full-vocabulary root-mean-square
logit changes, and active controls. Mean target recovery is a ratio of mean
signed effects to mean native effects. No SELECT, FINAL or OOD was opened.
The raw 6,560,733,847bytes are RAM-backed; compact results and hashes are durable.

The original pilot already establishes exact accumulator closure and the
distinction between shared update computation and separate editable histories.
The09:10–10:20 work also rules out several weak prefix and single-value-branch
explanations. Another rank scan or exact accumulator fixture would not answer
the present uncertainty: does the failed joint test imply bad algebra, or an
incorrect claim about which downstream computations the edited quantity serves?

[Beckers and Halpern, Abstracting Causal Models](https://arxiv.org/abs/1812.03789)
formalize consistency between low- and high-level models over specified
interventions. Our native executor is the low-level model, the proposed circuit
is the high-level model, and a physical patch requires a corresponding abstract
edit. Agreement on an answer alone does not specify that correspondence. This
is a definition and consistency framework, not a tensor-discovery algorithm or
a uniqueness guarantee for learned circuit coordinates. Finite exhaustive
checking costs the number of contexts times interventions times execution cost;
that becomes prohibitive for the full transformer.

[Massidda et al., Causal Abstraction with Soft Interventions](https://arxiv.org/html/2211.12270v1),
Definitions 4, 9 and 12, and Theorem 2, extend the treatment to mechanism changes.
They distinguish agreement on generated states from stronger consistency over
endogenous settings and derive an explicit intervention map for constructive
soft abstractions. This is relevant to our live output additions. Applying
their guarantee would require a specified surjective state map, admissible
mechanism changes, and the required consistency identities. We have not
established those assumptions for R594, particularly for donor-dependent edits
whose native factors retain context. The paper supplies no automatic search
for our shared computation. Conditioning on recorded donors and explicitly
declaring their inputs is necessary before such a mapping can be assessed.

[CLUE](https://arxiv.org/abs/2004.11961) provides a different exact route:
constrained linear lumping of polynomial differential equations. Its polynomial
vector field and invariant-Jacobian-subspace object is not our normalized,
layer-dependent discrete transition with donor edits. We therefore retain its
closure principle from the original pilot rather than claim its minimality
algorithm solves the transformer. Tensor recurrence, polynomial identity
checking and linear observability address execution identities; none alone
identifies the permitted consumer-specific intervention.

The executable consequence is a falsifier of an overly broad inference from
R594. Let selector s and payload assignment p be Boolean. Define

    E(s) = [1-s, s],  U(p) = [[p, 0], [1-p, 1]],
    B(E(s),U(p)) = [y,c] = [(1-s)p+s(1-p), s],
    output = [y,c,y*c].

The first consumer copies the selected payload. The second reads selector
context. The third reuses both outputs. Every factor interchange implements
the exact intended computation. Yet `(s,p)->(1-s,1-p)` preserves y while
changing c by one. Thus perfect factorization and perfect target transfer are
compatible with a failed answer-preserving full-output invariance test.
This is not evidence that the trained model has this explanation; it proves
that the failed test cannot distinguish this explanation from contaminated
factors by itself. The stronger selective-circuit claim still fails.

Further, `(s,p)=(0,0)` and `(1,1)` both give y=0 but different c. No decoder
of y alone can preserve the full output on even these two inputs. Dropping
context is therefore insufficient. A consumer-port model instead computes
`y=(1-s_target)p+s_target(1-p), c=s_context`, then recomputes y*c. Naturally
the selector ports coincide. An edit of the shared producer changes both;
an edit of its target-consumer port changes only s_target. These are different
registered operations. Keeping the third consumer fixed after changing y
would itself be an additional intervention, not ordinary downstream execution.
This illustrates a within-module split plus shared reuse, with every adapter
and consumer port explicit. It does not discover that split in native weights.

`shared_selector_intervention_counterexample_v1.py` was executed on CPU using
exact integers:4 natural worlds,48 factor-interchange cells,64 consumer-port
cells, and all4 diagonal counterexamples pass. The two-input decoder collision
is checked. The source hash is bound in
`SHARED_SELECTOR_INTERVENTION_COUNTEREXAMPLE_V1_RESULT.json`. No model calls,
fitting, rank choice or relaxed scientific threshold. This hand-specified toy
has constant-size arithmetic; reuse of y is explicit, but it is not a newly
recovered computation or a structural-cost win against a competent baseline.

Decision: preserve the R594 null and the strong target-transfer evidence.
The next native candidate must name an additional consumer or distinguish
producer and consumer-port edits, with predicted full-output consequences.
Support for shared computation would require one independently generated
quantity to predict both consumers and their separate/joint edits on held-out
cases. Failure of that prediction rejects the proposed sharing. A projection
chosen only to hide vocabulary changes or improve the original answer margin
would not meet this requirement. The cheap exact counterexample resolves the
logical ambiguity before another costly native screen; it does not justify
rerunning R594 with weaker invariance or control bars.

Actual continuation receipts: R594 raw arithmetic/scoring audit executed after
the native result, followed by this claimed and executed mathematical falsifier.
Next hourly review11:14 UTC; next mathematical review13:49 UTC. Full goal active.
