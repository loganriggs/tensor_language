# Sources of the native token-context gate

10 September2026. Original handoff/pilot. The token/norm score program still
requires contextual coefficients. Existing v185 source-versus-causal-response,
readout and middle-module dossiers are prior art; native module boundaries are
screening sites, not assumed semantic circuits.

For each row's fixed answer-minus-foil reader v, define k_v=2Q17(v)e and
a_v=v^T D17[(L17 e)*(R17 e)]. The context-only reader is k_perp=k_v-2a_v e;
tau=k_perp^T u17. Its contribution to the local response is delta*tau, with
the scalar/self contribution handled separately. This is the actual token
contrast, not the old two generic output readers. No fitted direction or rank.

Use existing A1/A2/G16 base contexts. Natural context donor for row i is the
next base row (i+1 mod16), same grammatical frame. Evaluate the recipient's
fixed reader on both inputs; do not change reader identity with the donor.
This isolates lexical context variation. It does NOT assert that the native
model prefers each primed verb over the next verb; no lexical behavior circuit
or new-text/OOD claim follows from this internal-gate screen.

One native base body/panel caches all36 attention/MLP final-position outputs.
Then one all-site own-cache no-op, one all-site cyclic restore and36 individual
cyclic module-output restores.39 bodies/panel,117forwards1872seq, <=900seconds.
Attention's separate first-value cache and other token positions remain live.
Later computations recompute normally. MLP17 output is downstream of its input
gate and must have exactly zero gate effect. All-chain restoration should
recover the cyclic input gate because the final embedding token matches.

Let d_tau be the cyclic natural gate difference and p_tau the patched gate
difference across the16 rows. Report transfer=<p_tau,d_tau>/||d_tau||^2 and
relative error=||p_tau-d_tau||/||d_tau||. Aggregate norms avoid dividing by tiny
individual row changes. Also report correct-token CE change under each restore;
G agreement is the collateral control. Target lexical patches are not required
to preserve original verb CE, and gate transfer is not behavioral recovery.

Predictions:

* A instrument:117/1872, existing native state/readout bridges; all48 base
  endpoints correct versus their grammatical foil; same final token within each
  cyclic pair; all three gate difference norms>1e-4; no-op logits maxabs<=1e-3
  and rel<=1e-5; all-chain gate error<=1e-3+1e-5*abs(reference) elementwise;
  MLP17-output gate change maxabs<=1e-7. Frozen hashes checked.
* B shared source nomination: A and at least one identical module-output site
  has gate transfer>=.50 and relative gate error<=.50 on BOTH A1/A2, with G
  mean absolute CE change<=.10. All36 sites reported; no top-site promotion,
  refit or post-hoc head subset. If B fails, reject singleton localization at
  this grain, not distributed production or a different causal variable.

As a weight-only map for a possible later producer test, fold k_perp into MLP16
products: c16=lambda17_0*k_perp^T*Down16. The exact raw-pre-MLP17 gate numerator
has this read of MLP16 products plus the other residual and attention terms;
divide the combined numerator by the actual RMS. Report coefficient norms,
not a causal share. This does not claim MLP16 is the source or repeat its closed
whole-layer approximation. All native weights remain, no model saving.

CPU continuation: paired intervals for nominated sites and G CE, including all
sites passing the prospective bars. Save the complete source screen; search
the corresponding module dossier before pursuing any nomination.
