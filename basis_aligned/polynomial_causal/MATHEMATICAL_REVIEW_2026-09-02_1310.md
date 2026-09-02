# Mathematical review — 2026-09-02 13:10 UTC (three-hourly)

Context: since the 1010 review the program delivered §2606 (the equality
query channel is one-dimensional per position; frame differences are scalar
magnitude bookkeeping), a three-site T/I similarity spectrum
(−.62 / .64 / .9998), four consecutive rungs of near-perfect descriptive
profile stability with failed operational selection, and the death of both
categorical context tables (token 485, bigram 486). The right mathematics
this cycle is about SCALAR RECOMBINATION LAWS and SITE-GRADED VARIABLE
IDENTITY — both now measured below.

## Ranked moves

### 1. Single-index scalar-gain law for multi-site composition — EXECUTED
**Object.** The 7-subset per-token Möbius effects of the query-MLP trio
(474 per-token bundle), reread through §2606: if each site contributes a
scalar magnitude m_i to ONE shared channel and CE is a per-token monotone
readout h_t(Σ m_i), then all three pair interactions at a token must share
a sign (curvature of h_t) — a parameter-free NECESSARY condition
(single-index models, Ichimura 1993; order-consistency of additive
indices).
**Measured (474 bundle, per token, null = 25% under independent signs):**
sign-coherence raw .458–.604 across the six window×source cells;
above-median-magnitude tokens .479–.760; strongest exactly where
composition is frame-stable (code-N: .604/.760), weakest on natural
(wave1: .458/.479). Binomial z ≥ 6 against the independence null in every
cell. The MULTIPLICATIVE-IN-CE alternative (interaction = product of
singleton effects, log-linear models в la Bishop–Fienberg–Holland) is DEAD:
corr(I_ij, y_i·y_j) ∈ [−.09, +.35], mostly ≈ 0.
**Reading, honestly bounded:** the single-index law is a dominant tendency,
not a theorem — 2× null everywhere, far from the ~100% a clean law gives;
its strength ORDERS with register exactly as the frame-stability results
do. **Cheapest next falsifier (registered here):** the sufficient-condition
version — per-token LP feasibility of an additive index reproducing the
full 8-subset rank order (order-polytope membership); CPU-only from the
same bundle; a feasibility rate near the sign-coherence rate confirms the
law's scope, near zero kills it despite the screen.

### 2. Site-graded variable identity along the consumer filtration — EXECUTED
**Object.** The T/I relation that neither D-shared nor D-split could
license in rungs 483–486.
**Operational definition.** Decompose the (T,I) response pair into common
(T+I) and difference (T−I) energy per consumer from 483's stored Gram
matrices: share_common = (G_TT+G_II+2G_TI)/2(G_TT+G_II).
**Measured (exact-removal frame; halves agree to ~1%):**
attention1 .507/.508 — a 50/50 read (the −.62 profile anti-alignment and
.015 cosine restated: attention1 reads the DIFFERENCE as strongly as the
sum); mlp1_direct .623/.628; mlp1_total .789/.791 — a MONOTONE GRADIENT:
the further downstream the observation, the more T and I are one variable
(3.7:1 common:difference by the composed path). Tangent frame shows the
same ordering (.65/.58/.78).
**Consequence.** The shared/split question was ill-posed as a binary: T/I
identity is a FILTRATION property. Any compiled program should carry T−I
only through the attention1 interface and may merge T+I at the MLP1-total
boundary — a concrete, testable architecture constraint. **Falsifier:** a
removal rung that deletes the T−I component at attention1 only; the
site-graded picture predicts large attention1-route damage with small
MLP1-total-route damage; the merged picture predicts proportional damage.
**Status:** measurement done; the removal rung is main-line material —
proposed to Codex, not enqueued (their live-state-reader route has the
lane and this may compose with it).

### 3. The measured MDL boundary for categorical context — framing
485/486 give the two-part-code argument empirical teeth: 698-token and
287-pair tables both cost parameters and BUY NEGATIVE held-out bits at the
finite-response grain. The program can now state, with receipts, that
categorical context coding is below the MDL waterline here, which is why
the routing to continuous state readers is forced (Rissanen; Grünwald,
*The MDL Principle*). No new experiment; a sentence for the dossier.

## Pruned this cycle
Lyapunov thread (out-of-sample tests degenerate under H-B; parked until
the fingerprint tripwire catches a second shift); further gauge/commutant
passes (479 closed); Hankel/automata; info bottleneck; standing prunes.

## Citations
Ichimura, "Semiparametric least squares estimation of single-index
models" (J. Econometrics 1993); Stanley, order polytopes / linear
extensions (Discrete Comput. Geom. 1986); Bishop, Fienberg & Holland,
*Discrete Multivariate Analysis* (log-linear interaction structure);
Björck & Golub, principal angles (Math. Comp. 1973); Rissanen 1978;
Grünwald, *The MDL Principle* (2007).
