# THREE-HOURLY MATHEMATICAL REVIEW — 2026-09-01 19:00 UTC
Context: MLP0 finished (rank curve = frontier; exact residual quadratic); the user's downstream-quotient program produced three individuation results (416–418: no multi-head sharing, no consumer-side collapse, zero atomic cross-head QK edges — yet diffuse ~45% regression overlap). Sign convention §2135 throughout.

## Top three moves (ranked), #1 executed

### 1. EXECUTED — The COMMON-CARRIER hypothesis, and its first quantitative test (on 418's stored pairwise geometry)
**Object:** the 18 attention0 head-branch QK token-function subspaces (128-dim each, folded exactly over the vocabulary).
**Hypothesis:** the "diffuse, never atomic" sharing pattern is a GLOBAL COMMON SUBSPACE C read by every head-branch, plus individuated private remainders — not graded/cliquish pairwise sharing. Distinguishing statistic: a global-C model predicts pairwise projector overlaps that are UNIFORM across all pairs (≈dim-share of C); cliquish sharing predicts heavy-tailed overlaps.
**Measured (144 cross-head pairs, from the 418 receipt, CPU):**
- Pairwise centered projector overlap: q mean **.1911, sd .0100 (cv .052)**; k mean .1857, sd .0144 (cv .078); nulls at .0032. Range .167–.215 — REMARKABLY uniform.
- Each pair shares ~11–12 dimensions at cos² ≥ .50 (leading principal cos² ~.82, .67, .63, …).
- Per-branch mean overlaps span only .175–.203 — no hub heads, no cliques; head 3's branch is the LOWEST (most private), independently consistent with 417's head-3 irreducibility.
**Verdict of the test: the uniformity strongly favors the common-carrier model** — roughly a ~24-effective-dim (11-strong-dim) subspace shared by essentially ALL branches uniformly, with private remainders carrying head identity. This single structure explains four results at once: 416 (TOTAL beats head-sum — the carrier lives in the total write), 418 (no atomic edges — pairwise sharing is exactly the carrier, below edge thresholds), the diffuse ~.45 regression R² (carrier + correlated privates), and MLP0's token-identity component (the carrier is plausibly the token-identity/frequency subspace every head must read).
**Cheapest falsifier (next rung, CPU, registrable):** extract Ĉ = top eigenvectors of the AVERAGE branch projector; test (a) pairwise overlaps are reproduced by Ĉ alone (predict .19 uniformly; residual overlaps after projecting Ĉ out should drop to ~null .003); (b) Ĉ aligns with the MLP0 L-component/token-identity subspace (cosine test vs the 395/397 objects) — which would JOIN the attention and MLP mechanism arcs through one shared object.

### 2. Certified damage bounds (approximation certificates)
The softcapped logit map (30·tanh(z/30)) is 1-Lipschitz; logits are linear in the residual stream; so per-position |ΔCE| admits an explicit bound from ‖Δwrite‖ through the frozen suffix operator norm. Would convert measured conversion gains into CERTIFIED intervals for small perturbations (precision casts, the degree-two rounding law). Likely loose for structural cuts; right-sized for precision claims. Proposal only.

### 3. Prequential pricing of the mechanism dossier (unchanged, mine, quiet hour).

## Pruned
Head-level quotients (417 closed), cross-head atomic vocabularies (418 closed), all metric/routing/producer families (certificates stand), DMRG/balanced-realization analogies (Codex's own 417 preamble rejected them correctly).
