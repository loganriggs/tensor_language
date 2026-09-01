# Plain-English update — 2026-09-01 05:45 UTC

**The one-sentence headline:** using the directions MLP0 actually receives on real text, rather than the directions
with the largest raw weights, produced a new compressed model that is smaller, more accurate, and more causally
faithful than our previous best MLP0 artifact.

**What changed.** MLP0 has two large input maps, Left and Right. We replace them with one shared encoder feeding
two smaller maps. The earlier version chose the shared directions by ordinary weight SVD and needed rank 768.
The new version measures which input directions the model actually uses on contextual text, then solves the matched
reduced-rank regression problem. At rank 640 it keeps the relevant action while discarding directions that are large
in weight space but rarely used.

**The new adopted frontier.** Lower added cross-entropy is better; certificates are 62 circuit-level behavioral
checks.

- 539,595,062 scalars: `+.004692`, 54/62 certificates — highest-fidelity compressed point.
- **535,613,750 scalars: `+.008265`, 52/62** — new dominant MLP0 artifact.
- **534,286,646 scalars: `+.010728`, 48/62** — new smallest adopted Pareto point.

The rank-640 point strictly improves the previous rank-768 artifact (536,940,854 scalars, `+.009012`, 50/62): it
is smaller, has lower error, and preserves more certificates.

**Why we trust it.** Both new points passed physical whole-program composition, exact literal billing, fresh
FineWeb windows, 120 shifted WikiText rows, and a direct signed a16 causal intervention. Rank640's WikiText mean
damage is `+.008580`; its signed-effect cosine is `.994483` and collateral-circuit Spearman is `.997653`. Rank512
is similarly strong at `.993405/.996939`. These compare signed intervention effects within each model, not an
unsigned or aggregate proxy.

**The mathematical lesson.** “Rank” is meaningless without a metric. Frobenius weight energy badly misranked
important directions, especially at late layers. Contextual input covariance exposed stable low-rank action even
where weight SVD appeared to require full rank. The gain transfers to MLP0 and survives every adoption gate. At the
same time, combining several individually good MLP replacements still incurs an approximately 1.3x interaction
tax; sequential refitting did not remove it. The next work therefore emphasizes deeper single-site MLP0 rank
frontiers and metric-aware replacements at genuinely different modules, with composition bars derived from the
measured tax rather than optimistic additivity.
