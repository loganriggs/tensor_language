# Three-hour mathematical review — 2026-09-03 18:30 UTC

The model is a tensor network, so component names are coordinates, not necessarily computational atoms. The R566 split
suggests a useful factorization of numeric behavior:

$$
\text{logits}=D\bigl(S_F(x),\,R_F(x)\bigr),
$$

where $F$ is prompt format, $S_F$ extracts a current numeric state, $R_F$ extracts a relation or structural rule, and
$D$ maps state and rule to the next-token distribution. Numbered lists may use a structural rule “next list index,”
while comma sequences use a data-derived relation. Their outputs can still share part of $D$.

The right equivalence relation is downstream causal use: two internal vectors represent the same factor when every
registered downstream consumer responds equivalently under matched interchanges, including interaction terms. This is
gauge-resistant because it is defined by contractions and interventions, not coordinate overlap.

For a bilinear consumer $B(x,y)$ and source replacements $x',y'$, isolate its interaction by

$$
\Delta_{xy}=B(x',y')-B(x',y)-B(x,y')+B(x,y).
$$

R567 should later support this analysis separately for the state input, rule input, and output map. A shared successor
map would predict equivalent downstream effects for digit and word state representations after the appropriate input
translation; a format-specific shortcut would fail that cross-format interchange. This is a stronger target than a
low-rank approximation and directly serves extraction, OOD prediction, and selective intervention.
