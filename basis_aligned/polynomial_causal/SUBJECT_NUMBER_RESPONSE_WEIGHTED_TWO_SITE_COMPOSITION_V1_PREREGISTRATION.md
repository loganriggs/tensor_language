# Subject-number response-weighted two-site composition V1

Registered before running the state-dependent generator on the already-open
two-clause authority. The earlier two-site result established composition of a
fixed rank-one write; this assay asks whether two independently executed instances
of the complete `H -> (z,s) -> beta -> alpha*u` graph compose.

For each two-clause prompt, capture the native MLP8 raw state and L11H3 response
interface independently at subject positions 5 and 14. Generalize the existing
head reconstruction only by replacing its hard-coded subject position with the
registered site position. At each site $j$, execute the frozen singular-to-plural
prototype:

$$
z_j=u^TH_j(x_j),\quad s_j=u^T[H_j(x_j+p)-H_j(x_j)],\quad
\alpha_j=[1,z_j,s_j,z_js_j]\beta,\quad w_j=\alpha_j u.
$$

No donor state, coefficient fit, or behavior outcome enters either generator.
Evaluate native, zero-write, site-1-only, site-2-only, and both-write forwards.

Instrumentation requires the generalized $H_j(x_j)$ to reproduce the captured
native head at both sites within `5e-5`, zero writes to reproduce native logits
within `1e-5`, finite coefficients, native number accuracy at least `.75` in every
site/template cell, and exactly seven forwards over 112 sequences.

Each single-site generated write must have target effect RMS at least `.01`, move
at least `.75` of rows toward the opposite-number answer, and keep the registered
can/will control below `.75` of target RMS. The later-site edit must affect the
earlier answer by at most `1e-5`.

For effects $E_1,E_2,E_{12}$, composition requires $E_1+E_2$ to predict $E_{12}$
overall and at each site with cosine at least `.95`, relative L2 at most `.25`,
sign agreement at least `.90`, and norm ratio in `[.80,1.20]`. Passing establishes
reuse/composition of the complete extracted generator on two sites; it does not
remove the categorical requested-direction port or compress the two dense prototype
vectors.
