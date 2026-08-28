# Contextual compiler no-go theorem and admission gate

Date: 2026-08-28

Status: mathematical certificate and mandatory protocol for future compiler candidates.

## No-go theorem for position-wise replacement grammars

Let the initial state at sequence position $j$ be

$$
h_j^{(0)} = e(t_j),
$$

where $t_j$ is the current token. Suppose every installed attention and MLP replacement
at every layer has the form

$$
h_j^{(\ell+1)} = \Phi_\ell\!\left(h_j^{(\ell)}, t_j\right),
$$

with no argument from any position $i\ne j$ and no sequence-valued state.

Then there is a function $g_\ell$ such that

$$
h_j^{(\ell)} = g_\ell(t_j)
$$

for every layer. The proof is induction: it holds at layer zero; if it holds at layer
$\ell$, then

$$
h_j^{(\ell+1)}
= \Phi_\ell(g_\ell(t_j),t_j)
= g_{\ell+1}(t_j).
$$

Consequently the final logits are a function of the current token alone and, wherever
the derivative exists,

$$
\frac{\partial z_j}{\partial h_i^{(\ell)}}=0
\qquad\text{for every }i\ne j.
$$

No increase in table rank, correction rank, polynomial degree, or position-wise hidden
width can repair this. At least one explicit sequence primitive is necessary.

## Empirical certificate for the old compiler

The matched poke diagnostic gives the same conclusion operationally:

- live model, uncovered-position poke: maximum later covered loss change $0.118383$ nat;
- live model, covered-position control: $0.072373$ nat;
- fully installed old program, both pokes: exactly $0$ later covered loss change.

Thus S1747--S1758 measure a per-token null compiled through 36 position-wise layers.
They do not measure a compressed contextual transformer.

A discovery-only bigram fitted on the same 24,576 rows obtains covered CE $7.88804$
at skip7000 and $7.90729$ at skip11000, versus $6.57512$ and $6.57289$ for the best
36-site program. The old program therefore computes a materially richer current-token
function than a simple bigram. This does not restore any prefix dependence.

## Mandatory admission gate

A candidate may enter the whole-model simplicity frontier only if all of the following
are reported on preregistered discovery and held-out roles:

1. **Owned sequence primitive.** The program exposes the exact cross-position operator
   it uses: attention-shaped contraction, convolution/state-space recurrence, explicit
   lag/prefix tensor, or another total-support sequence map.
2. **Paired-prefix response.** For sequence pairs with the same current token and
   controlled different prefixes, report the native and candidate logit differences at
   the current position. The lower confidence bound on nonzero candidate prefix
   sensitivity must exceed zero.
3. **Internal causal transport.** Apply matched perturbations at earlier positions and
   report later-position loss/logit effects. Define context recovery as

   $$
   R_{\mathrm{ctx}}
   = \frac{\mathbb E\lVert\Delta z_{\mathrm{candidate}}\rVert_2^2}
           {\mathbb E\lVert\Delta z_{\mathrm{native}}\rVert_2^2}.
   $$

   The complete curve is reported; a zero value is an automatic rejection as a
   transformer reconstruction.
4. **All-position predictive score.** Score CE on every eligible target, split by
   current-token seen/unseen status. Covered-only CE is diagnostic, never promotional.
5. **Executable ownership.** Literal native attention/MLP calls and hidden buses are
   zero, total input support is true, and storage/compute include every table, bias,
   state, and correction.
6. **Composed measurement.** Context recovery and CE are remeasured after all candidate
   sites are installed. One-site behavior cannot substitute for whole-program behavior.

Simplicity is then evaluated only among candidates passing this gate, using stored bits,
multiply-adds, tensor/arithmetic-circuit rank, description length, and edit/extraction
consequences. A cheaper per-token map remains a useful null model but is not on the same
contextual-program frontier.

## Admissible next grammars

The lowest-cost existing option is the owned tensor-preserving attention bank: it has a
real prefix contraction and already passes exact identity. Shared-QK-384 is its first
compressed point. MLP compression should be inserted inside that contextual bank rather
than replacing attention outputs with position-wise regressors. Alternative sequence
primitives are admissible only if they beat this attention-shaped reference under the
same context-recovery, CE, ownership, and cost accounting.
