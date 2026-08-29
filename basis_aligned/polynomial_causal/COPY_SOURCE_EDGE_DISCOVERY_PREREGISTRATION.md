# Exact copy-source edge discovery

Status: **exploratory, frozen before any model outcome from this runner**.

This is deliberately a discovery-lane experiment. It reuses the already exposed
`selection_natural` rows, begins with 32 documents, and writes no authority, lock,
fresh role, or confirmatory receipt. A positive result must be repeated on the first
128 rows and then on genuinely fresh data before it can support a final claim.

## Question

Earlier work reduced much of the four-head copy front end and showed that the
cross-layer $\lambda v_1$ route is important, but it intervened on whole heads,
attention windows, fitted axes, or routes across every source. Which **exact source
edge** carries the L8 fetchers' copy payload on natural text, and does that edge use
the shared $\lambda_8 v_1$ code or the context-refined fresh value?

For destination position $p$, let $j(p)$ be the nearest earlier position within 128
tokens with the same current token. This is computable from the input alone. The
predicted successor source is $k(p)=j(p)+1$. At L8 heads H3 and H4, the exact additive
write from source $k$ is

$$
w_{h,k}(p)=a_h(p,k)P_hv_h(k),
$$

where $P_h$ is the matching slice of `c_proj`. The physical intervention subtracts
this term from the otherwise native L8 attention write. No full
`[batch,head,query,key,d_model]` tensor is materialized.

## Frozen arms

All interventions use the same input-only nearest-match policy and act at every
eligible destination, including positions before the scoring window so downstream
propagation remains part of the effect.

1. `native`: no intervention.
2. `edge_mixed`: remove the complete H3/H4 contribution from source $j+1$.
3. `edge_fresh`: remove only its $(1-\lambda_8)v_{\mathrm{fresh}}$ contribution.
4. `edge_broadcast`: remove only its $\lambda_8v_1$ contribution.
5. `edge_wrong`: remove the complete contribution from $j$, the matched query rather
   than its successor. This is the adjacent-source directional control.
6. `heads_full`: remove the complete H3/H4 writes at the same eligible destinations.
   This is the same-position causal ceiling, not a proposed selective program.

Scored positions are 64--255. A `copy_positive` has an eligible $j$ and target token
equal to the observed successor token at $j+1$. A `repeat_negative` has an eligible
$j$ but a different target. `nonrepeat` contains the remaining scored positions.

## Frozen gates and escalation

On the first 32 rows:

- P1, consequential edge: `edge_mixed` copy-positive $\Delta$CE is at least `0.05`
  nat.
- P2, meaningful physical share: its copy-positive $\Delta$CE is at least 25% of
  `heads_full` damage when the latter is positive.
- P3, directional source: `edge_wrong` damage is at most half of `edge_mixed` damage.
- P4, input-policy specificity: `edge_mixed` copy-positive damage exceeds
  `repeat_negative` damage by at least `0.03` nat, and `nonrepeat` damage is at most
  25% of copy-positive damage.

Automatically escalate to the first 128 rows only if P1 and P2 pass. P3/P4 are
diagnostic rather than escalation requirements. The fresh-versus-broadcast comparison
is deliberately two-sided: fresh dominance means context-refined payload; broadcast
dominance means the static token-identity bus is the main exact edge payload.

Failure of P1 or P2 rejects this source edge as the useful physical grain. It does not
reject the older copy mechanism or other source-selection policies.
