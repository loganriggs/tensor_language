# Exact natural-text copy-source edge: discovery findings

Date: 2026-08-29  
Status: **exploratory replicated discovery; not fresh-data confirmation**

## Plain-language result

At a position where the current token occurred earlier, the model can predict the
next token by looking at what followed the earlier occurrence. For example, if the
current token at position $p$ matches position $j$, then the useful payload is the
token/state at $j+1$.

For the known L8 fetcher heads H3 and H4, almost the entire copy-relevant effect at
these positions comes from exactly that one source edge:

> destination $p$ $\leftarrow$ earlier successor $j+1$.

Deleting only that edge caused `0.12792` nat of loss on natural-text copy positions.
Deleting the entire H3/H4 write at the same destinations caused `0.13403` nat. Thus
the single successor edge accounts for **95.4% of the whole-head CE damage** under
this matched intervention. Deleting the adjacent but wrong source edge caused
`-0.00057` nat—effectively zero.

The payload on the correct edge is overwhelmingly the model's shared block-0 value
bus, the $\lambda_8v_1$ route:

- delete complete successor edge: `+0.12792` nat;
- delete only $\lambda_8v_1$ on that edge: `+0.11692` nat;
- delete only the fresh/context-refined value on that edge: `+0.00544` nat.

So the best current description is:

> L8 H3/H4 implement a conditional key--value lookup. On a repeat, their
> copy-specific output is almost entirely one edge to the earlier occurrence's
> successor, carrying a mostly static/shared token-identity payload through $v_1$.

This is substantially more specific than “these heads are related to induction.” It
identifies the physical source edge and which additive value route carries it.

## Exact computation

For head $h$, destination $p$, and source $k$, attention contributes

$$
w_{h,k}(p)=a_h(p,k)P_hv_h(k),
$$

where $a_h(p,k)$ is the model's actual attention-pattern scalar, $P_h$ is the H3 or
H4 slice of L8 `c_proj`, and

$$
v_h(k)=(1-\lambda_8)v_{\mathrm{fresh},h}(k)+\lambda_8v_{1,h}(k).
$$

The full head write is the exact source sum

$$
w_h(p)=\sum_k w_{h,k}(p).
$$

The implemented intervention subtracts $w_{h,j+1}(p)$ from the otherwise native
attention write. It gathers only the requested pattern scalars and value vectors;
it does not allocate the enormous tensor of every source contribution to every
destination and output coordinate.

The reusable primitive is `HeadWriteTransaction.source_write()` in
`terminal_copy_attention_adapter.py`. It separately exposes `mixed`, `fresh`, and
`broadcast` routes and was checked by exact additive unit tests.

## Input-only source policy and scored cells

For every destination $p$, the intervention finds the nearest $j$ within the previous
128 positions such that the **input** token at $j$ equals the input token at $p$.
The intervention never reads the future target. It chooses source $j+1$ at every such
repeat position.

Targets are used only afterward to define evaluation cells:

- `copy_positive`: target at $p$ equals the already observed token at $j+1$;
- `repeat_negative`: a nearest repeat exists, but its successor is not the target;
- `nonrepeat`: no eligible nearest repeat exists.

The 128-row copy positives are not dominated by adjacent repeats: median match
distance is 34 tokens, mean 42.46, and none has distance one.

## Frozen arms

| Arm | Removed from L8 H3/H4 |
|---|---|
| `edge_mixed` | complete contribution from source $j+1$ |
| `edge_broadcast` | only $\lambda_8v_1$ from source $j+1$ |
| `edge_fresh` | only $(1-\lambda_8)v_{\mathrm{fresh}}$ from source $j+1$ |
| `edge_wrong` | complete contribution from adjacent source $j$ |
| `heads_full` | complete H3/H4 writes at the same repeat destinations |

All other components remain native and every arm uses the same rows and cells.

## Numerical replication

The preregistered discovery began at 32 documents. It escalated to 128 only after the
edge exceeded `0.05` nat and 25% of the whole-head ceiling.

| Quantity | 32 documents | 128 documents |
|---|---:|---:|
| Copy-positive positions | 392 | 1,864 |
| Complete successor-edge $\Delta$CE | 0.11302 | **0.12792** |
| Whole H3/H4 $\Delta$CE | 0.12936 | **0.13403** |
| Edge / whole-head damage | 87.4% | **95.4%** |
| Broadcast-edge $\Delta$CE | 0.10392 | **0.11692** |
| Fresh-edge $\Delta$CE | 0.00497 | **0.00544** |
| Wrong-edge $\Delta$CE | 0.00017 | **-0.00057** |
| Repeat-negative $\Delta$CE, complete edge | -0.00634 | **-0.00837** |
| Nonrepeat $\Delta$CE, complete edge | -0.00037 | **-0.00024** |
| Complete-edge KL on copy positions | 0.05088 | **0.05632** |
| Runtime | 11.5 s | 16.8 s |

All four frozen gates passed at both sample sizes. At 128 documents, the unweighted
document-mean complete-edge effect is `0.11150` nat with SE `0.01139`; the broadcast
effect is `0.10194` with SE `0.01069`. The adjacent-source control's document mean is
`-0.00002` with SE `0.00060`.

Deleting the correct copy edge actually *improves* CE by `0.00837` nat on repeat
positions whose earlier successor is not the target. This is consistent with a real
conditional copying service: the same attempted copy is useful when the continuation
repeats and mildly harmful when it does not.

The exact adapter's maximum all-head recomposition relative error was `0.002585`,
matching the previously checkpoint-certified bfloat16 accumulation discrepancy for
L8 (`0.002627`). Its full native write remains the physical native contraction; the
small discrepancy concerns summing separately projected head writes.

## What this accomplishes

This provides a much finer causal and editable interface than a whole-head ablation:

1. **Physical localization:** one named source edge, not merely two heads.
2. **Payload localization:** the shared $v_1$ broadcast, not a generic 128-dimensional
   head subspace or most of the fresh value.
3. **Input-side policy:** nearest token equality chooses the edge without consulting
   the target.
4. **Selective effect:** large damage where that copy is correct, slight improvement
   where it is wrong, and negligible propagation into nonrepeat positions.
5. **Composability:** the edge write is an exact additive tensor term that can be
   removed, replaced, or transplanted while leaving every other source through the
   same heads intact.

## Boundaries and next experiment

This is not yet a standalone compact program. The experiment uses the native L8
pattern scalar $a_h(p,j+1)$ and native earlier residual trajectory. It simplifies the
payload and physical routing, but it has not yet replaced the query/key computation
that assigns the scalar.

The next high-return experiment is therefore small and specific: measure how much of
$a_h(p,j+1)$ is captured by (a) a per-head constant, (b) the historical weights-only
match score, and (c) one affine calibration of that score, fit on 32 documents and
evaluated on disjoint exposed rows. Replace the native edge—not inject an extra
stream vector—and score CE, KL, repeat-negative benefit, and nonrepeat collateral.
If a constant or one-scalar calibration preserves this result, the L8 copy payload
becomes an executable token-equality lookup plus two signed/scaled $v_1$ writes.

Fresh natural text and OOD code remain unopened. No strict whole-model ledger moves
from this exploratory result.

Artifacts:

- `COPY_SOURCE_EDGE_DISCOVERY_PREREGISTRATION.md`
- `copy_source_edge_discovery_32_results.json`
- `copy_source_edge_discovery_128_results.json`
- `discover_copy_source_edges.py`
- `test_discover_copy_source_edges.py`
