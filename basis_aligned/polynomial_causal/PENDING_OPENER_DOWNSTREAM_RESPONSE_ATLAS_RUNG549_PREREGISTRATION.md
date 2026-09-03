# Rung 549 preregistration: downstream responses to the pending-opener site

**Frozen:** 2026-09-03 16:02 UTC, after the independently audited R546/R548 site confirmation and before any R549
model call

## Question

When the complete layer-13 attention-head-8 output is replaced with its value from a counterfactual prompt, does a
later attention head or MLP produce a stable response that identifies the requested change across two independently
constructed prompt families?

This is a screen for an additional causal consequence to use in the next learned interchange. It does not fit a DAS
subspace, choose a rank, open FINAL_TEST/OOD, or claim that a later native module is itself the circuit.

## Frozen rows and interventions

The run uses all 540 R545 FIT+SELECT pairs: two answer-changing families and three answer-preserving families. R546
already established native capability and a live complete-state intervention at L13H8 on exactly these rows. For
each pair and both directions, the run replaces the complete 128-dimensional L13H8 state at the actual final token.

The later candidate outputs are fixed before the run:

- the 1,152-dimensional MLP write from layers 13 through 17;
- each of the nine 128-dimensional attention-head outputs from layers 14 through 17, immediately before that
  layer's attention output projection.

There are 41 candidates. One native and two patched passes per batch of eight gives exactly 204 forwards and zero
backwards. All row-level response vectors are saved in a tensor bundle with row IDs.

The exact tie-breaking order is `mlp13_write`, followed at each layer 14, 15, 16, and 17 by
`attn{layer}h0_output` through `attn{layer}h8_output`, then `mlp{layer}_write`.

## Exact response

For candidate output $y_j$, base prompt $b$, donor prompt $d$, and the base run patched with the donor L13H8 state,
define

$$
p_j^{b\rightarrow d}=y_j(b\leftarrow h_d)-y_j(b),
\qquad
n_j^{b\rightarrow d}=y_j(d)-y_j(b).
$$

$p$ is the candidate's causal response to the L13H8 intervention. $n$ is the full natural difference between the
two prompts. The reverse direction is computed symmetrically. For answer-changing rows, every response is labeled
by one of the six ordered closer transitions, such as parenthesis-to-quote.

## FIT-only selection

For each candidate and ordered transition, average its FIT causal responses to make a transition template. Two
cross-family tests are then run on FIT:

1. templates from direct type substitutions classify the ordered transition of completed-then-reopened examples;
2. templates from completed-then-reopened examples classify direct substitutions.

Classification chooses the template with largest cosine to the response vector; chance accuracy is $1/6$. A
candidate is FIT-eligible only if both cross-family accuracies are at least 50%, the median cosine between a
transition template and the negative reverse-transition template is at least 0.30, and the median maximum absolute
cosine of answer-preserving responses with any transition template is at most 0.40. Among eligible candidates, the
fixed score

$$
\min(a_{\mathrm{direct\to order}},a_{\mathrm{order\to direct}})
-\operatorname{median}_{\mathrm{controls}}\max_t|\cos(p,T_t)|
$$

selects one candidate; ties use the candidate order stated above. If none is eligible, no candidate is selected.

## SELECT validation

The selected candidate is accepted as an independently validated downstream consequence only if:

- both leave-one-family-out transition accuracies are at least 50% on SELECT;
- the median answer-preserving maximum absolute template cosine is at most 0.35;
- the median ratio $\lVert p\rVert/\lVert n\rVert$ on answer-changing SELECT rows is at least 0.05, ruling out a
  numerically tiny but classifiable response.

All 41 candidates' FIT and SELECT metrics are reported, but SELECT cannot change the selected candidate. The strong
null is that no candidate is FIT-eligible or that the FIT-selected candidate fails any SELECT bar. Such a null means
the next learned interchange may use the endpoint plus invariance penalties, but cannot cite an independently
validated later-module response.

## Readout-alignment diagnostic

For each MLP-write template, the run reports cosine with the three pairwise closer-token unembedding contrasts. For
each attention-head template, those contrasts are pulled back through that head's slice of the output projection
before taking cosine. High alignment does not invalidate the response, but marks it as likely another copy of the
closer readout rather than a distinct downstream computation. This diagnostic is never used for selection.
