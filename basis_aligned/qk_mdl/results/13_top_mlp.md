# The top MLPs: diffuse input, low-rank contextual output

Files 11–12 left one component unnamed: the top MLPs (L13–17), whose windowed reads
cost +0.59 composed. This file characterizes them.

## Input side: broad aggregation, no nameable channel (TM-1)

bilin18's MLP is PURE bilinear (`Down(Lx ⊙ Rx)`, ungated), so MLP outputs decompose
EXACTLY over stream pairs — the same gated machinery as the QK map (file 11). Result:
bottom MLPs read a tight recent window (L2: 99% recent×recent, L5: 94% — why windowing
them was free), but **L13 is diffuse: 19% recent, top pair 3%** — broad aggregation
over dozens of old streams. L16 (44%) and L17 (65%) sit between, with the attn5 hub
stream reappearing in their top pairs. Unlike selection (one dominant stream + one
hub), the top-MLP contextual input has no channel to name.

## Output side: a few contextual directions (TM-2)

The H7 playbook (file 12) applied to MLP outputs — token-conditional mean + rank-k
deviation projection with LIVE coefficients:

| | mean-only | rank-1 | rank-4 | rank-16 | dev PC shares |
|---|---|---|---|---|---|
| mlp16 | +0.141 | +0.099 | **+0.040** | +0.024 | 40%, 17%, 8%, 4%… |
| mlp13 | +0.041 | +0.034 | +0.033 | +0.031 | 4%, 2%, 1%… (diffuse) |

**mlp16's contextual function factors through ~4–16 scalars**: its output is a token
mean plus a handful of contextual feature directions with live gains — the same shape
as H7 one level up (rank ~4–16 instead of 1). mlp13 is individually cheap and its small
deviation is genuinely diffuse; the composed +0.59 top-MLP windowing damage is
interaction compounding, not any single layer's irreducible high-rank content.

## The full picture of bilin18's contextual computation

| component | contextual object | rank of context |
|---|---|---|
| selection, bottom (L1–4, 6+) | token-static tables + local window | 0 |
| selection, L5.H5 | content match (induction), identity payload | high (identity) |
| transport, L5.H7 | one structure feature × live gain | **1** |
| MLPs, bottom 2/3 | local window only | 0 |
| mlp16 (dominant top MLP) | ~4–16 feature directions × live gains | **~4–16** |
| mlp13–15, 17 | small, diffuse | — |

Everything contextual in this 546M model is either literal token identity (H5's
payload), or a SMALL number of live scalar gains on fixed feature directions. Caveat:
rank-k-with-live-coefficients is a structural statement (the function factors through
k scalars), not a compute reduction — the scalars are computed by the live model.
Naming the mlp16 directions (logit-lens / examples) is the queued follow-up.

## The directions have names: document register (TM-3)

Logit-lens + extreme-firing contexts for mlp16's top deviation PCs (`../mlp16_dirs.py`):

| dir (var share) | identity | example firing context |
|---|---|---|
| 0 (40%) | legal-citation register | `…609, 614 (1965); see also` |
| 1 (18%) | general prose continuation | `…hit reality TV show was ensnared` |
| 2 (8%) | legal captions/names | `…Plaintiff-Appellee\n\nDouglas K.` |
| 3 (5%) | XML/markup code | `…EndOf="parent"\n    app:` |
| 4–7 (<2% each) | blog boundaries, technical prose, patent numerics | — |

At first read these look like DOCUMENT REGISTER (domain/genre) features. The causal
test says otherwise (TM-4, `../mlp16_register_swap.py`): patching ONLY the top-4
coefficients inside a rank-64 live reconstruction —

| top-4 coefficients | ΔCE |
|---|---|
| live (reference) | +0.023 |
| document-mean (slowness test) | **+0.103** |
| swapped across documents | +0.158 |
| zeroed | +0.113 |

Document-constant coefficients destroy most of the value — so these are NOT slow
register state. The honest naming: the directions fire in register-specific contexts
(hence the lens/examples), but their causal content is FAST-VARYING local structure
within those registers — where you are inside the citation/markup/numeric pattern,
not which document you are in. The top MLPs resist windowing because this structural
state is computed from diffuse long-range input (TM-1) yet changes token-to-token.
(Sample caveat: pile-10k's early slice is legal-heavy — the structure, a few live
gains on fixed directions, is the finding, not the register labels.)

Files: `../mlp_stream_interactions.py/.json`, `../mlp16_rank.py/.json`,
`../mlp16_dirs.py/.json`.
