# Mathematical review: positional transport and the consumer being preserved

Due04:49 UTC, performed after the native query-phase result. The full goal
remains a smaller explained tensor program with OOD prediction, independent
extraction, selective manipulation and composition/reuse. Original bilinear
handoff/pilot governs; the stored better_math wording is stale.

## Object, domain and executable consequence

Bilin18 has18 layers,9 heads of width128, residual dimension1152, MLP width4608,
vocabulary50304 and545902902 native parameters. The current selected program
uses24 heads in13 layers and one fitted direction per selected layer block:
3072 fitted axis coefficients, total projector rank13. That is an interface
proposal with the entire native prefix/suffix retained, not an independent
algorithm or a parameter saving.

At one attention call, normalized content factors q1,k1,q2,k2 and the mixed
value v are fixed native quantities. Position acts on64 coordinate pairs per
factor. For each pair, R(t)=[[c_t,s_t],[-s_t,c_t]]. The actual tables are rounded
BF16 values, generated from FP32 frequencies. The causal mask is unchanged.
Changing a query phase is the allowed local intervention; changing the text,
source mask, upstream content or first-value payload is not part of it.

Expanding one score yields sum_m A_m C_m(t,s)+B_m S_m(t,s), where

    A_m=q_a*k_a+q_b*k_b; B_m=q_a*k_b-q_b*k_a
    C_m=c_t*c_s+s_t*s_s; S_m=c_t*s_s-s_t*c_s.

The two scores multiply, divide by128², multiply v, sum over causal sources,
and feed the output map and native suffix. With factors as independent inputs,
this read is degree5. With shared linear query/source inputs and normalization
omitted its local numerator has bidegree(2,3). Actual normalized producers and
the full model are not polynomial maps of raw text/residual coordinates.

For the actual tables, det R(t)=c_t²+s_t²>0. Therefore the exact real-arithmetic
phase transport is R(t') R(t)^-1, with R(t)^-1=R(t)^T/det R(t). Dense rotations
cannot be moved through native RoPE or coordinatewise products without
transforming those operators. No new dense-gauge freedom is assumed.

## Known results: exact mappings and limits

[RoFormer](https://arxiv.org/abs/2104.09864) supplies ideal rotary relative-position
structure. Our C/S expansion maps directly to its rotated query–key bilinear
form. The relative-phase group identity needs true trigonometric rotation;
the rounded native table violates that premise. Keeping both absolute tables
restores the exact local formula. This is an execution identity, with no
learned-feature identifiability guarantee. Evaluation at one query costs
O(H*T*d) arithmetic and O(H*T) score workspace, plus retained projections,
prefix and suffix. Pair coefficient banks need128 entries, not an expanded
64² cross-frequency feature table. We did not allocate a full symbolic tensor.

[Bilinear weight analysis](https://arxiv.org/html/2410.08417v2) maps an MLP and a
linear consumer to a symmetric quadratic form. The A/B coefficients above
likewise admit bilinear weight contractions into chosen query/source input
spaces. A decomposition of these forms remains a candidate grammar; no
Kruskal uniqueness assumption or stable semantic feature basis has been
established. Shared positional tables alone give architectural sharing.

[CLUE](https://arxiv.org/abs/2004.11961) preserves specified linear observables
under polynomial ODE dynamics by constrained lumping. The relevant mapping
here is the choice of a consumer matrix W, not a direct application of its
algorithm: this model has discrete normalized transitions, and output
projection alone is not closed dynamics. Its minimality theorem does not
license a smaller transformer. Similarly, Hankel minimal realizations apply
to the specified linear weighted-automaton class, not our normalized state
updates. Tensor contraction-width results govern execution cost, not semantic
identification. None makes a reader-null direction automatically removable.

[Causal abstraction](https://jmlr.org/papers/v26/23-0058.html) makes the claimed
observables and intervention correspondence explicit. In our setting, a
task-only answer reader and the complete selected heads' vocabulary effects
define different objects to preserve. Neither can silently stand in for the
other. This observation yields the next executable norm partition below.

## Trained falsifier and decision

ABOUT_FOR_QUERY_PHASE_V1 reused the unchanged v473 native fit on five short
frames, then froze it before alternate phase evaluation. All128 held pairs
were previously opened. CPU token audit shows one changed token and equal
lengths within every A1 pair; index gaps3/4 in fit,4 in fu,9 in fw. The fixed
map caps the final query phase at cue_index+4, changing only fw by minus5.
Each alternate donor endpoint is native_base+virtual_donor-virtual_base;
the same BLOCK-LIVE axes and full native suffix implement the interchange.
Both native prefix runs remain necessary. No new text-only executor exists.

Instrument A and control/identity D pass. All capture and short identity
bridges are zero; maximum original recovery replay error.000460. CPU transport
FP64 error<=1.34e-15; deployed FP32 primitive error9.54e-7. Native rotation
squared scale ranges.994644–1.005508, so the transpose-inverse control fails
as intended (.01738 max error). No numerical issue rescues the scientific null.

B fails: fw axis fraction falls from.688135 to.378006; complete-set recovery
also falls from.937137 to.526582 with the alternate phase endpoints. C fails:
all target full-vector errors span.764–.946. Execution was58.719s,
1405 forwards,107792 sequence evaluations,120 backward steps. All weights
remain, zero structural savings. Fitting/selection dominates this run; future
tests that reuse a fit should save its frozen axes rather than rerun it.

Close this direct query-phase portability proposal. No key-phase, shift,
head-subset or rank rescue. The result does not prove absence of positional
computation: it rejects this particular content-held adapter. Contextual
content and its producers remain unexplained.

## A task reader is not the whole head: exact diagnostic

For a centered vocabulary-effect error e and answer contrast
w=e_answer-e_foil, w^T w=2. Orthogonal decomposition gives

    ||e_parallel||² = (w^T e)²/2
    ||e_perpendicular||² = ||e||² - (w^T e)²/2.

Sum over rows even when answer/foil orientation changes per row. The saved
full-vector and margin norms suffice; no additional logits or native run are
needed. Executed output_reader_error_partition_v1.py finds that at least
99.9912% of the squared approximation error on the seven about/for panels is
outside this one answer-contrast direction. On the unchanged short panels
the minimum is99.9990%. This is Euclidean geometry, not a semantic label for
the other output changes or a KL guarantee.

Thus C's failure rejects reconstruction of the whole selected-head effect.
It does not independently reject a narrower task-specific circuit. Conversely,
good answer margins cannot establish the whole-output program required by the
user. The next scientific object should specify and account for the other
consumer functions, rather than improving one answer axis and calling the
module explained. Avoid both full-head-equals-circuit and single-margin-equals-
model errors.

For several consumers W, the exact observed-error norm is

    ||P_W e||² = (W e)^T (W W^T)^+ (W e),

where + is the Moore–Penrose inverse and P_W projects onto row(W). Overlapping
answer tokens make WW^T non-diagonal; summing squared margin errors can double
count directions. This is invariant to redundant/mixed reader coordinates
when their row space is unchanged. Forming a sparse-reader Gram costs
O(m²k) for m readers with at most k entries each; dense eigensolve O(m³), and
per-row scoring O(m²). No vocabulary-size square tensor is needed. Rank and
conditioning must be reported; numerical near-dependencies are not exact
population identities.

Concrete next CPU action: verify this joint-reader metric with overlapping
answer pairs, duplicated readers and a three-pair cycle. This prevents shared
answer-token geometry from masquerading as shared computation when two
circuits consume the same module. It is metric machinery, not itself a circuit
discovery or a reason to stop the broader goal. Next mathematical review07:49;
hourly checkpoint remains05:14.
