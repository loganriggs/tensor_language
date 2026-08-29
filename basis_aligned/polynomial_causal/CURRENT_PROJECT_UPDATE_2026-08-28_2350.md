# Current project update — 2026-08-28 23:50 UTC

**Last updated with the final context-cross result: 2026-08-28 23:58 UTC.**

## UPDATE: what is new since the last explanation

There are two substantial updates.

1. We now have a real, measured **simplicity-versus-fidelity frontier** for the
   context-free program.  It is not a one-dimensional rank curve: vocabulary
   coverage and rank interact, so both must be priced together.
2. The first prospective experiment outside that context-free class is now
   complete.  It tested whether MLP0/1/2 and the later contextual network
   communicate through a small, predictive interaction state.  **Neither rank 3 nor
   rank 4 passed on both document roles.**  The failure is informative: one class of
   broad downstream replacement was predicted very well, while local attention3 and
   attention3+MLP3 replacements were not.  A single dense low-rank law is therefore
   too crude for this interface.

## 1. The context-free program is understood as a function class

The program replaces each of the model's 36 attention/MLP outputs with a row that
depends only on the token at that position.  At full rank, it exactly reaches the
model's own best output in this position-wise class on covered tokens.

This does **not** mean it reconstructs the live model.  Its all-position CE remains
roughly 2.74 nats worse because the live model uses surrounding tokens.  A fixed
token row cannot represent that context.  This is now a clean mathematical boundary:
more optimization inside the same class cannot recover the missing information.

## 2. The useful definition of simplicity became two-dimensional

We varied two properties of the executable token-row program:

- **coverage:** how many token types get learned rows;
- **rank:** the dimension of the continuous basis shared by those rows.

For every program we measured:

- total stored real numbers, which is a concrete description-size cost; and
- all-position CE on the same evaluation population, which is the model-fidelity
  cost.

A program is **Pareto-dominated** when another measured program is both smaller and
better.  This matters because a dominated program is never the rational choice,
regardless of how one trades memory against CE.

The important example is:

| Program | Stored reals | All-position CE |
|---|---:|---:|
| full rank, 5,419 covered types | 230.1M | 6.01167 |
| rank 256, 16,110 covered types | 164.5M | 5.98851 |

The second program is about 29% smaller **and** better.  So “lower rank is simpler”
or “full rank is the faithful reference” is not enough.  The whole representation
must be priced jointly.

At 16,110 covered types, the newly completed high-rank curve is:

| rank | stored reals | CE above that coverage's per-token ceiling |
|---:|---:|---:|
| 256 | 164.5M | 0.09252 |
| 384 | 244.0M | 0.05107 |
| 512 | 323.5M | 0.02896 |
| 1024 | 641.4M | 0.00157 |
| full | 673.5M | 0.00000 |

Rank 512 captures 69% of the improvement from rank 256 to full rank while paying
only 31% of the extra storage.  Rank 1024 is almost exact but costs 95% of full rank,
so the last bit of fidelity is very expensive.  Across the full measured frontier,
marginal cost ranges by roughly 1,960 times.  This is the first genuinely useful
compression curve: it tells us what each extra unit of executable description buys.

An odd but robust interaction is that more coverage helps at rank 16 and above, but
hurts at rank 4.  A four-dimensional basis is too small to share across 16,110 token
types; the additional types consume capacity faster than their coverage helps.

## 3. Why this is still not an interpretation of MLP0

The low-rank token-row code tells us that much of the context-free behavior can be
represented in a shared continuous basis.  It does not assign stable semantic names
to the basis coordinates.  Because a change of basis can rotate the coordinates
without changing the program, interpreting coordinate 17 as “city” would generally
be arbitrary.

What is meaningful so far is functional:

- tokens share a continuous code rather than requiring wholly separate outputs;
- the code has a measured cost/fidelity curve;
- MLP0/1/2 cannot be simplified independently, because their causal effects compose
  non-additively and MLP2 changes role after MLP0+1 are repaired.

The next semantic decomposition should therefore be joint and downstream-aware.  A
sparse dictionary or SAE is useful only if its features improve prediction of
compositions, OOD transport, extraction, or selective removal—not merely if its
weights look sparse.

## 4. What the running experiment computes, in detail

### 4.1 What is being replaced

We have one fixed, already-built replacement program.  When installed at a model
site, it replaces that site's output with the rank-64 **context-free token-row
approximation** described above.  It does not refit itself for this experiment.
Every site not selected for replacement remains the original live model.

The experiment puts a physical boundary after layer 2 and varies replacements on
both sides of that boundary.

The **early-MLP choice** $P_i$ is one of all eight subsets of MLP0, MLP1, and MLP2:

| $i$ | early sites whose outputs are replaced |
|---:|---|
| 0 | none |
| 1 | MLP0 |
| 2 | MLP0 and MLP1 |
| 3 | MLP1 |
| 4 | MLP0, MLP1, and MLP2 |
| 5 | MLP2 |
| 6 | MLP1 and MLP2 |
| 7 | MLP0 and MLP2 |

The **contextual suffix choice** $S_j$ selects later sites, on the far side of the
boundary:

| $j$ | later sites whose outputs are replaced |
|---:|---|
| 0 | none |
| 1 | attention in layer 3 |
| 2 | MLP3 |
| 3 | attention3 and MLP3 |
| 4 | every attention site in layers 3--8 |
| 5 | every attention and MLP site in layers 3--8 |
| 6 | every attention site in layers 3--17 |
| 7 | every attention and MLP site in layers 3--17 |

“Contextual suffix replacement” is easy to misread.  The replacement itself is
**not contextual**: it depends only on the current token.  The *native sites being
replaced* are contextual, especially attention, and they lie downstream of the
early MLPs.  Thus these interventions progressively remove the model's original
context-sensitive downstream computation.  “Suffix” means the downstream side of
the layer-2 boundary; it does not mean that every choice replaces one contiguous
tail.

### 4.2 What a cell is

A **cell** $(i,j)$ is one complete intervention on the model:

$$
\text{replaced sites in cell }(i,j) = P_i \cup S_j.
$$

For example:

- cell $(0,0)$ replaces nothing, so it is the fully live model;
- cell $(1,0)$ replaces only MLP0;
- cell $(0,5)$ replaces all attention and MLP outputs in layers 3--8;
- cell $(1,5)$ makes both of those changes at once.

We run that modified whole model on the evaluation documents and record its
cross-entropy (CE).  So a cell is **not** a neuron, feature, tensor coordinate, or
one token.  It is one model configuration and its aggregate behavioral result.  The
eight early choices times the eight downstream choices form 64 cells.  We measure
the same 64 interventions on each of two document-disjoint roles, producing 128
physical cell measurements.

Let $C_{ij}$ be the whole-model CE in cell $(i,j)$.  Merely knowing that replacing
MLP0 hurts by one amount and replacing a downstream block hurts by another does not
tell us whether doing both will add, cancel, or amplify.  We therefore subtract the
two separate effects:

$$
\Delta_{ij}
= C_{ij}-C_{i0}-C_{0j}+C_{00}.
$$

Here $\Delta_{ij}$ is the **non-additive interaction**:

- $\Delta_{ij}=0$ means the two interventions compose additively in CE;
- $\Delta_{ij}>0$ means the joint replacement is worse than adding the two separate
  harms predicts;
- $\Delta_{ij}<0$ means the two replacements partly compensate for one another.

This is the formal version of the earlier observation that MLP2 can change role
depending on whether MLP0 and MLP1 are live or repaired.

### 4.3 What “rank 3” or “rank 4” would mean

The 64 values $\Delta_{ij}$ form an $8\times8$ interaction matrix.  If that matrix
has rank $r$, each early replacement choice needs only $r$ numbers to describe how
it interacts with *all* the downstream replacement choices, and vice versa.  For
fixed row indices $I$ and column indices $J$, the tensor-cross formula is

$$
\widehat{\Delta}
= \Delta_{:,J}\,
  \left(\Delta_{I,J}\right)^{-1}\,
  \Delta_{I,:}.
$$

In words: measure a carefully chosen set of full rows and columns, learn how those
interaction profiles combine at their small intersection, and use them to predict
the remaining cells.  This is a matrix “cross” or skeleton factorization.  The
rank-3 version inverts only a $3\times3$ intersection; rank 4 uses a $4\times4$
intersection.

This rank is **not** claiming that the residual stream, MLP0 output, or entire model
has only three or four dimensions.  It is a narrower and testable claim: for this
family of causal replacements, the scalar whole-model CE interactions across this
physical boundary are governed by three or four reusable response profiles.

### 4.4 What this is trying to accomplish

Independent simplifications have already failed to compose reliably.  A replacement
that looks good at MLP0 alone can change what MLP1, MLP2, and later attention must
do.  Measuring every combination separately does not scale: with many candidate
components, the number of combinations grows exponentially.

A successful low-rank cross would give us a small **interface law** between early
MLPs and downstream contextual computation.  It would let us:

1. predict the CE of untouched combinations from a much smaller measurement set;
2. choose several simplified components jointly rather than trusting independent
   local errors;
3. test the same interface on new documents and adjacent layer boundaries;
4. use the interaction coordinates as downstream-defined functional roles before
   trying to attach semantic names or sparse dictionary features to MLP0.

That is why this goes beyond reconstruction of one tensor.  Its success criterion is
an **out-of-sample prediction about a new composed model**, which is one operational
test that the proposed simpler structure is actually useful.

It would still be only a first interface, not a full reverse engineering.  CE is a
single scalar summary, so a pass would not by itself prove token-level agreement,
semantic meaning, selective-removal safety, or OOD equivalence.  Those require the
planned action-level and OOD ledgers.  Conversely, a rank-4 failure would tell us to
stop treating this interface as a dense low-rank law and try structured sparse or
hierarchical interactions instead.

### 4.5 How the prospective test avoids grading its own homework

The test is staged:

- rank 3 fits 48 discovery cells and predicts seven untouched validation cells;
- only if rank 3 fails, rank 4 fits those 55 cells and predicts nine final heldout
  cells;
- the same fixed test must pass separately on two document-disjoint roles;
- CE chooses the model; top-1 is reported but has no post-hoc pass threshold.

The implementation had four audit rounds and passes 47 focused tests.  It binds
exact source, rows, model, and replacement-program hashes; serializes all 128
physical call ledgers; publishes no partial outcomes; and writes its final receipt
last.  Synthetic or injected test data cannot obtain a canonical scientific score.

## 5. UPDATE: final context-cross result

The measurement and the 2,000-draw document bootstrap are complete.  The terminal
receipts are authoritative and hash-bound.  The selected rank is **none**:

| role | rank | prediction set | cross RMSE (nats) | additive RMSE | $R^2$ | result |
|---|---:|---|---:|---:|---:|---|
| skip7000 | 3 | 7 validation cells | 2.208 | 4.117 | -1.385 | fail |
| skip11000 | 3 | 7 validation cells | 1.373 | 4.477 | 0.133 | fail |
| skip7000 | 4 | 9 heldout cells | 4.371 | 2.461 | -1.535 | fail |
| skip11000 | 4 | 9 heldout cells | 4.033 | 2.518 | -1.123 | fail |

**RMSE** is the typical size of the CE prediction error across the untouched cells;
lower is better.  The **additive RMSE** is what we get from assuming
$\Delta_{ij}=0$, meaning that the two interventions do not interact.  **$R^2$** asks
how much variation among the held-out outcomes the cross predicts; zero means “no
better than predicting their mean,” and a negative value means worse than that
simple baseline.

Rank 3 showed a real but unstable partial pattern.  On the skip11000 point estimate,
it reduced RMSE from 4.477 to 1.373 nats.  But its document-bootstrap interval
included negative $R^2$, its fixed $3\times3$ pivot was poorly conditioned, and the
other role had negative point $R^2$.  It therefore did not generalize robustly across
documents.

Rank 4 was a clearer structural failure.  It was worse than the additive baseline on
both roles and had negative $R^2$ on both.  Its pivot condition numbers were about
1,491 and 1,634, meaning small changes in measured entries can be greatly amplified
by the matrix inverse.  Moving from rank 3 to rank 4 did not repair the interface; it
made the frozen cross prediction less stable and less accurate.

### The strange and useful part of the failure

The rank-4 error is strongly organized by the kind of downstream replacement:

| held-out downstream choice | RMSE divided by additive RMSE, skip7000 | skip11000 |
|---|---:|---:|
| attention3 only | 1.43 | 1.59 |
| attention3 + MLP3 | 3.41 | 2.74 |
| every attention and MLP site in layers 3--8 | 0.0160 | 0.0158 |

A ratio below 1 means the cross beats the additive model.  Thus the same rank-4 law
is extraordinarily accurate for the broad layers-3--8 replacement—about 1.6% of
the additive error—while failing badly for the local layer-3 interventions.  This
pattern repeats on both document roles, so it is unlikely to be mere sampling noise.

The best interpretation is not “there is no simple structure.”  It is:

> There is no single dense rank-3/rank-4 interaction law that covers both local and
> broad downstream replacement scales.  The interface likely needs an explicit
> hierarchy or sparse correction keyed to replacement scale and site class.

This is exactly where a hierarchical/DAG or Möbius decomposition becomes more
motivated than simply increasing matrix rank.  We should model a shared broad-scale
interaction plus sparse local corrections for attention3 and the layer-3 block.
Because the heldout outcomes have now been seen, any new model and its next test must
be frozen on a new mask family or new layer boundary before evaluation; we cannot
claim a prospective success by refitting this grid and scoring the same cells.

## 6. What happens next

1. **Fit a descriptive hierarchical/Möbius residual model to this completed grid.**
   Its purpose is to identify the smallest candidate grammar—such as broad suffix,
   local attention, and local MLP correction—not to claim held-out generalization.
2. **Freeze a new prospective composition test** at an adjacent layer boundary or
   on new suffix masks.  Compare the hierarchy against additive, dense low-rank, and
   unstructured baselines under the same document bootstrap.
3. **Add vector-valued outcomes**, especially selected token/logit responses, so an
   interface that hides errors inside scalar average CE cannot pass.
4. **Connect any surviving interface roles to MLP0's joint sparse dictionary.**  A
   lexical feature is useful only if its coefficients sparsely predict these
   downstream response classes and support extraction or removal.
5. **Begin the 68-action ledger.**  The next simplification should be judged not just
   by CE reconstruction but by whether it predicts, extracts, or selectively removes
   named behavior on untouched data.

The largest whole-project gaps are unchanged: strict named causal recovery remains
10.923% with 4.72714 nats unexplained, and the final extraction/removal/OOD action
ledger remains 0/68.  The current experiment is valuable because it directly tests
whether those gaps can be organized into a small composable program rather than
another locally accurate surrogate.
