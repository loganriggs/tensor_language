# What it would take to have better circuits

Strategy note for the bilinear GPT-2 program. Premise: we have many partial circuits over many datasets; the marginal value of another partial path is low. The plan is depth on one path until it meets a fixed definition of done, using weight information wherever the model's structure allows, and treating panel-conditioned attribution as candidate generation rather than evidence.

---

## 1. Definition of done

A component is a circuit when all of the following hold, each against a preregistered gate and a null:

| Property | Test | Null |
|---|---|---|
| **Simple** | Description length in a stated basis: number of readers, products, writers, and the size of any lookup tables. Reported as a count, not an adjective. | Same count for a random component of matching effect size. |
| **Predicts OOD** | Frozen formula predicts effect on rows with new templates, new cities, new endpoints, ideally new corpus. Strong form: input is token IDs, not native states. | Effect predicted by a constant, and by a fit on the training panel. |
| **Extracted** | Runs standalone with declared inputs. Port count (native states still required) reported and decreasing over time. | n/a; report the count. |
| **Selective** | Removal changes the target; unrelated readers (at least three) move less than gate, on fresh rows. | Random equal-norm direction removed at the same site. |
| **Composes** | Möbius interaction terms among the component's pieces are below gate relative to the smallest piece's effect; joint effect predicted from singles. | Interaction magnitude for random splits of the same write. |

Until a component clears all five it is a path, term, or route. Most current results are exact folds or single edits; only the even-key 8.2/9.8 producer has cleared removal plus one preservation control, and it has no null.

## 2. What is missing from the regional path today

The regional UK/US path is the depth target. Its traced edges are real; the gaps between them are filled with native states. Closing gaps in this order:

1. **Head 9.8's QK2.** Held native throughout, yet QK2 controls are .7–.8× the QK1 edit. Split it by source module the way QK1 was split; run the grouped-block census and a fresh routing edit. Without this the head's routing is half explained.
2. **How the late group computes from tokens.** The 289-term census split the carry by writing module, not by source token or by what those modules read. Fold attention 5 and MLPs 5–7 one more step backward: for MLPs, the exact quadratic form in their inputs; for attention 5, its QK×V product by source position. Target: an edge from the city token into D.
3. **Head 17.2 as an edit, not a response.** Freeze it, split its QK1×QK2×V product into earlier-source self and cross terms, run a fresh selective edit with the three-reader control battery.
4. **The direct residual route.** The 54% "direct" response lumps head 9.8's write read by the unembedding with MLP 9–16 responses. Decompose it exactly with the per-layer response formula; report the vector.
5. **Template controls for the framing interface.** Run the same rows with the colon removed, with the quote removed, and with the quoted sentence moved. If the offset-0/−1 finding survives, it is a mechanism; if it moves with the boundary, it is positional.
6. **Normalizers.** Every RMS denominator is native. For each traced edge, report the effect of freezing versus carrying the denominator so the executor's dependence on them is quantified.

## 3. Use the weights as much as the architecture allows

Every nonlinearity in this model is a product. That makes several things exact that are approximate elsewhere, and the program should lean on them harder than it currently does.

### 3.1 Prefer weight-only objects; report which parts are data
Every fold has a weight part (the terms) and a data part (their magnitudes on a panel). Separate them in the code and the write-up. The weight part is reusable across datasets and is where any OOD claim must come from. Panel-conditioned ratios nominate candidates and nothing more.

### 3.2 Select by causal effect, not by weight magnitude
Truncating the MLP8 eigenspectrum by |λ| failed on the near-quote construction. Rank candidate terms by their effect under intervention at calibrated scale (the DCT causal-importance criterion), or at minimum by their aligned fraction on a fresh panel, never by norm in weight space alone.

### 3.3 Folds nominate, edits decide
A backward fold finds terms large in the native computation and gives native signs. The route an edit's effect takes forward is a different object: it includes the residual skip, which is never a term of any MLP's bilinear form, and each downstream layer's response `2H₂(δ, z)`, whose sign is not fixed by the native term. Rule: no suffix is proposed without a forward response census (one edited forward, decomposed into module responses) run first.

### 3.4 Compute the first downstream layer's response exactly
For an edit `z → z − aw` at the input of a bilinear MLP, the response is exact: `Δ(a) = −aw + (ρ₀²/ρₐ² − 1) m₀ − (a/ρₐ²) J_w z + (a²/2ρₐ²) J_w w`. Use it before any panel measurement of the same quantity. It gives signs, the normalization contribution, and the four-vector span of all responses at that layer.

### 3.5 Keep the product terms; never analyze QK1 and QK2 separately
An edit to QK1 and an edit to QK2 interact even if each is individually additive. All routing analyses expand the full `QK1 ⊙ QK2 ⊙ V` product with both edits in every slot.

### 3.6 Test nonadditivity at its source
The five-arm design (install both single deltas at the head without their product; subtract from the true joint) isolates a head's own bilinear cross term from downstream curvature. Use it whenever two branches meet in a head. It is the direct test of the composition property and is cheap.

### 3.7 Token-only generators wherever the architecture provides one
The first-layer value branch of every head (`μ V₀ e_s`) depends only on the token. Build and store those tables for every head on the path. Each one is a closed port. The MLP0 output is likewise a fixed function of the token; fold it.

### 3.8 Close ports by folding, not by fitting
When an edge needs a native state, the next step is to express that state as an exact sum of upstream writes and fold one more step, not to fit a proxy. The subject-number thread showed that a fitted proxy can transfer well on its own and still fail after a high-gain multiplication. Fits are for hypothesis generation; ports close only by exact folding or by a token-only generator.

### 3.9 Open the context slot instead of averaging over prompts
For any second-order object (DCT Hessian, cross term, response kernel), the background state z enters polynomially. Keep z as an explicit slot (weights contracted with z a fixed number of times) rather than averaging over prompts; averaging kills sign-changing terms, and the near-quote reversal was one. One-layer slices are exact in this form; deeper slices need a stated truncation in a stated metric.

## 4. Nulls and controls that are currently missing

Run these before any further attribution is reported:

- **Random-group null** for the "compact at grouped grain" result: the same four-block census for 20 random 4-module groups drawn from the 17 sources, and for the 4 latest modules regardless of identity.
- **Random-direction null** for every preservation control: remove an equal-norm random direction at the same site; report the control-reader movement distribution.
- **Random-reader null** for mixed-term dominance: cross-term share of cue contrast for 100 random reader directions versus the selected reader.
- **Template controls** as in Section 2.5.
- **Preservation battery** of at least three unrelated readers with gates registered against fixtures, applied to every component before it is called selective.

## 5. Panel hygiene

- Every panel is labeled fresh, opened, or replay relative to every selection step, and the label is carried into the result JSON.
- Rows selected on opened panels (framing offsets, late group D, head 17.2) are candidates until a fresh edit passes.
- New templates must vary the structural feature under test. Two more templates with the same colon-quote boundary do not test the boundary.
- Effective sample size is the number of distinct (template, city pair) cells, not the row count. Report both.

## 6. Depth plan for the regional path

Ordered by information per GPU-second; each step has a fold part and an edit part.

1. Forward response census for every existing edit that lacks one (QK2 controls, value edit, even-key producers). Fold: none. Edit: one forward each.
2. QK2 source split for head 9.8 and grouped-block census. Fold: exact. Edit: fresh routing removal with the three-reader battery and the random-direction null.
3. Head 17.2 fold and fresh edit (Section 2.3).
4. Late-group backward fold one step: exact quadratic forms of MLPs 5–7 in their inputs; attention 5 by source position. Edit: donate the city-token contribution to D on fresh rows.
5. Template controls for the framing interface.
6. Rebuild the executor with every closed port removed from its input list; publish the port count and the standalone parameter count. Compare to the previous executor.
7. Only then: attempt compression of any remaining weight-only term (shared readers, grouped bases), selected by causal effect and tested by fresh edit.

## 7. What not to spend time on

- New behaviors or new datasets until the regional path meets the definition of done or a step above fails irrecoverably.
- Compressing four readings to one, or any scalar collapse, before the inputs to those readings are traced.
- Fitted proxies as components. They may guide the next fold; they are not circuits.
- Improving replay error on opened rows. Small local error on the same examples does not move any of the five properties.
