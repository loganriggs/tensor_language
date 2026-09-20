# Communicating circuit results: a style guide for research updates

For the automated research program on the bilinear GPT-2 and for human write-ups derived from it. The goal is that a reader can tell, for every number, what kind of evidence it is, what it was evaluated on, and what would have falsified it.

---

## 1. Lead with the story, not the sequence

Updates currently read as logs: "we did A, then B failed, then C." Readers need the state of knowledge, not the order of operations.

- First paragraph: what the path is now, in one sentence, with its endpoints. Then what changed since the last update, in one sentence.
- One figure at the top: the path diagram (Section 5).
- The most instructive result goes second, even if it is a failure. A falsified route teaches more than three passing gates.
- Chronology, invalid runs, and corrections go in the appendix.

## 2. Tag every claim with its evidence type

There are exactly four kinds of evidence in this program, and they support different claims. Tag each number with one.

| Tag | What it is | What it can support | What it cannot support |
|---|---|---|---|
| **fold** | Exact algebraic attribution of the native computation on native prompts. No intervention. | Which terms are large in what the model currently does; native signs; exact structure. | Anything about what happens when a term is changed or removed. |
| **edit** | Causal intervention with the full recursive suffix recomputed. | Effect of removing/donating a component; sufficiency and selectivity when paired with controls. | Localization of *where* downstream the effect travels. |
| **response** | Decomposition of the model's response to an upstream edit into downstream module contributions. | Which downstream modules carry an edit's effect. | That editing the downstream module itself would have the same effect. |
| **fit** | Any regression, probe, or optimized coefficient. | Predictive relationships. | Mechanism, unless followed by a fresh edit. |

In prose: "Head 17.2 carries the response (response, opened rows)." In tables: a dedicated column. Never let a **response** result be read as an **edit** result; the current updates are careful about this in the limitations section and not careful in the summary.

## 3. State what every number was evaluated on

Every number in this program is a weight-defined quantity evaluated on a specific set of native activations. Say both parts.

- **Structure is weights; magnitudes are data.** The fold identity holds for any input; the ratio `.98` holds for these 96 rows. Write "on the 96-row panel, head 9.8 carries .98 of the aligned change," not "head 9.8 carries .98."
- **Fresh or opened.** Every result gets one of: *fresh* (rows unused at any prior stage), *opened* (rows seen during discovery or selection), *replay* (rows used to select the very thing being measured). A follow-up on opened rows is not confirmation.
- **Template shape.** State the shared structure of the prompt families in one sentence the first time they appear (for the regional task: description containing the city, colon, opening quote, quoted sentence ending at the spelling endpoint). Any positional finding must be flagged as confounded with that structure until a template-varying control has run.

## 4. Metrics: define once, use consistently

Put a metrics box near the top. Define each metric exactly once per document and link back rather than redefining.

- **Change-norm ratio**: L2 norm of a term's paired change vector over the parent's. Signless. Exceeds 1 under cancellation. Say "cancellation" once, in the box, not at every table.
- **Aligned fraction**: dot product with the parent change over the parent's squared norm. This is the signed quantity. Tables that report a norm ratio must also report the aligned fraction or a cosine.
- **Replay error**: relative L2 error when a subset stands in for the parent.
- **Transfer %**, **CE change**, and any task-specific metric: same treatment.

Never mix a nats-scale and a logit-margin-scale number in one sentence without units.

## 5. The path diagram

One diagram per update, updated in place. Conventions:

- Nodes are modules or grouped sources. A node with no incoming edge is drawn that way; do not imply an input edge that has not been traced.
- Edges are colored by evidence tag: fold, edit, response, and a dashed "falsified" style for routes tested and rejected. Keep the four colors fixed across updates.
- Backward folds are drawn as a separate arc from forward edit routes. They are different objects and readers conflate them.
- Every edge label carries one number and its evaluation set: ".53–.60 of head, fresh."
- Native states that fill gaps between traced edges are shown as gaps, not as edges.

## 6. The claim table

Each update carries a table with columns: **claim, evidence tag, fresh/opened, key numbers, status**. Status is one of: exact, passes, established, candidate, falsified, fails, not yet tested. The table accumulates across updates; falsified rows stay in it.

This table is what a reader skims. Prose should not repeat it.

## 7. Precision and language

- Two significant figures in the body. Full precision in the appendix and JSON.
- No closure receipts (1e-16, 1e-7) in the body. One sentence: "all algebraic closures pass; see appendix."
- Define terms of art on first use: carry, port, suffix, opened rows, aligned fraction. "Carry" appeared in the Sept 15 update as if standard; it is not.
- Say "attribution" for folds and "effect" for edits. Do not write "contributes" for both.
- Avoid "circuit" for anything that has not passed removal plus a preservation control against a null. Use "path," "route," "term," or "component."
- Do not summarize a failed gate as "narrowly missed." Report the number and the gate.

## 8. Separate threads, separate documents

One behavior per update. The subject-number and regional-spelling threads use different heads, different methods (one has fitted OLS coefficients, the other has none), and different datasets. Mixing them in one file makes each harder to follow and makes the nulls of one read as failures of the other. If two threads must share a document, give each its own diagram, metrics box, and claim table.

## 9. Limitations belong in the summary

The current updates put honest limitations at the end. Move the load-bearing ones into the summary paragraph: "response localization, not yet edited"; "opened rows"; "no random-group null." A reader who stops after the summary should not believe more than the evidence supports.

## 10. Appendix policy

Appendix contains: dataset manifest and example rows; per-run execution settings; gates as preregistered; invalid runs and corrections; closure values; receipts. Body contains none of it. A reader auditing the work should find everything there; a reader learning the result should never need to open it.

## 11. Checklist before publishing an update

- [ ] First paragraph states the current path and the delta since last update.
- [ ] Diagram updated; no untraced edges drawn; falsified routes shown.
- [ ] Every number has an evidence tag and an evaluation set.
- [ ] Metrics defined once; signed and unsigned quantities both reported.
- [ ] Claim table updated; falsified rows retained.
- [ ] Limitations that change the reading of the result appear in the summary.
- [ ] Template confounds named for any positional or token-level finding.
- [ ] Two significant figures in body; receipts in appendix.
- [ ] One behavior per document.
