# The aggregate experiment list (Logan, 2026-08-10)

Logan's message, verbatim, kept here for looking back on. A status ledger follows at the bottom;
it is the only part of this file that gets edited.

---

## 1. Experiments, ranked

**Tier 1: cheap, decision-relevant, directly flagged**

1. **Retest the scalar-mass collapse at bilin18.** One run: refit the layer-0 dictionary under
   w_t·‖δ‖² only (per-token scalar, no geometry) at 2–3 bit budgets, compare against the full
   context-expected metric on the 307k audit. The doc itself calls this "a prediction, worth one
   run." If it replicates, several ticks of derivation collapse to "allocate bits by exposure" and
   the anchor machinery simplifies too (together with the FINDING 13 stratified-sparsity result).
   Highest information per GPU-hour in the whole list.
2. **Ledger on the MLP's realised interface.** Decompose the third moment of the block-0 MLP
   restricted to its rank-~10-16 realised channels instead of the rank-4608 weight tensor. This is
   the best direct test of whether "content is spectral" is a fact about the model or about running
   CP in a gauge-polluted 4608-dim space. Also the main open door on Path C, the only path into
   layer-1 QK that isn't understood.
3. **Static-fraction-by-depth port profile.** Run the layer-1 port test (static mean-residual
   tables vs destruction floor) at layers 2, 3, 4... Cheap, and the decay curve is a publishable
   result on its own, plus it tells you how far the whole ledger pipeline extends before you need
   activations.

**Tier 2: calibration experiments (your controlled-DGP wheelhouse)**

4. **Planted-modular-content DGP.** Build a toy language where content provably factors into
   modular units, train a bilinear model, run the full pipeline (CP, naming hypotheses,
   equal-ablation). If the pipeline fails to recover planted modular content, the selection/content
   dichotomy is partly tool artifact; if it recovers it, "content is spectral" at bilin18 is a
   discovery. This adjudicates the doc's deepest philosophical claim, and the semantics red-team's
   shared-priors worry makes it non-optional.
5. **Archetype extraction into a standalone machine.** In a known-DGP setting: plant grammar rules,
   train, run ledger, hand-assemble the recovered archetypes into a small model, test whether it
   implements the planted grammar. Includes the specific sub-prediction that the extracted machine
   needs the shared/overlap directions to function (testing whether sub-additivity overlap is
   functional). Natural tie-in to distillation with tensor similarity losses.
6. **Planted-tree topology recovery.** DGP whose computation is hierarchical (pairwise feature
   merging with known intermediate vocabularies), then fit CP vs TT vs hierarchical Tucker with
   generator-charged MDL and per-topology transplant nulls + planted-recovery solver tests.
   Calibrates the topology-search instrument before pointing it at bilin18. Do not skip solver
   calibration; only 1 of 5 CP fitters passed last time, trees will be worse.

**Tier 3: worthwhile, less urgent**

7. **Weights-only unigram recovery.** Estimate p_t (or at least exposure ranking) from
   embedding/unembedding geometry alone; check rank correlation against corpus unigram and whether
   the frontier survives with recovered weights. If it works, layer-0 compression becomes fully
   data-free, a clean lemma for the weight-based program.
8. **Topology comparison on bilin18 folded objects.** The wild-model version of #6, only after #6
   calibrates the instrument.
9. **Windowed-unigram moment model.** Middle path for extending the ledger past where static tables
   port: moments under short-window statistics (motivated by the W=16 locality result), before
   conceding to full activations.

Dependency note: 4–6 share DGP infrastructure and probably batch; 8 is gated on 6; 2 informs
whether 9 is needed.

## 2. Dropped threads and unfinished thoughts

- **Path B was never closed.** I noted the composed layer0-attn → layer1-QK map "hasn't been named
  as such, only priced inside the adapter," and that summed head ablations understate joint
  effects. We moved on. There's a small open question: does layer-0 attention's contribution to
  layer-1 keys have its own archetype structure, or is it absorbed into the shrunk tables?
- **Head-1-as-detokenization was asserted, not designed.** I suggested "treat head 1's channel as a
  detokenization circuit and build a controlled synthetic version" and we never specced that DGP.
  It's arguably experiment #2's companion: a subword-continuation toy language with ground-truth
  merge structure.
- **The "task-AI from archetypes" question got answered only for toy settings.** Your original
  phrasing hinted at something more ambitious (extract a functional sub-model from a real trained
  model). I steered to the DGP version; whether aggregate-substitution-style extraction could yield
  a useful standalone artifact from bilin18 itself (e.g. a pure scaffold/boundary model as a
  component) was left hanging.
- **ODT as gauge-fixing was name-checked twice and never operationalized.** Both times ("your ODT
  canonicalization is a natural candidate" for the content basis, and gauge-fixing in the topology
  discussion) I pointed at it and moved on. The concrete unfinished thought: run ODT
  canonicalization on the rank-16 interface bases or the folded layer-0 objects and check whether
  the canonical gauge improves purity/codability scores relative to §34's failures. That's a
  specific, runnable version of "maybe content has modular structure in a basis nobody tried."
- **The offset-averaged rotary loss.** The doc mentions the coherent form "washes out 98.8% of the
  systematic signal and loses"; we never discussed why, and it might matter for anyone
  reimplementing the objective. Minor, but it's a place where your conceptual debt survey has a gap.
- **The T≈512 degradation.** Unnormalised score-product attention falling apart past T=512 was
  noted as an architectural constraint and never revisited. It's a real architecture question for
  the tensor-transformer program (does your 500M open-webtext model share it? is there a
  normalization-free fix?), separate from the interp thread but sitting right next to it.
- **Whether "fully interpreting" is even the right success criterion** got a proposed answer
  (priced substitution + named selection + exact spectra as content description = the mature form)
  but you never said whether you buy it. Since it reframes your original question ("how do we fully
  interpret the first two layers"), it deserves an explicit accept/reject, ideally before writing
  any of this up, because it changes what the writeup claims.

---

## Status ledger (edited as work lands)

| # | experiment | status |
|---|---|---|
| 1 | scalar-mass collapse retest at bilin18 | **DONE** (RESULTS_l0_mdl.md §35): collapse mostly replicates — scalar carries 91–101% of the metric's gain; geometry's sliver is real (5.3 SE) only at the tight budget. P1 refuted, P2 holds. |
| 2 | ledger on the MLP's realised interface | **DONE** (RESULTS_l0_mdl.md §37): weight-space null-tie WAS gauge pollution — 32/36 channels beat the null, median margin 0.24; but only 21% of components nameable (M2 refuted). Path C open. |
| 3 | static-fraction-by-depth port profile | **DONE** (RESULTS_l0_mdl.md §36): 98→93→76→86→60→37→15% then NEGATIVE at 17; all four predictions held; static form works through layer 4, dead by 9 |
| 4 | planted-modular-content DGP | queued (batch with 5, 6) |
| 5 | archetype extraction into a standalone machine | queued (batch with 4, 6) |
| 6 | planted-tree topology recovery | queued (batch with 4, 5) |
| 7 | weights-only unigram recovery | queued (tier 3) |
| 8 | topology comparison on bilin18 folded objects | gated on 6 |
| 9 | windowed-unigram moment model | gated on what 2 finds |
