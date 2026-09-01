# Plain-English plan — 2026-09-01 02:21 UTC

(Damage is extra next-token cross-entropy above the native model; **lower is better**.  “Stored values”
below means every scalar needed by a standalone program, counted once.)

## Our goal

We are trying to compile the 545,902,902-scalar bilin18 language model into a smaller explicit tensor
program.  Success requires four properties simultaneously:

1. **Predictive:** low error on the full held-out census, fresh documents, and shifted text distributions.
2. **Composable:** independently useful replacements continue to work when installed together.
3. **Manipulable:** named removals and edits have the same signed causal effects as in the native model.
4. **Literally simpler:** the complete standalone dependency graph uses fewer stored values and preferably
   less executed compute; hidden native modules, fit-only state, and large lookup tables cannot disappear
   from the bill merely because a hook overwrites their outputs.

The present milestone is therefore not “find another low-CE hook.”  It is to produce the first corrected
point whose behavior and *standalone* price are both verified, or to preserve the falsification and move to
the component family with the largest real compression opportunity.

## What the corrected experiments established

The earlier `+0.052` attention floor was an instrument artifact caused by `a1v`, a context-blind block-1
value table.  Restoring block 1's native value map removed essentially the whole floor.  Three physical
configurations were then separated rather than inferred by subtraction:

- corrected ct96: census `+0.0034195`, `56/62` certificates;
- contiguous top96: census `+0.00853845`, `52/62` certificates;
- true mixed104, using singular indices `{0..95,120..127}`: census `+0.00469196`, `54/62` certificates,
  with eight fresh-window damages in `[-0.0066,+0.0032]`.

The smallest eight singular directions therefore buy `0.0038465` nats and two certificates over top96.
This is a real spectral result, because the live index-set and factor-width tripwires passed.

The causal falsifier also passed.  Under the same signed a16 mean ablation, mixed104 and native effect
vectors had cosine `0.995879`, normalized error `0.096599`, norm ratio `1.028722`, collateral circuit
Spearman `0.997959`, and own-family median magnitude ratio `1.034785`.  Thus the old a16 anomaly was also
path contamination.  Mixed104 is behaviorally faithful on the tests run so far.

The compute-sparse companion was informative but not adoptable as the certificate base.  Per-token
top-1152 MLP execution added `+0.015464` census damage, confirming the old additive compute law, but reduced
certificates from 52 to 26.  It remains a valid 4x MLP-compute tier, not a certificate-preserving default.

## The exact-price correction

The historical `180M` mixed price and its corrected `123.4M` transport are not standalone program sizes.
They are replacement/coverage ledgers.  The evaluation harness loads and executes the native model, then
overwrites selected outputs.  A semantic deployment audit must remove overwritten operations while keeping
every native tensor still called by the resulting graph.

The exact audit found 38 unique motif heads in blocks 2–9 and all 72 heads in blocks 10–17, or 110 replaced
heads.  Four Q/K maps per head at width 104 require

$$
110\cdot4\cdot(128+1152)\cdot104=58{,}572{,}800
$$

factor scalars.  But those factors are only one line of the bill.  The standalone graph also retains two
untied vocabulary matrices, all eighteen full-width MLPs, all value/output maps except the overwritten
block-0 value map, the unreplaced Q/K head rows, and the learned mixing scalars.

There are consequently two honest totals:

| object | stored scalars | raw tensor bytes | comparison with native |
|---|---:|---:|---:|
| Native checkpoint | 545,902,902 | 2,067,669,612 | reference |
| Physically tested mixed104 artifact, including its fp16 block-0 token-value table | 596,164,022 | 2,152,921,964 | **50,261,120 scalars larger** |
| Storage-minimal mixed104 candidate, computing block-0 values with native `c_v0` | 539,595,062 | 2,042,438,252 | **6,307,840 scalars smaller** |

The first row is the true baseline.  The second is the literal object whose behavioral receipt we already
have, and it is dominated by native on both storage and prediction.  It cannot be adopted.  The third is a
promising implementation candidate, but it is not silently promoted: the tested table is fp16, whereas
online `c_v0` computation is fp32 from the bf16 embedding.  The two are algebraically related but not
bit-identical, so the 539.60M point needs its own live equivalence, census, and OOD receipt.

This also changes how to read the old frontier.  Its prices remain useful *incremental replacement costs*,
but they are not evidence that a 123M or 180M standalone model exists.  No historical point will be compared
with a literal total until its own dependency graph is rebilled under the same convention.

## The executing plan

### 1. Test the only literal mixed104 candidate that can beat native

Replace the block-0 fp16 lookup with online native `c_v0`, while keeping the exact mixed104 Q/K index sets
and every other corrected component fixed.  The first gate directly measures table-versus-online state and
logit differences; it must also assert that `a1v` and all class/dictionary tail hooks remain absent.

The candidate advances only if its census damage stays at most `0.0065`, at least `54/62` certificates
survive, and fresh windows remain below `0.020`.  Because online `c_v0` removes a quantized approximation,
slightly better fidelity is plausible, but it is a prediction rather than an assumption.  If it holds, the
literal point is `{539.595M scalars -> about +0.0047 damage, 54 certificates}`: a modest 1.16% scalar
compression, not the formerly claimed 77% compression.

### 2. Run shifted-corpus OOD before adoption

The winning implementation is then evaluated at high precision on disjoint source documents and at least
one genuinely shifted corpus.  The split and bars are frozen before execution.  This is the remaining
predictive gate because the current “fresh” windows come from a different region of the same Pile source,
not a different distribution.

### 3. Expand causal coverage only after price and OOD survive

The direct a16 signed test is strong but singular.  A broader battery should include both attention and MLP
sites, measuring effect-vector cosine, normalized error, magnitude ratio, and collateral circuit ranking.
The same intervention statistic must be computed inside each model; no unsigned aggregate subtraction is
allowed.  This step decides whether the small literal candidate is manipulable generally rather than only
at the hardest previously known attention site.

### 4. Adopt narrowly or preserve the falsification

If online-`c_v0` mixed104 passes equivalence, OOD, certificates, and broader causal gates, register it as the
first literal standalone Pareto point.  Its claim will be deliberately narrow: roughly 1.16% fewer scalars,
not a 123M program.  If any gate fails, preserve the receipt and do not return to rounded price transport.

## Different paths suggested by the corrected bill

The exact bill changes the priority order.  Uniform Q/K rank work has already removed only 6.31M scalars;
the dominant storage is elsewhere.

1. **Compress the bilinear MLP tensor directly.**  The eighteen Left/Right/Down maps occupy 286.67M
   scalars, over half the model.  Per-token top-k changes compute but stores the full maps.  The next serious
   storage route should seek a shared CP/Tucker/tensor-train basis across units and layers, selected under
   held-out CE and signed causal response rather than weight norm.  A 25% reduction here is about 72M
   scalars—more than eleven times the entire mixed104 Q/K saving.  Kill criterion: no held-out Pareto gain
   after composition, or certificate loss exceeds the storage-price rule.
2. **Jointly factor the untied vocabulary maps.**  Input embedding plus output head occupy 115.90M scalars.
   Test whether a shared vocabulary code with two small readout maps, plus a sparse residual for exceptional
   tokens, preserves logits and token-conditioned block-0 values.  This is mathematically different from a
   token lookup replacement because it compresses two globally required matrices together.  Kill criterion:
   shifted-corpus rare-token loss or logit-tail error dominates at equal price.
3. **Use causal-response coordinates for rank allocation.**  Mixed104 proves that smallest weight singular
   directions can matter.  Allocate Q/K and MLP directions by preserved signed intervention/circuit response
   per scalar, with held-out circuits and texts.  This may beat uniform rank even when ordinary SVD cannot.
   Kill criterion: no price-matched gain over mixed104 on both prediction and certificates.
4. **Compile the predictive causal state rather than modules.**  Construct a quotient of residual states
   identified by future logits and controlled interventions, using prefix/continuation Hankel or observable-
   operator rank to estimate the necessary state dimension.  This can remove redundancies spanning several
   native layers instead of compressing each weight matrix separately.  Kill criterion: the numerical rank
   grows with data, shifts across corpora, or fails to preserve intervention transport.
5. **Build executable error contracts and lower bounds.**  Attach local approximation bounds to each tensor
   contraction and propagate them through the residual stream; compare the empirical storage frontier with
   predictive-state and information lower bounds.  This tells us whether a large compression is structurally
   plausible before spending GPU time on another local sweep.  Kill criterion: the bounds are too loose to
   predict census/certificate failures or already meet the current measured bill.

The first path is now the highest-upside engineering direction, while the predictive-state quotient is the
highest-upside research direction.  More uniform Q/K rank sweeps are low priority unless the causal-response
coordinate experiment supplies a new basis.

## Decision order

1. live online-`c_v0` equivalence and corrected mixed104 census/certificate gate;
2. high-precision fresh and shifted-corpus OOD;
3. broader signed causal battery;
4. narrow adoption at the exact 539,595,062-scalar bill, or preserved falsification;
5. in parallel with safe CPU design work, preregister the first MLP-storage and joint-vocabulary screens.

At each elapsed hour the research driver restates the predictive/composable/manipulable/literal goal,
audits config identity and hidden native dependencies, and re-ranks these paths.  The exact-price correction
is not a pause: it is the evidence that redirects the next work toward the tensors where a genuinely small
program can still be won.
